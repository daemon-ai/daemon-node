//! `daemon-api` — the node's one canonical interface/surface.
//!
//! A node exposes a single operable surface; *how* a caller reaches it (in-process call, a Unix
//! socket, TCP, JSON-RPC, the C FFI buffer pump) is a transport detail. This crate defines that
//! surface **once** and is the invariant every transport binds to:
//!
//! - the **interface** — [`SessionApi`] (the §17 per-session surface: submit a command, drain
//!   outbound items, answer a host request) and [`ControlApi`] (node/operator ops: health, stats,
//!   sessions, assign, cancel, fleet), composed as [`NodeApi`];
//! - the **serializable mirror** — [`ApiRequest`]/[`ApiResponse`], a 1:1 reflection of the
//!   interface that every *non*-in-process transport marshals (CBOR, `wire_version`-governed; see
//!   `daemon-api.cddl`);
//! - the shared **[`dispatch`]** — decode a request, call the interface, encode the response. The
//!   in-process transport skips the mirror and calls the trait directly; the socket and the FFI run
//!   the *same* `dispatch`, differing only in how the bytes arrive.
//!
//! It is a pure contracts crate (no runtime, no substrate): it depends only on `daemon-common` and
//! `daemon-protocol`, and uses decoupled DTOs ([`HealthReport`], [`StatsReport`], [`SessionInfo`],
//! [`FleetReport`]) so the surface never drags the substrate's concrete types into the contract.

#![forbid(unsafe_code)]

use async_trait::async_trait;
pub use daemon_common::{
    Author, BlobRef, ByteRange, Revision, RevisionKind, SkillBundle, WorkspaceBinding,
};
use daemon_common::{
    ContentHash, DownloadId, DownloadStatus, GgufInfo, InstalledModel, ModelEngine, ModelFile,
    ModelId, ModelRef, ProfileRef, QuantRecommendation, QuantizeId, QuantizeStatus, SearchPage,
    SearchQuery, SessionId, UnitId, UsageDelta, WireVersion,
};
use daemon_protocol::{
    session_id_for, AgentCommand, DeliveryTarget, HostResponse, IsolationPolicy, Origin,
    RewindAnchor, TranscriptBlock, TransportId, UserMsg,
};
pub use daemon_protocol::{Outbound, SessionLogEntry};
use futures::stream::{self, BoxStream, StreamExt};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod profile;
pub use daemon_common::{SkillCreator, SkillState, SkillUsage};
pub use profile::{
    BoundAccount, BudgetSpec, ContextEngineSel, CredentialInfo, CuratorChange, CuratorEntry,
    Distribution, EngineTunables, MemoryProviderSel, ModelDescriptor, ProfileInfo, ProfileSpec,
    ProviderSelector, SessionOverlay, ToolsOverride,
};

/// One item of a [`LogStream`]: either a merged-log entry, or a `Lagged` signal that the live
/// broadcast dropped entries for a slow consumer (L2 resync). A transport that can re-baseline
/// (the socket mux) maps `Lagged` to a `Reset` frame so the client re-reads from the durable
/// journal; a transport that cannot simply skips it (the prior silent-drop behavior).
// `Entry` is intentionally inline (not boxed): it is the hot path - one item per streamed log
// entry - and the broadcast already owns/clones a `SessionLogEntry`, so boxing would add an
// allocation per event for no benefit. `Lagged` is rare.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum LogStreamItem {
    /// A merged-log entry in `seq` order.
    Entry(SessionLogEntry),
    /// The broadcast lagged; entries were missed and the consumer must re-baseline.
    Lagged,
}

/// A live, push-based stream of merged-log items (inbound + outbound), the delivery shape a
/// streaming transport (in-process, socket, HTTP/WS) returns from [`SessionApi::subscribe`].
/// Streaming is a *transport capability*, not a wire-mirror variant: the cursor read
/// ([`SessionApi::log_after`] / [`ApiRequest::Subscribe`]) is the one-shot/long-poll form every
/// transport marshals.
pub type LogStream = BoxStream<'static, LogStreamItem>;

/// A live, push-based stream of node-wide [`EventsPage`]s (L3 `EventsSince` feed), the delivery
/// shape a streaming transport returns from [`ControlApi::events_subscribe`]. Each page carries a
/// batch of payload-free [`NodeEvent`] pointers plus the feed cursors; one-shot transports poll
/// [`ControlApi::events_page`] over the same cursor instead.
pub type NodeEventStream = BoxStream<'static, EventsPage>;

/// The wire version of the api mirror (rides every framed request/response; governs evolution).
pub const API_WIRE_VERSION: WireVersion = WireVersion::CURRENT;

/// The per-session edit-approval **session mode** a caller selects ([`SessionApi::set_session_mode`]),
/// the wire mirror of `daemon-core`'s `ApprovalPolicy`. Mirrors hermes' Default / Accept-Edits /
/// Don't-Ask session modes (`acp_adapter/server.py` `_session_modes`) plus an explicit hard deny:
///
/// - [`Ask`](ApprovalMode::Ask): ask before every gated action (the host parks/suspends for a human);
/// - [`AcceptEdits`](ApprovalMode::AcceptEdits): auto-allow ordinary workspace edits, still ask for
///   *sensitive* paths and dangerous commands;
/// - [`AutoAllow`](ApprovalMode::AutoAllow): auto-allow every gated action except sensitive paths;
/// - [`Deny`](ApprovalMode::Deny): reject every gated action outright.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    /// Ask before every gated action (hermes "Default").
    #[default]
    Ask,
    /// Auto-allow workspace edits, still ask for sensitive paths / commands (hermes "Accept Edits").
    AcceptEdits,
    /// Auto-allow every gated action except sensitive paths (hermes "Don't Ask").
    AutoAllow,
    /// Reject every gated action outright.
    Deny,
}

impl ApprovalMode {
    /// The stable wire id for this mode (the value GUIs advertise / select).
    pub fn as_str(self) -> &'static str {
        match self {
            ApprovalMode::Ask => "ask",
            ApprovalMode::AcceptEdits => "accept_edits",
            ApprovalMode::AutoAllow => "auto_allow",
            ApprovalMode::Deny => "deny",
        }
    }

    /// The full set of advertisable modes (for a GUI mode picker / session-state advertisement).
    pub const ALL: [ApprovalMode; 4] = [
        ApprovalMode::Ask,
        ApprovalMode::AcceptEdits,
        ApprovalMode::AutoAllow,
        ApprovalMode::Deny,
    ];
}

/// An **in-process** outbound delivery sink: where the host pushes a session's outbound entries for
/// a transport instance to *post* (daemon-event-io-spec §5.9.3, the push half of the subscriber
/// model). A transport registers one keyed by its instance [`TransportId`] with the host; the host's
/// per-session event pump resolves the session's current [`DeliveryTarget`]s and calls
/// [`deliver`](DeliverySink::deliver) on the sink owning each — so a demoted `Primary` stops
/// receiving and a new one starts, with no work in the adapter.
///
/// This is deliberately **not** part of the wire [`ApiRequest`] surface: a sink is a live trait
/// object that cannot cross a process boundary, so cross-process transports use the pull path
/// instead ([`SessionApi::delivery_sessions`] + [`SessionApi::subscribe`]). Projection of the rich
/// [`SessionLogEntry`] stream down to a concrete message stays the sink's job (adapter-owned policy).
#[async_trait]
pub trait DeliverySink: Send + Sync {
    /// Post one outbound `entry` to `target` (the sink's transport instance). Called from the
    /// session's pump task, so a slow sink backs up only its own session's delivery.
    async fn deliver(&self, target: DeliveryTarget, entry: SessionLogEntry);
}

// ---------------------------------------------------------------------------
// The interface (two sub-surfaces, one node surface)
// ---------------------------------------------------------------------------

/// The §17 per-session surface: drive one interactive engine session.
#[async_trait]
pub trait SessionApi: Send + Sync {
    /// Submit a §17 command to `session` (opening it on the first `StartTurn`).
    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError>;

    /// Submit a §17 command attributed to a specific [`Origin`], so the per-event provenance ("from
    /// the Telegram user" vs "steered via the GUI") is recorded on the merged session log rather than
    /// collapsed to the host-local default. Default: drop the attribution and delegate to
    /// [`Self::submit`] (a transport that does not carry origins keeps working unchanged).
    async fn submit_from(
        &self,
        session: SessionId,
        _origin: Origin,
        command: AgentCommand,
    ) -> Result<(), ApiError> {
        self.submit(session, command).await
    }

    /// Submit a command, optionally binding the session to an explicit `profile` on open (the "open
    /// chat as agent X" seam). When `profile` is `Some`, the host binds it (sticky on first open,
    /// the same path [`Self::submit_routed`] uses) before submitting; `None` keeps the routing-config
    /// / default binding. Default: ignore `profile` and delegate to [`Self::submit_from`] /
    /// [`Self::submit`] (a host with no profile binding keeps working).
    async fn submit_as(
        &self,
        session: SessionId,
        origin: Option<Origin>,
        command: AgentCommand,
        _profile: Option<ProfileRef>,
    ) -> Result<(), ApiError> {
        match origin {
            Some(origin) => self.submit_from(session, origin, command).await,
            None => self.submit(session, command).await,
        }
    }

    /// Submit a command by handing the host only an [`Origin`] (no caller-chosen `SessionId`): the
    /// host's routing capability (daemon-event-io-spec §5.9) resolves the origin to a session, the
    /// profile that runs it, and where its replies post, then opens/binds the session and submits.
    /// Returns the derived [`SessionId`] so the caller can `subscribe`/`poll` it. This is the seam a
    /// chat transport (or any multi-tenant surface) uses instead of deriving the id itself.
    ///
    /// Default (a host with no routing registry): derive the id with [`IsolationPolicy::PerThread`]
    /// and delegate to [`Self::submit_from`], so a transport gets correct naming + attribution even
    /// before the host opts into agent-selection routing.
    async fn submit_routed(
        &self,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<SessionId, ApiError> {
        let session = session_id_for(&origin, IsolationPolicy::PerThread);
        self.submit_from(session.clone(), origin, command).await?;
        Ok(session)
    }

    /// Drain up to `max` outbound items (events + raised host requests) for `session`. A
    /// **destructive, single-consumer** convenience (each call consumes what it returns), for the
    /// FFI / MCP lowest-common-denominator only — NOT the multi-surface basis (a drain is inherently
    /// one-reader; cf. the unified cursored-stream contract in
    /// `daemon-core/docs/daemon-event-io-spec.md` §5.4.1). Multi-surface observers use the
    /// non-destructive cursored [`Self::log_after`] (live) / [`Self::session_history`] (durable).
    async fn poll(&self, session: SessionId, max: u32) -> Result<Vec<Outbound>, ApiError>;

    /// Answer a host request the session raised (matched by `response.request_id`).
    async fn respond(&self, session: SessionId, response: HostResponse) -> Result<(), ApiError>;

    /// Non-destructive, cursor-paged read of a session's **durable** verifiable history (decoded
    /// finished chat blocks + lifecycle records, with a per-entry `verified` flag) for reconnect /
    /// scroll-back. Complements [`Self::poll`] (which destructively drains the *live* delta stream):
    /// repeated reads from the same `after_cursor` return the same page. Default: an empty page (a
    /// transport with no durable journal, e.g. an ephemeral session-only FFI).
    async fn session_history(
        &self,
        _session: SessionId,
        _after_cursor: u64,
        _max: u32,
    ) -> JournalPageView {
        JournalPageView::default()
    }

    /// Non-destructive, cursor-paged read of the **merged live session event log** (inbound +
    /// outbound [`SessionLogEntry`] items past `after_seq`). Unlike [`Self::poll`] (a destructive
    /// single-consumer drain), this is the multi-surface observability read: the entry `seq` *is* the
    /// cursor, so N consumers each keep their own position and never steal each other's events.
    /// Repeated reads from the same `after_seq` return the same page; it is the long-poll/one-shot
    /// basis a streaming transport pages over. Default: an empty page (a transport with no live log).
    async fn log_after(
        &self,
        _session: SessionId,
        _after_seq: u64,
        _max: u32,
    ) -> Result<LogPageView, ApiError> {
        Ok(LogPageView::default())
    }

    /// Subscribe to the merged live session event log as a push stream of entries with `seq >
    /// after_seq` (`after_seq = 0` backfills from the start, then continues live). This is the
    /// push delivery a streaming transport (in-process, socket, HTTP/WS) holds open; the one-shot
    /// transports long-poll [`Self::log_after`] over the same cursor instead. Default: an empty
    /// stream (a transport that exposes no live log — callers fall back to `log_after`).
    async fn subscribe(&self, _session: SessionId, _after_seq: u64) -> Result<LogStream, ApiError> {
        Ok(stream::empty().boxed())
    }

    /// The session-activation generation of the live merged log (L2 resync), so a streaming
    /// transport can stamp the `epoch` on its pushed pages and on a `Reset`. Matches
    /// [`LogPageView::epoch`] for the same session. Default `0` (a transport with no live log /
    /// single-incarnation seam).
    async fn log_epoch(&self, _session: SessionId) -> u64 {
        0
    }

    /// The session's current outbound [`DeliveryTarget`]s — where its replies post (the `Primary`)
    /// and any passive `Spectator`s. A session property populated from the opening [`Origin`]; the
    /// authoritative source for "who receives the reply." Default: empty (a transport that does not
    /// track delivery targets).
    async fn delivery_targets(&self, _session: SessionId) -> Vec<DeliveryTarget> {
        Vec::new()
    }

    /// The live sessions a transport *instance* currently owns for delivery — every session whose
    /// `Primary` [`DeliveryTarget`] names `transport` (daemon-event-io-spec §5.9.3). This is the
    /// owned-session discovery primitive a transport calls on (re)connect to find which sessions it
    /// must resume posting for, without having tracked their ids itself (the reconnect-safe seam the
    /// `submit_routed`-returns-an-id path alone cannot cover after a restart). Default: empty (a
    /// transport with no live delivery state).
    async fn delivery_sessions(&self, _transport: TransportId) -> Vec<SessionId> {
        Vec::new()
    }

    /// Re-point a session's `Primary` reply sink to `target` (the single explicit "handover" op): the
    /// new target becomes `Primary` and any prior `Primary` is demoted to `Spectator`. Auth is
    /// single-tenant local-trust for v1 (any authenticated local client may hand over). Default:
    /// unsupported (a transport that does not track delivery targets).
    async fn handover(&self, _session: SessionId, _target: DeliveryTarget) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("handover".into()))
    }

    /// Record an observability-only transport/meta event ([`Disposition::Transport`]) on the merged
    /// log — presence, a surface attaching/detaching, a delivery receipt. It is fanned out to
    /// subscribers but never enters the prompt or the journal (cache-safe by construction). Default:
    /// accept and drop (a transport with no live log to record onto).
    async fn record_meta(
        &self,
        _session: SessionId,
        _origin: Origin,
        _kind: String,
        _body: Vec<u8>,
    ) -> Result<(), ApiError> {
        Ok(())
    }

    /// Switch a session's model (and optionally its provider) — a per-session override recorded on
    /// the session's [`SessionOverlay`] (the bound profile is not mutated). The override is
    /// **persisted** as host-level session metadata and applied to the running engine in place when
    /// the session is resident (swapped at the next turn boundary so an in-flight turn's prompt
    /// cache is preserved); a non-resident/durable session picks it up at its next (re)hydration,
    /// so the switch survives a restart. Default: unsupported. `provider = None` keeps the profile's
    /// current provider. A convenience wrapper over [`set_session_overlay`](Self::set_session_overlay).
    async fn set_session_model(
        &self,
        _session: SessionId,
        _model: String,
        _provider: Option<ProviderSelector>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("set_session_model".into()))
    }

    /// Set a session's edit-approval [`ApprovalMode`] (the §12 session mode) — a per-session override
    /// recorded on the session's [`SessionOverlay`] (the bound profile is not mutated). It governs
    /// how a gated tool action (an fs edit, a dangerous shell command) is serviced: auto-allow,
    /// deny, or ask (the host parks for a human on the live path or suspends the turn on the durable
    /// path). **Persisted** + applied in place when resident; restored on rehydration. Default:
    /// unsupported. A convenience wrapper over [`set_session_overlay`](Self::set_session_overlay).
    async fn set_session_mode(
        &self,
        _session: SessionId,
        _mode: ApprovalMode,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("set_session_mode".into()))
    }

    /// Replace a session's whole [`SessionOverlay`] — the unified per-session override surface
    /// (model / provider / tool allowlist / approval mode) layered on top of the bound profile. The
    /// overlay is **persisted** as host-level session metadata, so every override is restored on
    /// rehydration rather than lost on restart. What can be applied to a resident actor in place
    /// (model/provider/approval) is; a tool-allowlist change takes effect at the next (re)hydration
    /// (the live tool registry is fixed for an actor's lifetime). [`set_session_model`](Self::set_session_model)
    /// and [`set_session_mode`](Self::set_session_mode) are field-scoped conveniences over this.
    /// Default: unsupported (a transport with no live session overlay store).
    async fn set_session_overlay(
        &self,
        _session: SessionId,
        _overlay: SessionOverlay,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("set_session_overlay".into()))
    }
}

/// The node/operator control surface: inspect and steer the running node.
#[async_trait]
pub trait ControlApi: Send + Sync {
    /// The resident-service tree health.
    async fn health(&self) -> HealthReport;

    /// Durable queue depths + session/active counts.
    async fn stats(&self) -> StatsReport;

    /// A point-in-time telemetry [`TelemetryDump`] (folded usage + cost + events + health + queue
    /// depths) — the operator/GUI "live HUD" read. The default projects [`Self::stats`] +
    /// [`Self::health`] (no separate event counter); a node with a resident metrics aggregator
    /// overrides it to surface the folded event count and aggregator usage.
    async fn telemetry(&self) -> TelemetryDump {
        let stats = self.stats().await;
        TelemetryDump {
            usage: stats.usage,
            events: 0,
            healthy: self.health().await.all_ok,
            pending_jobs: stats.pending_jobs,
            pending_wakes: stats.pending_wakes,
            sessions: stats.sessions,
            active: stats.active,
        }
    }

    /// The roster's first page at [`SessionScope::TopLevel`] (back-compat convenience). Prefer
    /// [`Self::sessions_query`] for scoping/pagination. The unified roster surfaces *both* durable
    /// and live-interactive sessions; the default delegates to [`Self::sessions_query`].
    async fn sessions(&self) -> Vec<SessionInfo> {
        self.sessions_query(SessionQuery::default()).await.sessions
    }

    /// The scoped, paginated roster — the GUI inbox / per-agent / per-transport views. Unifies
    /// durable `session_record` rows with live-interactive submit/poll chats, filtering subagents
    /// out of [`SessionScope::TopLevel`]. Default: an empty page.
    async fn sessions_query(&self, _query: SessionQuery) -> SessionPage {
        SessionPage::default()
    }

    /// The full detail of one session (roster line + overlay/model/delivery/children/checkpoints) —
    /// the GUI detail-pane read. `None` if unknown. Default: `None`.
    async fn session_get(&self, _session: SessionId) -> Option<SessionDetail> {
        None
    }

    /// The roster grouped by owning profile (the "agent owns N conversations" view). Scoped to
    /// top-level conversations like the inbox. Default: empty.
    async fn sessions_by_profile(&self) -> Vec<(ProfileRef, Vec<SessionInfo>)> {
        Vec::new()
    }

    /// Full-text search over indexed session text (title + coalesced body), most-relevant first,
    /// capped at `limit` (`0` => a server default). Default: empty (no text index).
    async fn session_search(&self, _query: String, _limit: u32) -> Vec<SessionSearchHit> {
        Vec::new()
    }

    /// Apply a partial update to a session's roster metadata — the backend of daemon-app's "session
    /// actions" (rename, pin/reorder, archive). A read-modify-write of the session's
    /// `SessionMeta` that preserves the untouched fields (overlay/role/parent/bound profile) and
    /// promptly refreshes any live roster/tree subscribers. Default: unsupported.
    async fn session_update_meta(
        &self,
        _session: SessionId,
        _patch: SessionMetaPatch,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("session_update_meta".into()))
    }

    /// Ensure a durable session exists and wake it (start/resume work).
    async fn assign(&self, session: SessionId) -> Result<(), ApiError>;

    /// Cancel in-flight work for a session.
    async fn cancel(&self, session: SessionId) -> Result<(), ApiError>;

    /// The orchestration fleet roster + folded usage.
    async fn fleet(&self) -> FleetReport;

    /// The orchestration tree as the GUI/TUI drives it: every unit (single agent through
    /// fleets-of-fleets) with its parent/child structure, state, work, and folded usage. The default
    /// is an empty tree (a transport with no fleet projection, e.g. the session-only FFI).
    async fn tree(&self) -> TreeReport {
        TreeReport::default()
    }

    /// One unit's node view (`None` if unknown). Default: not available.
    async fn unit(&self, _id: UnitId) -> Option<UnitNode> {
        None
    }

    /// A bounded, **non-destructive** snapshot of up to `max` recent management-event views for one
    /// unit (GUI drill-down) — a read over a retained ring, not a drain (the impl in
    /// `daemon-orchestration` `runtime.rs` keeps the buffer; repeated reads return the same items).
    /// Default: empty.
    async fn unit_events(&self, _id: UnitId, _max: u32) -> Vec<ManageEventView> {
        Vec::new()
    }

    /// Drain up to `max` recent §17 [`Outbound`] items (streamed events + raised host requests) for
    /// one unit — the rich, transcript-fidelity drill-down a GUI reads to render a full transcript
    /// for *any* unit in the tree (not just a top-level interactive session). The coarse
    /// [`Self::unit_events`] is the fleet-dashboard view; this is the drill-down-to-transcript view,
    /// carrying the full §17 vocabulary (text, reasoning, tool I/O with opaque structured `detail`,
    /// opaque `ContentDelta`, usage, errors) plus blocking host requests, untouched. A **destructive,
    /// single-consumer** drain like [`Self::poll`] (each call consumes what it returns; `max == 0`
    /// drains all) — FFI/MCP-LCD only; a multi-surface consumer uses the non-destructive cursored
    /// [`Self::unit_history`] instead (the unified contract, daemon-event-io-spec §5.4.1).
    /// Default: empty (a transport with no fleet projection).
    async fn unit_outbound(&self, _id: UnitId, _max: u32) -> Vec<Outbound> {
        Vec::new()
    }

    /// Non-destructive, cursor-paged read of *any* unit's **durable** verifiable history (decoded
    /// finished chat blocks + management lifecycle records, each carrying the `verified` flag of its
    /// sealed segment) — the reconnect / scroll-back read for a GUI rendering a transcript for any
    /// node in the tree, and the auditor's one-chain verify pass. Complements (does not replace) the
    /// destructive live [`Self::unit_outbound`] drain. Default: an empty page.
    async fn unit_history(&self, _id: UnitId, _after_cursor: u64, _max: u32) -> JournalPageView {
        JournalPageView::default()
    }

    /// Pause a unit (lifecycle `ManageCommand`). Default: unsupported.
    async fn pause(&self, _id: UnitId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("pause".into()))
    }

    /// Resume a unit. Default: unsupported.
    async fn resume(&self, _id: UnitId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("resume".into()))
    }

    /// Scale a unit (an orchestrator sub-fleet) to `n` members. Default: unsupported.
    async fn scale(&self, _id: UnitId, _n: u32) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("scale".into()))
    }

    /// List parked §12 edit-approval requests awaiting an operator decision — for one `session` when
    /// given, else across all sessions (the operator HITL inbox). Default: empty (a transport with no
    /// durable approval store).
    async fn approvals_pending(&self, _session: Option<SessionId>) -> Vec<ApprovalInfo> {
        Vec::new()
    }

    /// Answer a parked §12 edit-approval request: record the operator's decision and wake the dormant
    /// session so it resumes (allow -> the gated tool runs; deny -> the tool returns an error). The
    /// `request_id` is the opaque id from [`Self::approvals_pending`]. Idempotent (a redelivered
    /// decision is a no-op). Default: unsupported (a transport with no durable approval store).
    async fn approval_decide(
        &self,
        _session: SessionId,
        _request_id: String,
        _allow: bool,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("approval_decide".into()))
    }

    /// The node's journal **verifying** key (hex-encoded dCBOR), so an auditor can independently
    /// verify the sealed segments returned by the history reads. `None` when the node exposes no
    /// journal signer. Default: `None`.
    async fn verifying_key(&self) -> Option<String> {
        None
    }

    /// List recorded §12 tool checkpoints (workspace snapshots taken before a mutating tool ran),
    /// newest first — for one `session` when given, else node-wide. Default: empty (a node with no
    /// checkpoint store). The GUI renders these as rewind points.
    async fn checkpoints(&self, _session: Option<SessionId>) -> Vec<CheckpointInfo> {
        Vec::new()
    }

    /// Rewind the workspace to a recorded checkpoint (`checkpoint_id` from [`Self::checkpoints`]).
    /// Best-effort restore of the captured workspace state. Default: unsupported (no checkpoint store).
    async fn checkpoint_rewind(
        &self,
        _session: SessionId,
        _checkpoint_id: String,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("checkpoint_rewind".into()))
    }

    /// The unified rewind op (conversation-rewind spec): truncate `session`'s transcript at
    /// `point.anchor` and, when `point.restore_workspace`, roll the workspace back to the matching
    /// checkpoint — sealing the journal and reconstructing the engine on both the live and the
    /// managed path (the two are made consistent here). Default: unsupported.
    async fn rewind(&self, _session: SessionId, _point: RewindPoint) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("rewind".into()))
    }

    /// Subscribe to the orchestration tree as a push stream of [`TreeEvent`]s, filtered for churn by
    /// `filter` (drop or coalesce transient subagents). The push delivery a streaming transport
    /// holds open; one-shot transports poll [`Self::tree`] instead. Default: an empty stream.
    async fn tree_subscribe(&self, _filter: TreeSubFilter) -> Result<TreeStream, ApiError> {
        Ok(stream::empty().boxed())
    }

    /// One-shot/long-poll read of the node-wide event feed past `cursor` (L3 `EventsSince`): the
    /// payload-free [`NodeEvent`] pointers a client uses to learn out-of-focus changes without
    /// polling, plus the feed cursors. Non-destructive. Default: an empty page (a node with no feed).
    async fn events_page(&self, _cursor: u64, _max: u32) -> EventsPage {
        EventsPage::default()
    }

    /// Subscribe to the node-wide event feed as a push stream of [`EventsPage`]s from `cursor`
    /// (backfill then live). The push delivery a streaming transport holds open; one-shot transports
    /// poll [`Self::events_page`]. Default: an empty stream (a node with no feed).
    async fn events_subscribe(&self, _cursor: u64) -> Result<NodeEventStream, ApiError> {
        Ok(stream::empty().boxed())
    }

    // -- chat→session routing pins (I5; daemon-event-io-spec §5.9) ------------------------------
    //
    // The resolve-first override table over the deterministic routing registry: a GUI/operator can
    // pin an inbound origin (chat/room) to a specific session (+ profile). Pins are durable and hot-
    // reloaded into the live registry. Defaults: empty / unsupported (a node without durable routing).

    /// List all chat→session routing pins. Default: empty.
    async fn routing_list_chats(&self) -> Vec<ChatRoute> {
        Vec::new()
    }

    /// Read the routing pin for an origin, if one is set. Default: `None`.
    async fn routing_get(&self, _origin: Origin) -> Option<ChatRoute> {
        None
    }

    /// Upsert a full chat→session routing pin (origin + session + profile + isolation). Default:
    /// unsupported.
    async fn routing_set(&self, _route: ChatRoute) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("routing_set".into()))
    }

    /// Pin an origin to a session (+ optional profile) — the convenience form of [`Self::routing_set`]
    /// with default isolation. Default: unsupported.
    async fn routing_bind_chat(
        &self,
        _origin: Origin,
        _session: SessionId,
        _profile: Option<ProfileRef>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("routing_bind_chat".into()))
    }

    /// Remove the pin for an origin (idempotent). Default: unsupported.
    async fn routing_unbind_chat(&self, _origin: Origin) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("routing_unbind_chat".into()))
    }

    /// Enumerate the rooms/chats a transport instance knows about (read-only), with the pinned
    /// session for each when one exists. Default: empty (a transport with no room enumeration).
    async fn transport_rooms(&self, _transport: TransportId) -> Vec<RoomInfo> {
        Vec::new()
    }

    // -- Transport adapters: the events-IO adapter framework (daemon-transport-adapter-spec.md) ---
    //
    // The declarative layer over the transport adapters: enumerate the available adapter families
    // (capabilities + account-setup schema, for the GUI "Add channel" picker) and the configured
    // instances with live connection/presence state (for the status bar + unified roster).
    // Read-only; defaults empty so a node without an `AdapterRegistry` inherits the surface, exactly
    // like the `transport_rooms` / `room_*` defaults.

    /// The events-IO transport adapters this node knows (families + capabilities + setup schema).
    /// Default: empty.
    async fn transport_adapters(&self) -> Vec<AdapterInfo> {
        Vec::new()
    }

    /// The configured transport instances (accounts) with their live connection/presence state.
    /// Default: empty.
    async fn transport_instances(&self) -> Vec<TransportInstanceInfo> {
        Vec::new()
    }

    // -- Rooms: the internal loopback transport CRUD (daemon-rooms-spec.md) ----------------------
    //
    // A Room is a first-class N-participant conversation backed by the internal loopback transport
    // (the `daemon-rooms` adapter). These ops are the Room-entity counterpart of the `routing_*`
    // pins above (which only bind an existing chat to a session): they create/destroy Rooms, edit
    // membership, and inject a post the RoomRouter fans out. Defaults: empty / unsupported (a node
    // built without the Rooms adapter), exactly like the `routing_*` / `transport_rooms` defaults.

    // -- Messaging-adapter management (daemon-messaging-adapter-spec.md §6.2): forwarded generically
    //    by the host to the owning adapter's `MessagingProtocol` feature interfaces. Defaults are
    //    empty / `Unsupported` (a node with no messaging adapter registered). --

    /// List the conversations a transport owns (`SupportsConversations::list`). Default: empty.
    async fn conv_list(&self, _transport: TransportId) -> Vec<ConversationInfo> {
        Vec::new()
    }

    /// Read one conversation by id (`SupportsConversations::get`). Default: `None`.
    async fn conv_get(&self, _transport: TransportId, _conv: String) -> Option<ConversationInfo> {
        None
    }

    /// Fetch the typed create-conversation form for a transport. Default: empty.
    async fn conv_create_details(&self, _transport: TransportId) -> CreateConversationDetails {
        CreateConversationDetails::default()
    }

    /// Create a conversation. Default: unsupported.
    async fn conv_create(
        &self,
        _transport: TransportId,
        _details: CreateConversationDetails,
    ) -> Result<ConversationInfo, ApiError> {
        Err(ApiError::Unsupported("conv_create".into()))
    }

    /// Fetch the typed channel-join form for a transport. Default: empty.
    async fn conv_join_details(&self, _transport: TransportId) -> ChannelJoinDetails {
        ChannelJoinDetails::default()
    }

    /// Join a channel. Default: unsupported.
    async fn conv_join(
        &self,
        _transport: TransportId,
        _details: ChannelJoinDetails,
    ) -> Result<ConversationInfo, ApiError> {
        Err(ApiError::Unsupported("conv_join".into()))
    }

    /// Leave a conversation. Default: unsupported.
    async fn conv_leave(&self, _transport: TransportId, _conv: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_leave".into()))
    }

    /// Send a message into a conversation, optionally attributed to a specific participant
    /// (`from = None` is the account/operator). Default: unsupported.
    async fn conv_send(
        &self,
        _transport: TransportId,
        _conv: String,
        _from: Option<Participant>,
        _message: UserMsg,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_send".into()))
    }

    /// Set a conversation's topic. Default: unsupported.
    async fn conv_set_topic(
        &self,
        _transport: TransportId,
        _conv: String,
        _topic: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_set_topic".into()))
    }

    /// Set a conversation's title. Default: unsupported.
    async fn conv_set_title(
        &self,
        _transport: TransportId,
        _conv: String,
        _title: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_set_title".into()))
    }

    /// Set a conversation's description. Default: unsupported.
    async fn conv_set_description(
        &self,
        _transport: TransportId,
        _conv: String,
        _description: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_set_description".into()))
    }

    /// Delete/destroy a conversation (`SupportsConversations::delete`). Default: unsupported.
    async fn conv_delete(&self, _transport: TransportId, _conv: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_delete".into()))
    }

    /// Read a conversation's durable, verifiable transcript — the merged conversation history keyed by
    /// `(transport, conv)` (daemon-messaging-adapter-spec.md). Default: empty.
    async fn conv_history(
        &self,
        _transport: TransportId,
        _conv: String,
        _after_cursor: u64,
        _max: u32,
    ) -> JournalPageView {
        JournalPageView::default()
    }

    /// Invite/add a participant to a conversation (`SupportsMembership::invite`). Default: unsupported.
    async fn member_invite(
        &self,
        _transport: TransportId,
        _conv: String,
        _who: Participant,
        _message: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_invite".into()))
    }

    /// Remove/kick a participant. Default: unsupported.
    async fn member_remove(
        &self,
        _transport: TransportId,
        _conv: String,
        _who: Participant,
        _reason: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_remove".into()))
    }

    /// Ban a participant. Default: unsupported.
    async fn member_ban(
        &self,
        _transport: TransportId,
        _conv: String,
        _who: Participant,
        _reason: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_ban".into()))
    }

    /// Set a participant's role/affiliation. Default: unsupported.
    async fn member_set_role(
        &self,
        _transport: TransportId,
        _conv: String,
        _who: Participant,
        _role: MemberRole,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_set_role".into()))
    }

    /// Fetch a remote contact's profile text (`SupportsContacts::get_profile`). Default: unsupported.
    async fn contact_get_profile(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
    ) -> Result<String, ApiError> {
        Err(ApiError::Unsupported("contact_get_profile".into()))
    }

    /// The contact's action menu (`SupportsContacts::action_menu`). Default: `None`.
    async fn contact_action_menu(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
    ) -> Option<ActionMenu> {
        None
    }

    /// Set a local alias for a contact (`SupportsContacts::set_alias`). Default: unsupported.
    async fn contact_set_alias(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
        _alias: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("contact_set_alias".into()))
    }

    /// Search the transport's contact/user directory (`SupportsDirectory::search_contacts`).
    /// Default: unsupported.
    async fn directory_search(
        &self,
        _transport: TransportId,
        _query: Option<String>,
    ) -> Result<Vec<ContactInfo>, ApiError> {
        Err(ApiError::Unsupported("directory_search".into()))
    }

    // -- ACP discovery + registry (catalog-style; the daemon probes its own PATH/endpoints) --

    /// Trigger a server-side ACP discovery scan (PATH + well-known locations + the curated
    /// known-agent recipe table + configured endpoints), confirming each candidate via the ACP
    /// `initialize` handshake. Operator-triggered (spawns subprocesses), like `model_search`.
    /// Default: empty.
    async fn acp_discover(&self) -> Vec<AcpAgentEntry> {
        Vec::new()
    }

    /// The last ACP discovery results plus any manually-registered recipes (the persisted catalog a
    /// GUI renders). Default: empty.
    async fn acp_catalog(&self) -> Vec<AcpAgentEntry> {
        Vec::new()
    }

    /// Manually register (persist) an ACP agent launch recipe — for a local path auto-detect missed
    /// or a remote endpoint. Default: unsupported.
    async fn acp_register(&self, _entry: AcpAgentEntry) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("acp_register".into()))
    }

    /// Remove a registered/cataloged ACP agent by name. Default: unsupported.
    async fn acp_remove(&self, _name: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("acp_remove".into()))
    }

    // -- Log-tail (I16): a node-level observability stream (shape now; thin impl later) --

    /// Subscribe to a node log-tail stream (resident-service / dashboard view). Default: an empty
    /// stream (a node exposing no log tail — clients fall back to `health`/`stats`/`telemetry`).
    async fn logs(&self, _filter: LogFilter) -> Result<LogLineStream, ApiError> {
        Ok(stream::empty().boxed())
    }

    // -- Forward-compat stubs (I11-I13, I15): shape now, runtime deferred. Grouped on ControlApi
    //    (defaulted) rather than new sub-traits to avoid breaking every NodeApi implementor; the
    //    DTOs are the stable wire contract a GUI can build against. --

    /// The runtime provider registry (I11). Default: empty (providers are frozen at node assembly;
    /// cloud-by-key already works via [`CredentialApi`]).
    async fn provider_list(&self) -> Vec<ProviderInfo> {
        Vec::new()
    }

    /// Register/configure a provider *type* at runtime (I11). Default: unsupported (assembly-time
    /// only today).
    async fn provider_register(&self, _provider: ProviderInfo) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("provider_register".into()))
    }

    /// The tools available at the node (I12). Default: empty (tools are launch-time / profile policy).
    async fn tool_list(&self) -> Vec<ToolInfo> {
        Vec::new()
    }

    /// Register a tool at runtime (I12). Default: unsupported (tools are launch-time policy today;
    /// only `ProfileSpec.tool_allowlist` is dynamic).
    async fn tool_register(&self, _tool: ToolInfo) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("tool_register".into()))
    }

    /// The daemon-authoritative command catalog — every operator/user command the node exposes
    /// (built-in node ops + context-engine / memory-provider / plugin contributions), as declarative
    /// [`CommandSpec`]s a thin client renders + autocompletes. Distinct from [`Self::tool_list`]
    /// (model-facing tools). Default: empty (a transport with no command registry).
    async fn command_list(&self) -> Vec<CommandSpec> {
        Vec::new()
    }

    /// Run a command by name (alias-aware), routing to its owning handler with the invocation's
    /// session/origin context, and return the rendered [`CommandOutput`]. Built-in commands are thin
    /// adapters over existing typed `NodeApi` ops; provider commands (e.g. `/lcm`, `/memory`) run on
    /// the contributing subsystem. Default: unsupported (a transport with no command registry).
    async fn command_invoke(
        &self,
        _invocation: CommandInvocation,
    ) -> Result<CommandOutput, ApiError> {
        Err(ApiError::Unsupported("command_invoke".into()))
    }

    /// Read the node's runtime config (I13). Default: unsupported (config is env/TOML startup-only).
    async fn config_get(&self) -> Result<NodeConfigView, ApiError> {
        Err(ApiError::Unsupported("config_get".into()))
    }

    /// Write the node's runtime config (I13). Default: unsupported.
    async fn config_set(&self, _config: NodeConfigView) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("config_set".into()))
    }

    /// List scheduled cron jobs (I15). Default: empty (the scheduler is PLANNED; builds on the
    /// `daemon-activation` wake/outbox substrate).
    async fn cron_list(&self) -> Vec<CronJob> {
        Vec::new()
    }

    /// Create a scheduled job (I15). Default: unsupported.
    async fn cron_create(&self, _spec: CronSpec) -> Result<String, ApiError> {
        Err(ApiError::Unsupported("cron_create".into()))
    }

    /// Update a scheduled job (I15). Default: unsupported.
    async fn cron_update(&self, _id: String, _spec: CronSpec) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("cron_update".into()))
    }

    /// Delete a scheduled job (I15). Default: unsupported.
    async fn cron_delete(&self, _id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("cron_delete".into()))
    }

    /// Fire a scheduled job now (I15). Default: unsupported.
    async fn cron_trigger(&self, _id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("cron_trigger".into()))
    }

    /// List recent runs of a scheduled job (I15). Default: empty.
    async fn cron_runs(&self, _id: String) -> Vec<CronRun> {
        Vec::new()
    }

    /// Pause or resume a scheduled job (I15): a paused job stays in the store but never fires;
    /// resuming recomputes its next fire from now. Default: unsupported.
    async fn cron_pause(&self, _id: String, _paused: bool) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("cron_pause".into()))
    }

    /// List pending cron-job suggestions (I15): consent-first proposals from the catalog/blueprints.
    /// Default: empty.
    async fn cron_suggestions(&self) -> Vec<CronSuggestion> {
        Vec::new()
    }

    /// Accept a suggestion (I15): create the backing job and mark the suggestion accepted; returns
    /// the new job id. Default: unsupported.
    async fn cron_accept_suggestion(&self, _id: String) -> Result<String, ApiError> {
        Err(ApiError::Unsupported("cron_accept_suggestion".into()))
    }

    /// Dismiss a suggestion (I15): latch it by `dedup_key` so it is never re-offered. Default:
    /// unsupported.
    async fn cron_dismiss_suggestion(&self, _id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("cron_dismiss_suggestion".into()))
    }

    // -- Filesystem / workspace surface (daemon-fs-surface-spec.md). Grouped (defaulted) here, not a
    //    new sub-trait, so every NodeApi implementor inherits the surface; a node with a workspace
    //    binds the real impl (backed by daemon-host's WorkspaceFs). --

    /// The browsable roots this node exposes: host browse roots (home + operator allowlist) +
    /// the workspace root + any opened session sandboxes. Default: empty (no workspace surface).
    async fn fs_roots(&self) -> Vec<FsRoot> {
        Vec::new()
    }

    /// One directory's children (root-relative `dir`, "" = the root). Ignored entries are *marked*
    /// (`FsEntry.ignored`), not hidden, when `show_ignored` is false the caller may still hide them.
    /// Default: unsupported.
    async fn fs_list(
        &self,
        _root: FsRootId,
        _dir: String,
        _show_ignored: bool,
    ) -> Result<Vec<FsEntry>, ApiError> {
        Err(ApiError::Unsupported("fs_list".into()))
    }

    /// One entry's metadata. Default: unsupported.
    async fn fs_stat(&self, _root: FsRootId, _path: String) -> Result<FsEntry, ApiError> {
        Err(ApiError::Unsupported("fs_stat".into()))
    }

    /// Read up to `max_bytes` (`0` = a server default) of a file, plus an etag + truncation flag.
    /// Default: unsupported.
    async fn fs_read(
        &self,
        _root: FsRootId,
        _path: String,
        _max_bytes: u64,
    ) -> Result<FsContent, ApiError> {
        Err(ApiError::Unsupported("fs_read".into()))
    }

    /// Write bytes with optimistic concurrency (`base_revision`; `None` = create-or-overwrite).
    /// `force` overrides the sensitive-path / `Deny` gate. `Workspace`/`Session` roots only —
    /// `Host` roots are read-only. Returns the new etag. Default: unsupported.
    async fn fs_write(
        &self,
        _root: FsRootId,
        _path: String,
        _bytes: Vec<u8>,
        _base_revision: Option<FsRevision>,
        _force: bool,
    ) -> Result<FsRevision, ApiError> {
        Err(ApiError::Unsupported("fs_write".into()))
    }

    /// Server-side project search over a root (content / regex), paginated. Default: unsupported.
    async fn fs_search(
        &self,
        _root: FsRootId,
        _query: FsSearchQuery,
    ) -> Result<FsSearchPage, ApiError> {
        Err(ApiError::Unsupported("fs_search".into()))
    }

    /// The cursor / long-poll form of the change stream (the wire-marshaled form of `fs_watch`):
    /// read change events under `dir` since `after_seq`. Cursored + resync-capable per the unified
    /// cursored-stream contract (`daemon-core/docs/daemon-event-io-spec.md` §5.4.1): the page carries
    /// `head_seq` (the live edge) and `reset = true` when `after_seq` aged out of the bounded watch
    /// ring (events evicted past the reader), signaling the client to re-list the dir to reconcile —
    /// the fs analogue of the merged log's `Lagged -> Reset`. Default: unsupported.
    async fn fs_watch_after(
        &self,
        _root: FsRootId,
        _dir: String,
        _after_seq: u64,
        _max: u32,
    ) -> Result<FsWatchPageView, ApiError> {
        Err(ApiError::Unsupported("fs_watch_after".into()))
    }

    /// A live push stream of changes under `dir` (transport capability). Default: empty stream.
    async fn fs_watch(&self, _root: FsRootId, _dir: String) -> Result<FsWatchStream, ApiError> {
        Ok(stream::empty().boxed())
    }

    // -- Content store (daemon-content-transfer-spec.md, Phase 1). Defaulted/unsupported so a node
    //    without a configured blob store inherits the surface. --

    /// Store bytes in the node content store, returning a content-addressed [`BlobRef`]
    /// (write-if-absent + dedup). Default: unsupported.
    async fn blob_put(&self, _bytes: Vec<u8>) -> Result<BlobRef, ApiError> {
        Err(ApiError::Unsupported("blob_put".into()))
    }

    /// Read a blob by hash (full read is integrity-verified; a `range` read returns an unverified
    /// slice). Default: unsupported.
    async fn blob_get(
        &self,
        _hash: ContentHash,
        _range: Option<ByteRange>,
    ) -> Result<Vec<u8>, ApiError> {
        Err(ApiError::Unsupported("blob_get".into()))
    }

    /// Metadata for a blob (presence + size). Default: absent.
    async fn blob_stat(&self, _hash: ContentHash) -> BlobStat {
        BlobStat {
            size: 0,
            present: false,
        }
    }

    /// Materialize a blob into a workspace path (the `Workspace`/`Session` write path, with the same
    /// containment + sensitive-path/`force` + checkpoint + `Conflict` gating as `fs_write`). Default:
    /// unsupported.
    async fn fs_write_from_blob(
        &self,
        _root: FsRootId,
        _path: String,
        _hash: ContentHash,
        _base_revision: Option<FsRevision>,
        _force: bool,
    ) -> Result<FsRevision, ApiError> {
        Err(ApiError::Unsupported("fs_write_from_blob".into()))
    }
}

/// The model-management sub-surface: search/download/cache/catalog/activate the local-inference
/// models the node can run. Every method has a default so a transport that does not host model
/// management (the session-only FFI, test stubs) inherits the surface without implementing it; the
/// node's [`NodeApi`] binds the real implementation (backed by `daemon-models`' `ModelManager`).
///
/// The discovery half is a **two-step** flow: [`Self::model_search`] returns matching repos (step
/// 1), then [`Self::model_files`] lists a chosen repo's loadable files (step 2); the client selects
/// one and calls [`Self::model_download`].
#[async_trait]
pub trait ModelApi: Send + Sync {
    /// Step 1 — search Hugging Face for repos loadable by the query's engine.
    async fn model_search(&self, _query: SearchQuery) -> Result<SearchPage, ApiError> {
        Err(ApiError::Unsupported("model_search".into()))
    }

    /// Step 2 — list a repo's loadable files for `engine` (the set a client selects to download).
    async fn model_files(
        &self,
        _repo: String,
        _revision: Option<String>,
        _engine: ModelEngine,
    ) -> Result<Vec<ModelFile>, ApiError> {
        Err(ApiError::Unsupported("model_files".into()))
    }

    /// Start downloading a model into the shared cache; returns the job handle.
    async fn model_download(&self, _model: ModelRef) -> Result<DownloadId, ApiError> {
        Err(ApiError::Unsupported("model_download".into()))
    }

    /// All download job statuses (in-flight + finished this run).
    async fn model_downloads(&self) -> Vec<DownloadStatus> {
        Vec::new()
    }

    /// Cancel a download (abandon partial bytes).
    async fn model_cancel(&self, _id: DownloadId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("model_cancel".into()))
    }

    /// Pause a download (keep partial bytes for resume).
    async fn model_pause(&self, _id: DownloadId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("model_pause".into()))
    }

    /// Resume a paused/failed download.
    async fn model_resume(&self, _id: DownloadId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("model_resume".into()))
    }

    /// The installed-model catalog.
    async fn model_catalog(&self) -> Vec<InstalledModel> {
        Vec::new()
    }

    /// Delete an installed model (catalog record + cached artifact).
    async fn model_delete(&self, _id: ModelId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("model_delete".into()))
    }

    /// Activate a cataloged model for a profile (`None` = the node's default local profile), so new
    /// worker spawns load it.
    async fn model_activate(&self, _id: ModelId, _profile: Option<String>) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("model_activate".into()))
    }

    /// Recommend a quantization for a repo given the detected hardware (the "tune"-like pick): for
    /// llama a GGUF file to download, for mistral.rs an in-engine ISQ level. `budget_bytes`
    /// overrides the auto-detected VRAM/RAM budget when set.
    async fn model_recommend(
        &self,
        _repo: String,
        _revision: Option<String>,
        _engine: ModelEngine,
        _budget_bytes: Option<u64>,
    ) -> Result<QuantRecommendation, ApiError> {
        Err(ApiError::Unsupported("model_recommend".into()))
    }

    /// Start an offline quantization of a repo's GGUF to `target_quant` (e.g. `Q4_K_M`); returns the
    /// job handle. `source_file` selects the source GGUF (`None` = the highest-precision one).
    async fn model_quantize(
        &self,
        _repo: String,
        _revision: Option<String>,
        _target_quant: String,
        _source_file: Option<String>,
    ) -> Result<QuantizeId, ApiError> {
        Err(ApiError::Unsupported("model_quantize".into()))
    }

    /// All quantization job statuses (in-flight + finished this run).
    async fn model_quantizes(&self) -> Vec<QuantizeStatus> {
        Vec::new()
    }

    /// Read GGUF metadata for a cataloged model (architecture, context length, file-type, …).
    async fn model_inspect(&self, _id: ModelId) -> Result<GgufInfo, ApiError> {
        Err(ApiError::Unsupported("model_inspect".into()))
    }

    /// The discoverable model catalog a GUI's model picker renders: well-known cloud models (incl.
    /// `claude-opus-4-8`) merged with locally-installed models. Default: the built-in cloud catalog.
    async fn models(&self) -> Vec<ModelDescriptor> {
        ModelDescriptor::builtin_cloud_catalog()
    }

    /// The model a profile currently resolves to (`None` profile = the active default). `None` when
    /// no profile/model is resolvable. Default: `None`.
    async fn model_current(
        &self,
        _profile: Option<String>,
    ) -> Result<Option<ModelDescriptor>, ApiError> {
        Ok(None)
    }
}

/// The profile sub-surface: create, inspect, edit, and select the agent configuration bundles
/// ([`ProfileSpec`]) a session binds to. A profile is the single durable configuration unit; it is
/// edited in full via `profile_update` (no separate partial-config surface), and a live session is
/// adjusted via a `SessionOverlay` ([`SessionApi::set_session_overlay`]). Every method defaults to
/// [`ApiError::Unsupported`] / empty so a transport that hosts no profile store (the session-only
/// FFI, test stubs) inherits the surface; the node's [`NodeApi`] binds the real implementation
/// (backed by a `ProfileStore`).
#[async_trait]
pub trait ProfileApi: Send + Sync {
    /// Whether this node hosts profile/skill versioning (a bound revision log). When false, the
    /// `profile_history`/`profile_revert` (and skill equivalents) ops resolve to
    /// [`ApiError::Unsupported`]. Advertised as the `versioning` Hello feature so a client can hide
    /// its history/revert affordances up front rather than discovering the gap per request.
    fn supports_versioning(&self) -> bool {
        false
    }

    /// All known profiles (listing view, with the active default marked).
    async fn profile_list(&self) -> Vec<ProfileInfo> {
        Vec::new()
    }

    /// Fetch one profile's full spec by id (`None` if unknown).
    async fn profile_get(&self, _id: String) -> Result<Option<ProfileSpec>, ApiError> {
        Err(ApiError::Unsupported("profile_get".into()))
    }

    /// Create a new profile (errors if the id already exists).
    async fn profile_create(&self, _spec: ProfileSpec) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("profile_create".into()))
    }

    /// Replace an existing profile (errors if the id is unknown).
    async fn profile_update(&self, _spec: ProfileSpec) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("profile_update".into()))
    }

    /// Delete a profile by id.
    async fn profile_delete(&self, _id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("profile_delete".into()))
    }

    /// Select the active default profile (new sessions bind to it unless overridden).
    async fn profile_select(&self, _id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("profile_select".into()))
    }

    /// Clone `source` into a new profile `new_id` (a local copy; starts a fresh revision history).
    async fn profile_clone(&self, _source: String, _new_id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("profile_clone".into()))
    }

    /// Export a profile as a portable [`Distribution`] (spec + its local skills + head revisions).
    async fn profile_export(&self, _id: String) -> Result<Distribution, ApiError> {
        Err(ApiError::Unsupported("profile_export".into()))
    }

    /// Import a [`Distribution`] as a new profile (defaults to the distribution's id; `new_id`
    /// overrides). Returns the created profile id.
    async fn profile_import(
        &self,
        _dist: Distribution,
        _new_id: Option<String>,
    ) -> Result<String, ApiError> {
        Err(ApiError::Unsupported("profile_import".into()))
    }

    /// The revision history of a profile (oldest first).
    async fn profile_history(&self, _id: String) -> Result<Vec<Revision>, ApiError> {
        Err(ApiError::Unsupported("profile_history".into()))
    }

    /// The profile spec as recorded at revision `seq`.
    async fn profile_at(&self, _id: String, _seq: u64) -> Result<ProfileSpec, ApiError> {
        Err(ApiError::Unsupported("profile_at".into()))
    }

    /// Revert a profile to revision `seq` (non-destructive: appends a new head equal to that
    /// revision, so roll-forward is reverting to a later `seq`).
    async fn profile_revert(&self, _id: String, _seq: u64) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("profile_revert".into()))
    }

    /// The revision history of a skill (oldest first).
    async fn skill_history(&self, _name: String) -> Result<Vec<Revision>, ApiError> {
        Err(ApiError::Unsupported("skill_history".into()))
    }

    /// The skill bundle as recorded at revision `seq`.
    async fn skill_at(&self, _name: String, _seq: u64) -> Result<SkillBundle, ApiError> {
        Err(ApiError::Unsupported("skill_at".into()))
    }

    /// Revert a skill to revision `seq` (non-destructive; rejected for binary-bundled skills).
    async fn skill_revert(&self, _name: String, _seq: u64) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("skill_revert".into()))
    }

    /// List a profile's skill library with per-skill usage + lifecycle state (the curator view).
    /// `profile` defaults to the node's active default.
    async fn curator_list(&self, _profile: Option<String>) -> Result<Vec<CuratorEntry>, ApiError> {
        Err(ApiError::Unsupported("curator_list".into()))
    }

    /// Pin a skill (protect it from automatic archiving).
    async fn curator_pin(&self, _profile: Option<String>, _name: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("curator_pin".into()))
    }

    /// Unpin a skill (re-expose it to automatic curation).
    async fn curator_unpin(&self, _profile: Option<String>, _name: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("curator_unpin".into()))
    }

    /// Archive a skill (move it out of discovery + the index into `.archive/`).
    async fn curator_archive(
        &self,
        _profile: Option<String>,
        _name: String,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("curator_archive".into()))
    }

    /// Restore an archived skill back into the live library.
    async fn curator_restore(
        &self,
        _profile: Option<String>,
        _name: String,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("curator_restore".into()))
    }

    /// Run the deterministic curator over a profile's library (stale/archive/reactivate), returning
    /// the lifecycle changes applied.
    async fn curator_run(&self, _profile: Option<String>) -> Result<Vec<CuratorChange>, ApiError> {
        Err(ApiError::Unsupported("curator_run".into()))
    }

    /// Read a skill bundle at its current head revision (the in-app view convenience — `skill_at` at
    /// the latest `seq`). Default: unsupported.
    async fn skill_get(&self, _name: String) -> Result<SkillBundle, ApiError> {
        Err(ApiError::Unsupported("skill_get".into()))
    }

    /// Create or replace a skill bundle's body (the in-app edit half; records a new revision).
    /// Default: unsupported.
    async fn skill_put(&self, _bundle: SkillBundle) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("skill_put".into()))
    }
}

/// The credential sub-surface: set / list (redacted) / remove the provider secrets the node's
/// credential authority provisions onto each model request (`Request.auth`). Keyed by profile /
/// credential-ref, mirroring hermes' `/api/env`. Every method defaults to
/// [`ApiError::Unsupported`] / empty so a transport that hosts no credential store inherits the
/// surface; the node binds the real implementation.
#[async_trait]
pub trait CredentialApi: Send + Sync {
    /// Store (or replace) the secret for `profile`.
    async fn credential_set(&self, _profile: String, _secret: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("credential_set".into()))
    }

    /// List the stored credentials, redacted (never returns secrets).
    async fn credential_list(&self) -> Vec<CredentialInfo> {
        Vec::new()
    }

    /// Remove the secret for `profile`.
    async fn credential_remove(&self, _profile: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("credential_remove".into()))
    }
}

// ---------------------------------------------------------------------------
// Interactive auth (SSO / OAuth2) — the client-driven login seam
// ---------------------------------------------------------------------------
//
// See `daemon-interactive-auth-spec.md`. A decoupled (possibly remote) client drives a
// browser-redirect login: `auth_begin` mints an authorization URL against a redirect_uri the *client*
// owns, the client opens a browser and captures the redirect, then relays the callback to
// `auth_complete`. The daemon never owns a browser or a loopback; it parks a pending flow between the
// two calls and writes the resulting credential into the same `CredentialStore` as `CredentialApi`.

/// The kind of interactive auth flow (informs the client how to capture the redirect).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthFlowKind {
    /// Matrix SSO: the redirect carries a single-use `loginToken`.
    MatrixSso,
    /// OAuth2 / OIDC authorization-code + PKCE: the redirect carries `code` + `state`.
    OAuth2Pkce,
}

/// Optionally bind the freshly-authenticated account to a profile on success.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthBindRequest {
    /// The profile to attach the account to (edits its `bound_accounts`).
    pub profile: ProfileRef,
    /// The instance-qualified transport id (e.g. `matrix/@bot:hs.org`); `None` when it is only known
    /// after login (the family derives it in `auth_complete`).
    pub transport_instance: Option<TransportId>,
    /// The `CredentialStore` key to store the blob under; defaulted/derived if `None`.
    pub credential_ref: Option<String>,
}

/// Begin an interactive auth flow.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthBeginRequest {
    /// Transport/provider family, e.g. `"matrix"`.
    pub family: String,
    /// Family-specific parameters (e.g. matrix: `homeserver`, optional `idp_id`).
    pub params: BTreeMap<String, String>,
    /// The redirect URI the client controls and will capture (loopback URL or custom-scheme deep link).
    pub redirect_uri: String,
    /// Optionally bind the resulting account to a profile on success.
    pub bind: Option<AuthBindRequest>,
}

/// The parked-flow handle returned by `auth_begin`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthBeginResponse {
    /// The single-use flow id to pass to `auth_complete` / `auth_cancel`.
    pub flow_id: String,
    /// The URL the client opens in a browser.
    pub authorization_url: String,
    /// The redirect URI the flow expects (echoed back).
    pub redirect_uri: String,
    /// Flow TTL (unix seconds); the flow is evicted after this.
    pub expires_at: u64,
    /// Which flow kind this is (how to capture the redirect).
    pub flow_kind: AuthFlowKind,
}

/// Finish a flow from the captured redirect.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthCompleteRequest {
    /// The flow id from `auth_begin`.
    pub flow_id: String,
    /// The captured callback: the full redirect URL or just its query string (carries the
    /// `loginToken`, or `code` + `state`).
    pub callback: String,
}

/// The outcome of a completed flow.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthCompleteResponse {
    /// The `CredentialStore` key the session blob was stored under.
    pub credential_ref: String,
    /// A human label for the account (e.g. the resolved `@user:hs.org`).
    pub account_label: String,
    /// The instance-qualified transport id the account resolves to.
    pub transport_instance: TransportId,
    /// The profile the account was bound to, if a bind was requested and honored.
    pub bound_profile: Option<ProfileRef>,
}

/// One field of a family's `params` form (capability discovery).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthParamField {
    /// The `params` key.
    pub key: String,
    /// A human label for the field.
    pub label: String,
    /// Whether the field is required.
    pub required: bool,
}

/// A registered interactive-auth provider (capability discovery for the client).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthProviderInfo {
    /// The transport/provider family (`auth_begin.family`).
    pub family: String,
    /// The flow kind.
    pub flow_kind: AuthFlowKind,
    /// A human display name.
    pub display_name: String,
    /// The `params` fields the client should collect.
    pub params_schema: Vec<AuthParamField>,
}

/// The interactive-auth sub-surface. Every method defaults to [`ApiError::Unsupported`] / empty so a
/// transport that registers no auth factory inherits the surface; the node binds the real impl.
#[async_trait]
pub trait AuthApi: Send + Sync {
    /// Begin a flow: mint the authorization URL against the client-supplied `redirect_uri` and park it.
    async fn auth_begin(&self, _req: AuthBeginRequest) -> Result<AuthBeginResponse, ApiError> {
        Err(ApiError::Unsupported("auth_begin".into()))
    }

    /// Finish a flow from the captured redirect; persist the credential and optionally bind a profile.
    async fn auth_complete(
        &self,
        _req: AuthCompleteRequest,
    ) -> Result<AuthCompleteResponse, ApiError> {
        Err(ApiError::Unsupported("auth_complete".into()))
    }

    /// Drop a pending flow (user aborted / cleanup). Idempotent.
    async fn auth_cancel(&self, _flow_id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("auth_cancel".into()))
    }

    /// The interactive-auth providers this node offers (for the client to render the right form).
    async fn auth_providers(&self) -> Vec<AuthProviderInfo> {
        Vec::new()
    }
}

/// The whole node surface: the session, control, model-management, profile/config, credential, and
/// interactive-auth sub-surfaces.
pub trait NodeApi:
    SessionApi + ControlApi + ModelApi + ProfileApi + CredentialApi + AuthApi
{
}
impl<T: SessionApi + ControlApi + ModelApi + ProfileApi + CredentialApi + AuthApi> NodeApi for T {}

// ---------------------------------------------------------------------------
// Outbound drain item (§17 events + raised host requests share one queue)
// ---------------------------------------------------------------------------
//
// The drain item is the canonical `daemon_protocol::Outbound` union, re-exported above. Events and
// blocking host requests ride the **same** drain queue (daemon-ffi-spec §3.3), so a poll-based
// embedder (see [`SessionApi::poll`]) sees both in order.

// ---------------------------------------------------------------------------
// Report DTOs (decoupled from the substrate's concrete types)
// ---------------------------------------------------------------------------

/// The resident-service tree health (a projection of the host supervisor handle).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthReport {
    /// Whether every resident service is currently `Ok`.
    pub all_ok: bool,
    /// Per-service health.
    pub services: Vec<ServiceHealth>,
}

/// One resident service's health line.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceHealth {
    /// The service name (e.g. `JobOutboxDispatcher`).
    pub name: String,
    /// Whether the service is currently healthy.
    pub ok: bool,
    /// How many times it has been restarted.
    pub restarts: u32,
    /// A human-readable detail when not `ok`.
    pub detail: Option<String>,
}

/// Durable queue depths and live counts (a projection of `StoreStats` + active sessions).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatsReport {
    /// Pending background jobs on the durable job outbox.
    pub pending_jobs: u64,
    /// Pending wake hints on the durable wake outbox.
    pub pending_wakes: u64,
    /// Total durable session records.
    pub sessions: u64,
    /// Currently-active (in-memory) incarnations.
    pub active: u64,
    /// The folded durable usage total across every session (tokens, cache, reasoning, and estimated
    /// `cost_micros`) — the node-wide accounting line a GUI renders alongside the queue depths.
    #[serde(default)]
    pub usage: UsageDelta,
}

/// A point-in-time observability snapshot exposed over the control surface (the API-level mirror of
/// the resident `daemon-telemetry` metrics dump): folded usage + an event counter + aggregate health
/// + the durable queue depths. This is the `Dump` op a GUI/operator polls for a live HUD.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryDump {
    /// Cumulative usage folded across every unit reporting to the node aggregator (incl. cost).
    pub usage: UsageDelta,
    /// Management events folded so far (the resident aggregator's event counter).
    pub events: u64,
    /// Aggregate service-tree health bit.
    pub healthy: bool,
    /// Pending background jobs on the durable job outbox.
    pub pending_jobs: u64,
    /// Pending wake hints on the durable wake outbox.
    pub pending_wakes: u64,
    /// Total durable session records.
    pub sessions: u64,
    /// Currently-active (in-memory) incarnations.
    pub active: u64,
}

/// Whether a session is owned by the durable (control surface, `assign`) or the live-interactive
/// (session surface, `submit`) lifecycle. The two are mutually exclusive for a given id; the unified
/// roster surfaces both so a GUI sees in-progress interactive chats *and* durable sessions in one
/// list.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Lifecycle {
    /// Durable-managed (a `session_record` row; driven via [`ControlApi::assign`]).
    #[default]
    Durable,
    /// Live-interactive (an in-memory submit/poll chat; driven via [`SessionApi::submit`]).
    Live,
}

/// A session's identity + lifecycle state + roster metadata. Enriched for the GUI roster: it carries
/// the bound profile (agent identity), an optional title, last-activity (for sort), the
/// durable-vs-live [`Lifecycle`], and the hierarchy [`SessionRole`] + `parent` so a client can keep
/// the `Primary` inbox separate from drill-down children.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionInfo {
    /// The session id.
    pub session: SessionId,
    /// Its durable lifecycle state.
    pub state: SessionState,
    /// Whether the session supports conversation rewind (`AgentCommand::RewindTo`, conversation-rewind
    /// spec). `true` for daemon-core-backed sessions (the daemon owns the conversation state and can
    /// truncate it); `false` for foreign backends (e.g. ACP) whose protocol has no truncate-at-anchor
    /// primitive. A GUI/TUI reads this to hide rewind for non-rewindable sessions.
    #[serde(default = "default_rewindable")]
    pub rewindable: bool,
    /// The profile this session binds its engine to (the agent that owns it); `None` = the node's
    /// active default.
    #[serde(default)]
    pub bound_profile: Option<ProfileRef>,
    /// A human-readable conversation title, when set (generation is deferred).
    #[serde(default)]
    pub title: Option<String>,
    /// Unix-millis of the last activity on this session, for roster sort (`None` if never stamped).
    #[serde(default)]
    pub last_activity_ms: Option<u64>,
    /// Whether this session is durable-managed or live-interactive.
    #[serde(default)]
    pub lifecycle: Lifecycle,
    /// This session's hierarchy role (`Primary` is the only role in the `TopLevel` roster).
    #[serde(default)]
    pub role: SessionRole,
    /// The parent session id, when this is a child/subagent.
    #[serde(default)]
    pub parent: Option<SessionId>,
    /// Whether the operator pinned this conversation (sorts ahead of activity order in the roster).
    #[serde(default)]
    pub pinned: bool,
    /// Whether the operator archived this conversation (excluded from the default roster scopes,
    /// surfaced only under [`SessionScope::Archived`]).
    #[serde(default)]
    pub archived: bool,
}

/// `serde` default for [`SessionInfo::rewindable`] on older wire payloads that predate the field:
/// daemon-core sessions are rewindable, so the safe default is `true`.
fn default_rewindable() -> bool {
    true
}

/// The scope filter for [`ControlApi::sessions_query`] — the GUI roster query. The tree is the lazy
/// drill-down for children, so the default `TopLevel` returns only `Primary` conversations (the
/// inbox); the by-profile / by-transport scopes back the per-agent / per-transport views.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SessionScope {
    /// Only top-level (`Primary`) conversations — the inbox. Children are reached via `tree()`.
    #[default]
    TopLevel,
    /// Sessions bound to a specific profile (the per-agent view).
    ByProfile(ProfileRef),
    /// Sessions whose `Primary` delivery target names a specific transport instance.
    ByTransport(TransportId),
    /// Only archived `Primary` conversations — the explicit archived view. The default scopes
    /// (`TopLevel`/`ByProfile`/`ByTransport`) exclude archived sessions; this is the opt-in to see them.
    Archived,
    /// Every session regardless of role (explicit opt-in; can be large in a fleets-of-fleets node).
    All,
}

/// A scoped, paginated roster query. The cursor is the last session id from the previous page
/// (`None` for the first page); `limit == 0` means a sensible server default.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionQuery {
    /// The roster scope filter.
    #[serde(default)]
    pub scope: SessionScope,
    /// The exclusive cursor: the last [`SessionId`] returned by the previous page (`None` = start).
    /// An **id-cursor** for pagination — a deliberate exception to the seq/journal cursor vocabulary
    /// (it is not an `after_seq` live position nor an `after_cursor` journal key); see
    /// daemon-event-io-spec §5.4.2. L4 roster *delta* reads use [`Self::since_rev`] instead.
    #[serde(default)]
    pub after: Option<SessionId>,
    /// Maximum sessions to return (`0` = a server default).
    #[serde(default)]
    pub limit: u32,
    /// L4 delta read: when `Some(R)`, ask for only the sessions whose roster metadata changed after
    /// revision `R` (plus the `removed` list and current `rev`), instead of the full page. `None`
    /// (the default) = a full page — today's behavior, and the back-compat path for an old daemon
    /// (which ignores the field) or a cold client (no persisted `rev`).
    #[serde(default)]
    pub since_rev: Option<u64>,
}

/// A page of the scoped roster: the matching sessions plus the cursor to fetch the next page
/// (`None` when the page is the last).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionPage {
    /// The sessions in this page (already scope-filtered + ordered).
    pub sessions: Vec<SessionInfo>,
    /// The cursor to pass as [`SessionQuery::after`] on the next read; `None` => no more pages.
    #[serde(default)]
    pub next_cursor: Option<SessionId>,
    /// L4: the roster revision this page reflects. The client persists it and passes it back as
    /// [`SessionQuery::since_rev`] on the next read to fetch only the delta. `0` for a node with no
    /// event feed (no revision tracking) — the client then always takes full pages.
    #[serde(default)]
    pub rev: u64,
    /// L4 delta read: session ids removed from the roster since the requested `since_rev` (so the
    /// client prunes them). Empty on a full page and effectively empty today (archive is a *change*
    /// with `archived=true`, not a removal); reserved for a future hard-delete path.
    #[serde(default)]
    pub removed: Vec<SessionId>,
}

/// A partial update to a session's roster metadata — the backend of daemon-app's "session actions"
/// (rename, pin/reorder, archive). Each field is `None` to leave it unchanged; `title` is a
/// double-option so a `Some(None)` clears the title (rename-to-empty) while `None` leaves it intact.
/// Applied as a read-modify-write of [`SessionMeta`](../daemon_store/struct.SessionMeta.html) that
/// preserves the untouched fields (overlay/role/parent/bound profile).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMetaPatch {
    /// Set/clear the conversation title (`None` = leave; `Some(None)` = clear; `Some(Some(t))` = set).
    #[serde(default)]
    pub title: Option<Option<String>>,
    /// Pin/unpin the conversation (`None` = leave unchanged).
    #[serde(default)]
    pub pinned: Option<bool>,
    /// Archive/unarchive the conversation (`None` = leave unchanged).
    #[serde(default)]
    pub archived: Option<bool>,
}

/// The full detail of one session — the single round-trip a GUI detail pane reads: roster `info`
/// plus the resolved overlay/model/provider, delivery targets, parent/children ids, and a
/// checkpoint count.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDetail {
    /// The roster line for this session.
    pub info: SessionInfo,
    /// The session's persisted per-session overlay (model/provider/tools/approval), when recorded.
    #[serde(default)]
    pub overlay: Option<SessionOverlay>,
    /// The model the session currently resolves to, when known.
    #[serde(default)]
    pub model: Option<String>,
    /// The session's outbound delivery targets (where its replies post).
    #[serde(default)]
    pub delivery_targets: Vec<DeliveryTarget>,
    /// This session's direct children (subagents / managed children), for tree drill-down.
    #[serde(default)]
    pub children: Vec<SessionId>,
    /// How many §12 tool checkpoints are recorded for this session (rewind points).
    #[serde(default)]
    pub checkpoints: u32,
}

/// One full-text session-search hit — the transport-stable mirror of the store's `SessionSearchHit`
/// ([`ControlApi::session_search`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSearchHit {
    /// The session that matched.
    pub session: SessionId,
    /// The session's indexed title (empty when none was indexed).
    pub title: String,
    /// A highlighted excerpt of the matching body text.
    pub snippet: String,
}

/// A parked §12 edit-approval request awaiting an operator decision — the transport-stable mirror of
/// a durable `pending_approvals` row, surfaced by [`ControlApi::approvals_pending`] so a GUI/operator
/// can render the pending asks and answer them with [`ControlApi::approval_decide`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalInfo {
    /// The session that parked the request (and resumes when it is answered).
    pub session: SessionId,
    /// The opaque request id to pass back to [`ControlApi::approval_decide`].
    pub request_id: String,
    /// A human-readable summary of the proposed action.
    pub prompt: String,
    /// The target path, when the action is a file edit (`None` for a non-path action).
    #[serde(default)]
    pub path: Option<String>,
}

/// A recorded §12 tool checkpoint — the transport-stable mirror of a `daemon-core`
/// `CheckpointRecord`, surfaced by [`ControlApi::checkpoints`] so a GUI/operator can render the
/// rewind points and restore one with [`ControlApi::checkpoint_rewind`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointInfo {
    /// The opaque checkpoint id to pass back to [`ControlApi::checkpoint_rewind`].
    pub id: String,
    /// The session whose turn produced the checkpoint.
    pub session: SessionId,
    /// The mutating tool whose run was checkpointed (e.g. `fs` / `shell`).
    pub tool: String,
    /// Unix seconds at capture.
    pub created_unix: u64,
    /// The user-turn ordinal the checkpoint was taken under, when known — lets a GUI correlate a
    /// workspace checkpoint with a conversation rewind anchor ([`daemon_protocol::RewindAnchor`]).
    #[serde(default)]
    pub turn_ordinal: Option<u64>,
    /// The merged-log cursor (`seq`) at capture, when known — the other half of the rewind/checkpoint
    /// correlation (the live-log position the checkpoint lines up with).
    #[serde(default)]
    pub cursor: Option<u64>,
}

/// A transport-stable mirror of the durable session lifecycle (decoupled from `daemon-store`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    /// A live incarnation is (or was) running.
    Active,
    /// Suspended awaiting a background job.
    Suspended {
        /// The job this session is waiting on.
        job_id: String,
    },
    /// A completion is recorded; resumable.
    Ready,
    /// Terminal.
    Completed,
    /// State could not be resolved.
    Unknown,
}

/// The orchestration fleet roster + folded usage.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetReport {
    /// The ids of all registered children.
    pub children: Vec<UnitId>,
    /// The folded fleet usage total.
    pub usage: UsageDelta,
}

// `UnitKind`, `UnitState`, `UnitNode`, and `TreeReport` are defined in `daemon-protocol` (next to
// `Outbound`) so the management contract can carry the projection seam without depending on this
// surface crate; they are re-exported here so every transport and the cddl mirror are unchanged.
pub use daemon_protocol::{SessionRole, TreeReport, UnitKind, UnitNode, UnitState};

// `ManageEventView` is defined in `daemon-protocol` (so the `ManagedUnit` projection seam can carry
// it without a surface-crate edge) and re-exported here unchanged.
pub use daemon_protocol::{ManageEventView, SubagentPhase};

/// A live, push-based stream of [`TreeEvent`]s — the delivery shape a streaming transport returns
/// from [`ControlApi::tree_subscribe`]. Like [`LogStream`], streaming is a *transport capability*;
/// the one-shot/poll form is [`ControlApi::tree`].
pub type TreeStream = BoxStream<'static, TreeEvent>;

/// A churn-control filter for [`ControlApi::tree_subscribe`]. The tree is mostly stable
/// `ManagedChild` topology punctuated by transient `EphemeralSubagent` churn, so a client can opt
/// out of ephemeral nodes entirely or have rapid ephemeral changes coalesced into one update per
/// window.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeSubFilter {
    /// Whether to deliver `EphemeralSubagent` node changes at all (`false` = stable topology only).
    #[serde(default = "default_true")]
    pub include_ephemeral: bool,
    /// If set, coalesce changes into at most one update per this many milliseconds (debounce); the
    /// emitted snapshot reflects the latest state. `None` = deliver every change.
    #[serde(default)]
    pub coalesce_ms: Option<u64>,
}

impl Default for TreeSubFilter {
    fn default() -> Self {
        Self {
            include_ephemeral: true,
            coalesce_ms: None,
        }
    }
}

fn default_true() -> bool {
    true
}

/// One push update on the [`ControlApi::tree_subscribe`] stream. The foundation is a coalesced
/// whole-tree snapshot (simple, churn-bounded); finer node-delta events can fill in later behind the
/// same wire shape.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeEvent {
    /// The current tree snapshot (already scope/coalesce-filtered for the subscriber).
    Snapshot(TreeReport),
    /// A subagent/delegation lifecycle signal (mirrors [`ManageEventView::Subagent`]) for clients
    /// that want the churn signal without diffing snapshots.
    Subagent(ManageEventView),
}

/// Where to rewind a session to — the unified rewind address spanning conversation rewind (truncate
/// the transcript at `anchor`) and the optional workspace rollback. A single
/// [`ControlApi::rewind`] op replaces the separate conversation-`RewindTo` and workspace-only
/// [`ControlApi::checkpoint_rewind`] paths.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewindPoint {
    /// The conversation anchor to truncate the transcript at.
    pub anchor: RewindAnchor,
    /// Whether to also roll the workspace back to the checkpoint captured at/just before `anchor`.
    #[serde(default)]
    pub restore_workspace: bool,
}

/// Where a cataloged ACP agent's launch recipe came from.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcpSource {
    /// From the curated builtin known-agent recipe table.
    Builtin,
    /// Manually registered by an operator (via [`ControlApi::acp_register`]).
    Manual,
    /// A network endpoint (TCP / stdio-bus / remote), not a PATH binary.
    Endpoint,
}

/// A launch recipe for a foreign ACP agent — the catalog's mirror of the host's spawn spec. Either a
/// stdio subprocess (`program` + `args` + `env`) or a network `endpoint`; exactly one is meaningful
/// per `source`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcpRecipe {
    /// The program to exec for a stdio agent (the candidate binary, possibly an adapter shim).
    #[serde(default)]
    pub program: Option<String>,
    /// Arguments passed to the program (e.g. an adapter subcommand/flag).
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables for the spawned agent.
    #[serde(default)]
    pub env: Vec<(String, String)>,
    /// A network endpoint for a non-PATH agent (`source = Endpoint`), e.g. `tcp://host:port`.
    #[serde(default)]
    pub endpoint: Option<String>,
}

/// One entry in the ACP agent catalog ([`ControlApi::acp_catalog`] / [`ControlApi::acp_discover`]):
/// a known/registered agent, whether it is installed, and the ACP `initialize`-verified metadata.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcpAgentEntry {
    /// The agent display name (catalog key, e.g. `"gemini"`, `"goose"`, `"claude-via-zed"`).
    pub name: String,
    /// How to launch it.
    pub recipe: AcpRecipe,
    /// Where the recipe came from.
    pub source: AcpSource,
    /// Whether a candidate binary/endpoint was found (PATH/well-known/endpoint probe).
    #[serde(default)]
    pub installed: bool,
    /// The ACP protocol version the agent reported at `initialize`, when probed.
    #[serde(default)]
    pub version: Option<String>,
    /// Agent capabilities advertised at `initialize` (opaque key/value), when probed.
    #[serde(default)]
    pub capabilities: Vec<(String, String)>,
}

/// A push stream of [`LogLine`]s ([`ControlApi::logs`]). Streaming is a transport capability, like
/// [`LogStream`].
pub type LogLineStream = BoxStream<'static, LogLine>;

/// A filter for the node log-tail stream ([`ControlApi::logs`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogFilter {
    /// Minimum level to deliver (e.g. `"info"`, `"warn"`, `"error"`); `None` = all.
    #[serde(default)]
    pub min_level: Option<String>,
    /// Restrict to a resident-service/target substring, when set.
    #[serde(default)]
    pub target: Option<String>,
}

/// One node log line.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogLine {
    /// Unix-millis timestamp.
    pub ts_ms: u64,
    /// The level (`"info"` / `"warn"` / `"error"` / ...).
    pub level: String,
    /// The emitting target/service.
    pub target: String,
    /// The rendered message.
    pub message: String,
}

/// A runtime provider-registry entry (I11 stub DTO): a provider *type* the node can resolve engines
/// against (the cloud/network builder selection frozen at assembly today).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// The provider id/name (e.g. `"anthropic"`, `"openai"`, `"genai"`).
    pub name: String,
    /// A base URL override, when the provider is endpoint-configurable.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Whether the node currently has this provider wired/usable.
    #[serde(default)]
    pub available: bool,
}

/// A node tool entry (I12 stub DTO).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInfo {
    /// The tool name (as used in `ProfileSpec.tool_allowlist`).
    pub name: String,
    /// A short human description, when known.
    #[serde(default)]
    pub description: Option<String>,
}

// ---------------------------------------------------------------------------
// Command surface (daemon-authoritative operator/user commands)
// ---------------------------------------------------------------------------
//
// The daemon owns the command catalog and its execution; a GUI/TUI is a thin client that renders
// [`command_list`](ControlApi::command_list) and dispatches [`command_invoke`](ControlApi::command_invoke).
// This is distinct from [`ToolInfo`]/[`ControlApi::tool_list`] (model-facing tools the LLM calls):
// commands are human-invoked. The Rust port of hermes' `CommandDef` registry — a single declarative
// catalog decoupled from handlers; built-in handlers are thin adapters over existing typed `NodeApi`
// ops, so no operation logic is duplicated.

/// The minimum access tier required to run a command — the `slash_access.py` analog.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandAccess {
    /// Any authenticated user may run it (the read-only floor: e.g. `help`/`whoami`/`status`).
    #[default]
    User,
    /// Operator/admin only — mutating or node-wide ops.
    Admin,
}

/// Whether a command applies to a specific session or the node as a whole.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandScope {
    /// Operates on a specific session (an invocation should carry a [`SessionId`]).
    #[default]
    Session,
    /// Operates on the node as a whole (no session required).
    Node,
}

/// A client surface a command is exposed on. An empty `surfaces` list means "all surfaces" (the
/// common case); a non-empty list restricts the command (the hermes `cli_only`/`gateway_only` analog).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandSurface {
    /// A terminal/CLI shell.
    Cli,
    /// A graphical client (desktop/web composer).
    Gui,
    /// A messaging/chat transport.
    Chat,
}

/// A declarative command-catalog entry (the `CommandDef` analog). **Metadata only** — the handler
/// lives in the node-side command registry. Clients render menus + autocomplete from this and never
/// model commands themselves.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSpec {
    /// Canonical name without the leading slash (e.g. `"lcm"`, `"new"`).
    pub name: String,
    /// Alternative names that resolve to the same command (e.g. `["reset"]` for `new`).
    #[serde(default)]
    pub aliases: Vec<String>,
    /// One-line human description.
    #[serde(default)]
    pub summary: String,
    /// Grouping for client menus (e.g. `"Session"`, `"Context"`, `"Memory"`, `"Info"`).
    #[serde(default)]
    pub category: String,
    /// Short argument placeholder hint (e.g. `"<prompt>"`, `"[name]"`).
    #[serde(default)]
    pub args_hint: String,
    /// Tab-completable subcommands (e.g. `["status", "doctor", "backup", "rotate", "preset"]`).
    #[serde(default)]
    pub subcommands: Vec<String>,
    /// Whether the command applies to a session or the whole node.
    #[serde(default)]
    pub scope: CommandScope,
    /// The surfaces the command is exposed on (empty = all).
    #[serde(default)]
    pub surfaces: Vec<CommandSurface>,
    /// Whether the command mutates durable state (clients treat it as non-idempotent).
    #[serde(default)]
    pub side_effecting: bool,
    /// A UX hint that the client should confirm before running (e.g. destructive `apply` variants).
    #[serde(default)]
    pub confirm: bool,
    /// The minimum access tier required to run it.
    #[serde(default)]
    pub min_access: CommandAccess,
    /// The subsystem that owns the handler (e.g. `"node"`, `"lcm"`, `"mnemosyne"`) — diagnostics only.
    #[serde(default)]
    pub source: String,
}

/// A request to run a command. `args` is the raw trailing argument string (handlers parse it, like
/// hermes' `fn(raw_args: str)`); `session` is required for [`CommandScope::Session`] commands.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandInvocation {
    /// The command name or alias (with or without a leading slash).
    pub name: String,
    /// The raw trailing argument string (may be empty).
    #[serde(default)]
    pub args: String,
    /// The target session, for session-scoped commands.
    #[serde(default)]
    pub session: Option<SessionId>,
    /// Optional caller attribution, for access decisions and audit.
    #[serde(default)]
    pub origin: Option<Origin>,
}

/// The rendered result of a command invocation.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandOutput {
    /// The rendered output text (the client displays it).
    pub text: String,
    /// When `true`, the output is client-local feedback and must not enter the transcript/journal.
    #[serde(default)]
    pub ephemeral: bool,
}

/// An opaque view of the node's runtime config (I13 stub DTO). Carried as a serialized blob so the
/// concrete `NodeConfig` (a binary-layer type) need not leak into the contract; the shape is the
/// stable wire envelope, the encoding fills in with the runtime.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeConfigView {
    /// The config encoding discriminator (e.g. `"toml"` / `"json"`).
    pub format: String,
    /// The serialized config body.
    pub body: String,
}

/// How the scheduler treats a due job whose previous run is still in flight (I15). The default
/// [`OverlapPolicy::Skip`] also closes the manual-trigger-vs-tick double-fire race: a job already
/// running is not fired again.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverlapPolicy {
    /// Skip this fire if the job's previous run has not finished (default).
    #[default]
    Skip,
    /// Fire regardless — allow concurrent runs of the same job.
    Allow,
    /// Defer the fire until the in-flight run finishes, then run once.
    Queue,
}

/// How the scheduler treats a job that is overdue when a tick observes it (I15) — e.g. after the
/// node was down across the fire time. Within a grace window a missed fire still runs once; beyond
/// it the schedule fast-forwards to the next future occurrence (no thundering herd).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatchUpPolicy {
    /// Fire once if overdue within the grace window; otherwise fast-forward (default).
    #[default]
    Grace,
    /// Never fire a missed occurrence; always fast-forward to the next future fire.
    Skip,
    /// Always fire once for the most recent missed occurrence, regardless of how late.
    Always,
}

/// What initiated a recorded cron run (I15): the scheduler's own clock, or an explicit operator/GUI
/// "run now".
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunTrigger {
    /// Fired by the scheduler at its scheduled time (default).
    #[default]
    Scheduled,
    /// Fired explicitly via `cron_trigger` (operator/GUI "run now").
    Manual,
}

/// A scheduled-job spec (I15): when to fire and what to do. Beyond the schedule + payload the spec
/// carries lifecycle (`enabled`/`repeat`), correctness policy (`overlap`/`catch_up`), spread
/// (`jitter_secs`), locale (`timezone`), a script-only (`no_agent`) path, output-chaining
/// (`context_from`), delivery routing (`deliver`), and per-job run shaping (`enabled_toolsets`,
/// `workdir`, `model`/`provider`). All fields past `payload` are additive and default-friendly.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronSpec {
    /// A human name for the job.
    pub name: String,
    /// The schedule expression (cron syntax via `croner`, `@every <dur>`/ISO interval, or a single
    /// ISO timestamp for a one-shot — parsed by `daemon-schedule`).
    pub schedule: String,
    /// The session/profile the job drives, when it targets one.
    #[serde(default)]
    pub target: Option<String>,
    /// The opaque payload/command the job submits when it fires (the agent prompt, unless
    /// `no_agent`).
    #[serde(default, with = "serde_bytes")]
    pub payload: Vec<u8>,
    /// Whether the scheduler considers this job. A paused job stays in the store but never fires.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// IANA timezone for cron-expression evaluation (e.g. `"Europe/Berlin"`); `None` uses the node
    /// default. Interval/one-shot schedules are timezone-agnostic.
    #[serde(default)]
    pub timezone: Option<String>,
    /// Maximum number of fires before the job auto-deletes; `None` = forever.
    #[serde(default)]
    pub repeat: Option<u32>,
    /// Random delay (0..=jitter_secs) applied to each fire, to spread herds of identically-scheduled
    /// jobs; `None`/`0` = fire exactly on time.
    #[serde(default)]
    pub jitter_secs: Option<u32>,
    /// Behavior when a due fire overlaps the job's previous still-running run.
    #[serde(default)]
    pub overlap: OverlapPolicy,
    /// Behavior when the job is observed overdue (missed-fire catch-up).
    #[serde(default)]
    pub catch_up: CatchUpPolicy,
    /// A node-scripts-relative path to run instead of (or before) an agent turn.
    #[serde(default)]
    pub script: Option<String>,
    /// Run `script` only — no LLM turn. Requires `script`.
    #[serde(default)]
    pub no_agent: bool,
    /// Job ids whose most recent run output is injected into this job's seed prompt (output
    /// chaining).
    #[serde(default)]
    pub context_from: Vec<String>,
    /// Delivery routing for the run result (e.g. `"origin"`, `"all"`, `"<transport>:<chat>"`);
    /// `None` = store-only (no transport delivery).
    #[serde(default)]
    pub deliver: Option<String>,
    /// Restrict the cron run's toolset to these tool names; `None` = the cron-run default policy.
    #[serde(default)]
    pub enabled_toolsets: Option<Vec<String>>,
    /// Absolute working directory the run is bound to (context + exec root); `None` = node default.
    #[serde(default)]
    pub workdir: Option<String>,
    /// Per-job model override; `None` = the bound profile's model.
    #[serde(default)]
    pub model: Option<String>,
    /// Per-job provider override; `None` = the bound profile's provider.
    #[serde(default)]
    pub provider: Option<String>,
    /// Ordered skill names the scheduler preloads (their `skill_view` body injected ahead of the
    /// seed prompt) before an agent run, so a cron job carries the same skill context a chat would
    /// (v16; mirrors Hermes' `cronjob` `skills`). Empty = none. Ignored for `no_agent` jobs.
    #[serde(default)]
    pub skills: Vec<String>,
    /// The originating context captured at create time — the chat/session that asked for the job — so
    /// `deliver = "origin"` can route a run's result back to its creator (resolved through the same
    /// routing surface a live submit uses). `None` for jobs created without a routable origin (e.g.
    /// the CLI, or a node-internal creation). (wire v17)
    #[serde(default)]
    pub origin: Option<Origin>,
}

impl Default for CronSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            schedule: String::new(),
            target: None,
            payload: Vec::new(),
            enabled: true,
            timezone: None,
            repeat: None,
            jitter_secs: None,
            overlap: OverlapPolicy::default(),
            catch_up: CatchUpPolicy::default(),
            script: None,
            no_agent: false,
            context_from: Vec::new(),
            deliver: None,
            enabled_toolsets: None,
            workdir: None,
            model: None,
            provider: None,
            skills: Vec::new(),
            origin: None,
        }
    }
}

/// A scheduled job (I15): a [`CronSpec`] plus its id, next-fire time, and run bookkeeping so a GUI
/// list can show status without a `cron_runs` round-trip.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronJob {
    /// The opaque job id.
    pub id: String,
    /// The job spec.
    pub spec: CronSpec,
    /// Unix seconds of the next scheduled fire, when computed.
    #[serde(default)]
    pub next_fire_unix: Option<u64>,
    /// Whether the job is currently paused (mirror of `!spec.enabled`, surfaced for convenience).
    #[serde(default)]
    pub paused: bool,
    /// Unix seconds the job last fired, when it has.
    #[serde(default)]
    pub last_run_unix: Option<u64>,
    /// Whether the last completed run succeeded, when one has completed.
    #[serde(default)]
    pub last_ok: Option<bool>,
    /// A rendered detail of the last run (error text or summary), when present.
    #[serde(default)]
    pub last_detail: Option<String>,
    /// How many times the job has fired (for `repeat` accounting).
    #[serde(default)]
    pub fire_count: u32,
}

/// One recorded run of a scheduled job (I15).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronRun {
    /// Unix seconds the run started.
    pub started_unix: u64,
    /// Whether the run succeeded.
    pub ok: bool,
    /// A rendered outcome detail, when present.
    #[serde(default)]
    pub detail: Option<String>,
    /// Unix seconds the run finished, when it has completed.
    #[serde(default)]
    pub finished_unix: Option<u64>,
    /// The isolated `cron_{id}_{ts}` session the run fired, when an agent turn was materialized.
    #[serde(default)]
    pub session: Option<SessionId>,
    /// What initiated this run.
    #[serde(default)]
    pub trigger: RunTrigger,
}

/// The lifecycle of a [`CronSuggestion`] (I15): a consent-first proposal the operator accepts (which
/// creates the job) or dismisses (latched by `dedup_key` so it is never re-offered).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionStatus {
    /// Offered, awaiting an operator decision (default).
    #[default]
    Pending,
    /// Accepted — the backing job was created.
    Accepted,
    /// Dismissed — latched so it is not re-offered.
    Dismissed,
}

/// A ready-to-create cron job the node surfaces for operator consent (I15): catalog starters and
/// filled blueprints compile to one of these. Accepting it calls `cron_create(spec)`; there is no
/// second job engine.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronSuggestion {
    /// The opaque suggestion id.
    pub id: String,
    /// A short title for the proposal.
    pub title: String,
    /// A human description of what the job does.
    #[serde(default)]
    pub description: String,
    /// Where the suggestion came from (e.g. `"catalog"`, `"blueprint"`).
    #[serde(default)]
    pub source: String,
    /// The exact spec `cron_create` runs when the suggestion is accepted.
    pub spec: CronSpec,
    /// A stable key; once dismissed/accepted, a suggestion with the same key is never re-offered.
    pub dedup_key: String,
    /// The proposal's lifecycle.
    #[serde(default)]
    pub status: SuggestionStatus,
}

/// A chat→session routing pin (I5; daemon-event-io-spec §5.9): an explicit binding of an inbound
/// [`Origin`] to a specific session (+ optional profile + session-naming isolation), surfaced by the
/// `routing_*` ops so a GUI/operator can pin a chat to a named conversation. The host consults pins
/// **resolve-first**, overriding the deterministic `session_id_for` derivation.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRoute {
    /// The inbound origin (transport instance + scope) this pin matches.
    pub origin: Origin,
    /// The session the origin is pinned to.
    pub session: SessionId,
    /// An explicit profile to run the pinned session under (`None` = registry default precedence).
    #[serde(default)]
    pub profile: Option<ProfileRef>,
    /// The session-naming isolation the pin records (informational; the pinned id is authoritative).
    #[serde(default = "default_isolation")]
    pub isolation: IsolationPolicy,
}

/// `serde` default for [`ChatRoute::isolation`]: `PerThread`, matching the routing registry's
/// deterministic default naming when no binding overrides it.
fn default_isolation() -> IsolationPolicy {
    IsolationPolicy::PerThread
}

/// A room/chat a transport instance knows about ([`ControlApi::transport_rooms`], I5): the read-only
/// enumeration a GUI lists when binding a chat to a session. Carries the pinned session, when one
/// exists, so the GUI can show which rooms are already routed.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoomInfo {
    /// The transport instance the room belongs to.
    pub transport: TransportId,
    /// The room/chat handle (adapter-opaque).
    pub room: String,
    /// A human-readable room name, when known.
    #[serde(default)]
    pub name: Option<String>,
    /// The session this room is currently pinned to, when one exists.
    #[serde(default)]
    pub session: Option<SessionId>,
}

// ---------------------------------------------------------------------------
// Transport-adapter framework DTOs (daemon-transport-adapter-spec.md)
//
// The declarative layer over events-IO transport adapters: the descriptor + capabilities a GUI
// reads to render the "Add channel" picker and capability-gate affordances, and the live per-account
// connection/presence state for the status bar + unified roster. Populated by the adapters registered
// in the host `AdapterRegistry` (e.g. `daemon-matrix`, `daemon-rooms`); the `transport_adapters` /
// `transport_instances` ControlApi methods default empty only on a node with no registry.
// ---------------------------------------------------------------------------

/// The live connection state of a transport instance (the Pidgin `PurpleConnectionState` analogue),
/// surfaced per account so the GUI status bar can show a status dot.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    /// Not connected (disabled, signed out, or never started).
    #[default]
    Offline,
    /// A connect/login attempt is in flight.
    Connecting,
    /// Connected and serving.
    Connected,
    /// Failed (the adapter logs the specifics; this carries only the coarse state).
    Error,
}

/// A normalized presence primitive (libpurple `PurplePresencePrimitive` / Kopete `OnlineStatus`
/// category / Adium `AIStatusSummary`). Each adapter maps its wire-format presence into this so the
/// unified roster sorts/filters uniformly across transports.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresenceState {
    /// Presence is unknown / not reported by this transport.
    #[default]
    Unknown,
    /// Offline.
    Offline,
    /// Available / online.
    Available,
    /// Idle.
    Idle,
    /// Away.
    Away,
    /// Busy / do-not-disturb.
    Busy,
}

/// What an events-IO transport adapter can do — the capability descriptor generic UI reads to
/// capability-gate affordances (join channel, invite, set topic, send file) instead of switching on
/// a transport-family string. The daemon analogue of libpurple's `implements_*()` probes / Kopete's
/// `Capability` flags / Adium's per-service bool flags (daemon-transport-adapter-spec.md §3.2).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterCapabilities {
    /// Supports group/channel conversations (`OriginScope::Group`).
    pub rooms: bool,
    /// Supports 1:1 direct messages (`OriginScope::Dm`).
    pub direct_messages: bool,
    /// Reports per-account / peer presence.
    pub presence: bool,
    /// Supports live room/conversation enumeration (not merely routing pins).
    pub room_enumeration: bool,
    /// Supports file/attachment transfer.
    pub file_transfer: bool,
    /// Drives an interactive login (the `AuthApi` flow).
    pub interactive_auth: bool,
}

/// The typed account-setup form for an adapter — the generalisation of
/// [`AuthProviderInfo::params_schema`] beyond interactive-auth flows. An adapter with no login (the
/// HTTP surface, the internal Rooms loopback) still describes its instance config (a listen address,
/// a room-id prefix) with the same [`AuthParamField`] shape.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountSettingsSchema {
    /// The fields the client collects to configure one account/instance of this adapter.
    pub fields: Vec<AuthParamField>,
}

/// A self-describing events-IO transport adapter (the declarative analogue of libpurple's
/// `PurpleProtocol` / Kopete's `Kopete::Protocol` / Adium's `AIService`). Enumerated by
/// [`ControlApi::transport_adapters`] so the GUI renders the "Add channel" picker and
/// capability-gates UI (daemon-transport-adapter-spec.md §3).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterInfo {
    /// The transport family / adapter id (`"matrix"`, `"room"`, `"http"`, `"a2a"`).
    pub family: String,
    /// A human display name (`"Matrix"`, `"Rooms (internal)"`).
    pub display_name: String,
    /// What this adapter supports (drives capability-gated affordances).
    pub capabilities: AdapterCapabilities,
    /// The account-setup form for a new instance of this adapter.
    #[serde(default)]
    pub account_schema: AccountSettingsSchema,
}

/// One configured transport instance (account) plus its live status — what the GUI status bar and
/// the unified conversation roster render (the Pidgin/Kopete per-account status-icon analogue).
/// Closes the per-account connection-state gap (`EIO-9`) the channels user stories track.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportInstanceInfo {
    /// The instance-qualified transport id (e.g. `matrix/@bot:hs.org`, `room`).
    pub transport: TransportId,
    /// The adapter family this instance belongs to.
    pub family: String,
    /// A human label for the account (e.g. the resolved `@user:hs.org`).
    pub display_name: String,
    /// The live connection state.
    #[serde(default)]
    pub connection: ConnectionState,
    /// The reported presence (or `Unknown`).
    #[serde(default)]
    pub presence: PresenceState,
    /// The profile this instance is bound to, when one is.
    #[serde(default)]
    pub bound_profile: Option<ProfileRef>,
}

/// A self-describing events-IO transport adapter — the declarative analogue of libpurple's
/// `PurpleProtocol`, Kopete's `Kopete::Protocol`, and Adium's `AIService`
/// (daemon-transport-adapter-spec.md §3.1). It adds *identity + capabilities + a lifecycle entry
/// point* on top of the existing per-adapter mechanics: `serve` is expected to wire the reusable
/// `daemon-ingest` (inbound) + `daemon-delivery` (outbound) halves exactly as the bespoke
/// `serve(api, cfg)` functions do today — this trait does **not** reimplement them. Co-located with
/// the capability DTOs so an adapter crate (which already depends on `daemon-api`) implements it
/// without a new dependency, and the host `AdapterRegistry` can hold `Arc<dyn TransportAdapter>`.
///
/// Status: `daemon-matrix` and `daemon-rooms` implement this trait (and the `MessagingProtocol`
/// specialization + feature traits below); they are registered in the host `AdapterRegistry`, which
/// enumerates `info()`/`instances()` and drives `serve` via `AdapterRegistry::spawn_all`. Only
/// `daemon-http` (and the future `daemon-a2a`) remain to be retrofitted onto the trait.
#[async_trait]
pub trait TransportAdapter: Send + Sync {
    /// The transport family / adapter id (`"matrix"`, `"room"`, `"http"`, `"a2a"`).
    fn family(&self) -> &str;

    /// The descriptor the GUI reads (display name, capabilities, account-setup schema).
    fn info(&self) -> AdapterInfo;

    /// Drive the transport until shutdown, wiring `daemon-ingest`/`daemon-delivery`.
    /// Registry-spawned via [`crate`]-side `AdapterRegistry::spawn_all`.
    async fn serve(self: std::sync::Arc<Self>, api: std::sync::Arc<dyn NodeApi>);

    /// Configured instances (accounts) with live connection/presence state (the daemon analogue of
    /// libpurple's account manager). Default: empty.
    async fn instances(&self) -> Vec<TransportInstanceInfo> {
        Vec::new()
    }

    /// Is this transport a messaging protocol (the libpurple `PurpleProtocol` analogue)? Generic
    /// (non-chat) transports return `None` (daemon-messaging-adapter-spec.md §3.1).
    fn messaging(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn MessagingProtocol>> {
        None
    }
}

/// A messaging protocol — the faithful port of libpurple's `PurpleProtocol`: a [`TransportAdapter`]
/// that additionally validates accounts and exposes the optional conversation / membership / roster /
/// contacts / directory / file-transfer feature interfaces (daemon-messaging-adapter-spec.md §3.1.1).
#[async_trait]
pub trait MessagingProtocol: TransportAdapter {
    /// Validate proposed account settings (← `validate_account`). Default: Ok.
    async fn validate_account(&self, _settings: &AccountSettingsValues) -> Result<(), ApiError> {
        Ok(())
    }

    fn conversations(
        self: std::sync::Arc<Self>,
    ) -> Option<std::sync::Arc<dyn SupportsConversations>> {
        None
    }
    fn membership(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn SupportsMembership>> {
        None
    }
    fn roster(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn SupportsRoster>> {
        None
    }
    fn contacts(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn SupportsContacts>> {
        None
    }
    fn directory(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn SupportsDirectory>> {
        None
    }
    fn file_transfer(
        self: std::sync::Arc<Self>,
    ) -> Option<std::sync::Arc<dyn SupportsFileTransfer>> {
        None
    }
}

/// Per-verb capability probe for [`SupportsConversations`] (← libpurple's `implements_*`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationOps {
    pub create: bool,
    pub join_channel: bool,
    pub leave: bool,
    pub delete: bool,
    pub send: bool,
    pub set_topic: bool,
    pub set_title: bool,
    pub set_description: bool,
}

/// Conversation management (← `purpleprotocolconversation.h`). Methods default to
/// `Err(ApiError::Unsupported)` / empty; an adapter overrides what it supports and reports it in
/// [`ConversationOps`]. There is deliberately no `invite` verb here (membership is [`SupportsMembership`]).
#[async_trait]
pub trait SupportsConversations: Send + Sync {
    fn supported(&self) -> ConversationOps;

    async fn list(&self, _transport: TransportId) -> Vec<ConversationInfo> {
        Vec::new()
    }
    async fn get(&self, _transport: TransportId, _conv: String) -> Option<ConversationInfo> {
        None
    }

    async fn create_details(&self, _transport: TransportId) -> CreateConversationDetails {
        CreateConversationDetails::default()
    }
    async fn create(
        &self,
        _transport: TransportId,
        _details: CreateConversationDetails,
    ) -> Result<ConversationInfo, ApiError> {
        Err(ApiError::Unsupported("conv_create".into()))
    }

    async fn channel_join_details(&self, _transport: TransportId) -> ChannelJoinDetails {
        ChannelJoinDetails::default()
    }
    async fn join_channel(
        &self,
        _transport: TransportId,
        _details: ChannelJoinDetails,
    ) -> Result<ConversationInfo, ApiError> {
        Err(ApiError::Unsupported("conv_join".into()))
    }

    async fn leave(&self, _transport: TransportId, _conv: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_leave".into()))
    }
    async fn delete(&self, _transport: TransportId, _conv: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_delete".into()))
    }
    async fn send(
        &self,
        _transport: TransportId,
        _conv: String,
        _from: Option<Participant>,
        _message: UserMsg,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_send".into()))
    }
    async fn set_topic(
        &self,
        _transport: TransportId,
        _conv: String,
        _topic: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_set_topic".into()))
    }
    async fn set_title(
        &self,
        _transport: TransportId,
        _conv: String,
        _title: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_set_title".into()))
    }
    async fn set_description(
        &self,
        _transport: TransportId,
        _conv: String,
        _description: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("conv_set_description".into()))
    }
}

/// Per-verb probe for [`SupportsMembership`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MembershipOps {
    pub invite: bool,
    pub remove: bool,
    pub ban: bool,
    pub set_role: bool,
}

/// Membership administration of an existing conversation (daemon-messaging-adapter-spec.md §3.2.1):
/// invite is first-class cross-protocol (libpurple 2 `chat_invite`, Adium, Kopete); kick/ban/role are
/// optional. Methods default to `Err(ApiError::Unsupported)`.
#[async_trait]
pub trait SupportsMembership: Send + Sync {
    fn supported(&self) -> MembershipOps;
    async fn invite(
        &self,
        _transport: TransportId,
        _conv: String,
        _who: Participant,
        _message: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_invite".into()))
    }
    async fn remove(
        &self,
        _transport: TransportId,
        _conv: String,
        _who: Participant,
        _reason: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_remove".into()))
    }
    async fn ban(
        &self,
        _transport: TransportId,
        _conv: String,
        _who: Participant,
        _reason: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_ban".into()))
    }
    async fn set_role(
        &self,
        _transport: TransportId,
        _conv: String,
        _who: Participant,
        _role: MemberRole,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_set_role".into()))
    }
}

/// Per-verb probe for [`SupportsRoster`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterOps {
    pub add: bool,
    pub update: bool,
    pub remove: bool,
}

/// Account-level server-side contact list (← `purpleprotocolroster.h`). Defined; no adapter yet.
#[async_trait]
pub trait SupportsRoster: Send + Sync {
    fn supported(&self) -> RosterOps;
    async fn add(&self, _transport: TransportId, _contact: ContactInfo) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("roster_add".into()))
    }
    async fn update(&self, _transport: TransportId, _contact: ContactInfo) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("roster_update".into()))
    }
    async fn remove(&self, _transport: TransportId, _contact: ContactInfo) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("roster_remove".into()))
    }
}

/// Per-verb probe for [`SupportsContacts`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContactsOps {
    pub get_profile: bool,
    pub action_menu: bool,
    pub set_alias: bool,
}

/// Remote-contact operations (← `purpleprotocolcontacts.h`). Implemented by `daemon-matrix`
/// (`get_profile`); `action_menu`/`set_alias` off there.
#[async_trait]
pub trait SupportsContacts: Send + Sync {
    fn supported(&self) -> ContactsOps;
    async fn get_profile(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
    ) -> Result<String, ApiError> {
        Err(ApiError::Unsupported("contact_get_profile".into()))
    }
    fn action_menu(&self, _transport: TransportId, _contact: ContactInfo) -> Option<ActionMenu> {
        None
    }
    async fn set_alias(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
        _alias: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("contact_set_alias".into()))
    }
}

/// Contact/room directory search (← `purpleprotocoldirectory.h`; also the libpurple roomlist successor).
#[async_trait]
pub trait SupportsDirectory: Send + Sync {
    fn supported(&self) -> bool;
    async fn search_contacts(
        &self,
        _transport: TransportId,
        _query: Option<String>,
    ) -> Result<Vec<ContactInfo>, ApiError> {
        Err(ApiError::Unsupported("directory_search".into()))
    }
}

/// Per-verb probe for [`SupportsFileTransfer`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileTransferOps {
    pub send: bool,
    pub receive: bool,
}

/// File transfer (← `purpleprotocolfiletransfer.h`). Defined; no adapter yet.
#[async_trait]
pub trait SupportsFileTransfer: Send + Sync {
    fn supported(&self) -> FileTransferOps;
    async fn send(&self, _transfer: FileTransfer) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("file_transfer_send".into()))
    }
    async fn receive(&self, _transfer: FileTransfer) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("file_transfer_receive".into()))
    }
}

// ---------------------------------------------------------------------------
// Messaging-adapter data model (daemon-messaging-adapter-spec.md §5)
// ---------------------------------------------------------------------------

/// ← PurpleConversationType.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationType {
    /// Unset / unknown (faithful round-trip; libpurple keeps `*_UNSET`).
    #[default]
    Unset,
    /// A 1:1 direct message.
    Dm,
    /// A group direct message (a protocol-bounded number of participants).
    GroupDm,
    /// A multi-user channel.
    Channel,
    /// A thread within a conversation.
    Thread,
}

/// ← PurpleTypingState.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypingState {
    /// Not typing.
    #[default]
    None,
    /// Currently typing.
    Typing,
    /// Typed but paused.
    Paused,
}

/// Normalized presence primitive (← `PurplePresencePrimitive`, the faithful 8-value set).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresencePrimitive {
    /// Offline / unknown.
    #[default]
    Offline,
    /// Online and available.
    Available,
    /// Online but idle.
    Idle,
    /// Online but invisible to others.
    Invisible,
    /// Away from the device.
    Away,
    /// Do not disturb.
    DoNotDisturb,
    /// Streaming.
    Streaming,
    /// Out of office.
    OutOfOffice,
}

/// ← PurplePresence: a primitive plus optional status decorations.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Presence {
    /// The presence primitive.
    pub primitive: PresencePrimitive,
    /// A free-text status message, when set.
    #[serde(default)]
    pub message: Option<String>,
    /// A mood emoji, when set.
    #[serde(default)]
    pub emoji: Option<String>,
    /// Whether the peer is on a mobile device.
    #[serde(default)]
    pub mobile: bool,
    /// Unix seconds since which the peer has been idle (`None` = not idle).
    #[serde(default)]
    pub idle_since: Option<u64>,
}

/// ← PurpleContactInfoPermission.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactPermission {
    /// Unset.
    #[default]
    Unset,
    /// Allowed to contact the user.
    Allow,
    /// Denied.
    Deny,
}

/// ← PurpleContactInfo: the information used wherever a remote party is referenced.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContactInfo {
    /// The adapter-opaque contact id.
    pub id: String,
    /// A human display name, when known.
    #[serde(default)]
    pub display_name: Option<String>,
    /// The contact's presence.
    #[serde(default)]
    pub presence: Presence,
    /// Whether the contact may message the user.
    #[serde(default)]
    pub permission: ContactPermission,
}

/// A per-participant role/affiliation (← Adium `AIGroupChatFlags` / libpurple badges / XMPP affiliations).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberRole {
    /// No special role.
    #[default]
    None,
    /// Voice.
    Voice,
    /// Half-operator.
    HalfOp,
    /// Operator.
    Op,
    /// Founder/owner.
    Founder,
}

/// Who an [`SupportsMembership`] op targets, and the [`SupportsConversations::send`] author. `Contact`
/// is the faithful libpurple identity (a human/remote contact); `Agent` is the delineated daemon
/// extension — an agent bound as a participant (`member` is its in-conversation handle, `profile`
/// resolves to a session).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Participant {
    /// A human/remote contact.
    Contact(ContactInfo),
    /// A daemon agent participant.
    Agent {
        /// The profile the agent runs under.
        profile: ProfileRef,
        /// The in-conversation member handle.
        member: String,
    },
}

/// One occupant of a conversation (← PurpleConversationMember). Observed state the protocol populates
/// from sync; `session` is the daemon extension binding the member to an engine incarnation.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationMember {
    /// The participant's contact info.
    pub contact: ContactInfo,
    /// A conversation-local alias.
    #[serde(default)]
    pub alias: Option<String>,
    /// A conversation-local nickname.
    #[serde(default)]
    pub nickname: Option<String>,
    /// The member's typing state.
    #[serde(default)]
    pub typing: TypingState,
    /// The member's observed role/affiliation.
    #[serde(default)]
    pub role: MemberRole,
    /// The engine session this member drives, when it is a daemon agent (daemon extension).
    #[serde(default)]
    pub session: Option<SessionId>,
}

/// A conversation as the host/GUI sees it — the `list`/`get` projection.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationInfo {
    /// The transport that owns the conversation.
    pub transport: TransportId,
    /// The adapter-opaque conversation id within `transport`.
    pub id: String,
    /// The conversation kind.
    pub kind: ConversationType,
    /// A human title, when set.
    #[serde(default)]
    pub title: Option<String>,
    /// The topic, when set.
    #[serde(default)]
    pub topic: Option<String>,
    /// A description, when set.
    #[serde(default)]
    pub description: Option<String>,
    /// The observed members/occupants.
    #[serde(default)]
    pub members: Vec<ConversationMember>,
}

/// Filled values for an adapter-described settings form (the companion to [`AccountSettingsSchema`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountSettingsValues {
    /// The filled `key -> value` pairs (keyed by [`AuthParamField::key`]).
    #[serde(default)]
    pub values: BTreeMap<String, String>,
}

/// ← PurpleCreateConversationDetails: the typed common core plus adapter-described extras the UI fills
/// before `create` (e.g. the Rooms floor policy rides in `extras`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateConversationDetails {
    /// Max participants (`0` = unlimited).
    #[serde(default)]
    pub max_participants: u32,
    /// The initial participants (create-time only).
    #[serde(default)]
    pub participants: Vec<ContactInfo>,
    /// The adapter-provided extras form.
    #[serde(default)]
    pub extras_schema: AccountSettingsSchema,
    /// The filled extras (e.g. room name + policy).
    #[serde(default)]
    pub extras: AccountSettingsValues,
}

/// ← PurpleChannelJoinDetails.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelJoinDetails {
    /// The channel name.
    #[serde(default)]
    pub name: Option<String>,
    /// Max channel-name length (`0` = no limit).
    #[serde(default)]
    pub name_max_length: u32,
    /// The user's channel nickname.
    #[serde(default)]
    pub nickname: Option<String>,
    /// Whether per-channel nicknames are supported.
    #[serde(default)]
    pub nickname_supported: bool,
    /// Max nickname length.
    #[serde(default)]
    pub nickname_max_length: u32,
    /// The channel password.
    #[serde(default)]
    pub password: Option<String>,
    /// Whether passwords are supported.
    #[serde(default)]
    pub password_supported: bool,
    /// Max password length.
    #[serde(default)]
    pub password_max_length: u32,
    /// The adapter-provided extras form.
    #[serde(default)]
    pub extras_schema: AccountSettingsSchema,
    /// The filled extras.
    #[serde(default)]
    pub extras: AccountSettingsValues,
}

/// Minimal avatar carrier for the (deferred) avatar/file interfaces.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Image {
    /// The content-addressed image blob.
    pub blob: BlobRef,
}

/// Minimal contact action-menu carrier (← `BirbActionMenu`; deferred).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionMenu {
    /// The action labels.
    pub items: Vec<String>,
}

/// Minimal file-transfer carrier (← `PurpleFileTransfer`; deferred).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileTransfer {
    /// The file name.
    pub name: String,
    /// The content-addressed blob.
    pub blob: BlobRef,
}

// ---------------------------------------------------------------------------
// Verifiable journal read DTOs (the non-destructive reconnect / scroll-back surface)
// ---------------------------------------------------------------------------

/// The decoded payload of one journal entry: a coarse management lifecycle record or a coalesced
/// finished chat block (a `daemon-protocol` [`TranscriptBlock`], already decoded for the consumer —
/// the GUI renders it directly, an auditor reads one timeline). Streaming deltas never appear here;
/// they stay on the ephemeral live drains.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JournalRecordPayload {
    /// A management lifecycle / credential-audit record (its human/structured detail).
    Management {
        /// The record detail.
        detail: String,
    },
    /// A finished chat block, decoded from the entry's opaque body.
    Block {
        /// The decoded transcript block.
        block: TranscriptBlock,
    },
}

/// One decoded + verified journal entry, as returned by a history read.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalRecord {
    /// The stream-monotonic pagination cursor (pass as the next `after_cursor`).
    pub cursor: u64,
    /// The segment (turn / incarnation) this entry belongs to.
    pub segment: u64,
    /// Monotonic per-`(stream, segment)` sequence number.
    pub seq: u64,
    /// The incarnation epoch active when recorded (0 for non-durable / first turn).
    pub epoch: u64,
    /// The correlation trace context active when recorded.
    pub trace: u64,
    /// The entry's kind label (e.g. `"mgmt.started"`, `"block.message"`).
    pub kind: String,
    /// Milliseconds since the Unix epoch when the entry was recorded.
    pub timestamp_ms: u64,
    /// Whether the sealed segment carrying this entry verified end-to-end (root recomputed +
    /// signature checked + chain linked). `false` for an as-yet-unsealed (open) segment or when the
    /// node exposes no verifying key.
    pub verified: bool,
    /// The decoded entry payload.
    pub payload: JournalRecordPayload,
}

/// A page of a unit/session's verifiable journal: decoded entries past a cursor plus the pagination
/// cursors. Non-destructive — repeated reads from the same `after_cursor` return the same page.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalPageView {
    /// The decoded entries in cursor order.
    pub entries: Vec<JournalRecord>,
    /// The cursor to pass as `after_cursor` on the next read (the last entry's cursor, or the input
    /// `after_cursor` when the page is empty).
    pub next_cursor: u64,
    /// The highest cursor currently stored for the stream (how far a reader can scroll).
    pub head_cursor: u64,
    /// The active conversation-rewind seal cursor, when the session has been rewound (conversation
    /// rewind spec §6). `Some(c)` means a rewind occurred at stream cursor `c`: the journal remains a
    /// complete audit log, but a reconnecting client should reconcile its view against the engine's
    /// truncated conversation (the authoritative `Snapshot`/`ConvView`) rather than replaying the raw
    /// post-`c` audit tail verbatim. `None` when the session has never been rewound.
    #[serde(default)]
    pub sealed_after: Option<u64>,
}

/// A page of a session's **merged live event log**: the [`SessionLogEntry`] items past a cursor plus
/// the pagination cursors. Non-destructive — repeated reads from the same `after_seq` return the same
/// page. This is the live-log analogue of [`JournalPageView`] (which pages the *durable* journal).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogPageView {
    /// The merged-log entries (inbound + outbound) in `seq` order.
    pub entries: Vec<SessionLogEntry>,
    /// The cursor to pass as the next `after_seq` (the last entry's `seq`, or the input `after_seq`
    /// when the page is empty).
    pub next_seq: u64,
    /// The highest `seq` currently retained for the session (how far a reader can advance now).
    pub head_seq: u64,
    /// The session-activation generation this page belongs to (L2 resync). A fresh in-memory log
    /// after a restart/reactivation carries a strictly greater `epoch`, so a client that tracks
    /// `(epoch, seq)` detects the generation change and re-baselines from the durable journal rather
    /// than mis-applying a new log's entries onto the old one. `0` for the first activation
    /// (matching `Snapshot::fresh`). `#[serde(default)]` keeps old (epoch-less) encodings decodable.
    #[serde(default)]
    pub epoch: u64,
}

// ---------------------------------------------------------------------------
// Node-wide event feed DTOs (L3; daemon-sync-protocol-spec.md §5)
// ---------------------------------------------------------------------------

/// One payload-free node-wide notification (L3 `EventsSince`). A pointer, not a payload: it tells a
/// client that *something* changed out of focus so it can update a badge / mark a roster row stale /
/// nudge a focused turn and then lazily fetch the detail; it never carries transcript/model bytes.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeEvent {
    /// A session's live merged log advanced (coalesced to the latest per session).
    SessionAdvanced {
        /// The session whose log grew.
        session: SessionId,
        /// Its activation epoch (L2).
        epoch: u64,
        /// The highest `seq` now retained.
        head_seq: u64,
    },
    /// A session's roster metadata changed (rename / pin / archive / activity).
    SessionMetaChanged {
        /// The affected session.
        session: SessionId,
        /// The roster revision at the change.
        rev: u64,
    },
    /// The roster set changed (a session opened/closed/moved); the client refetches (delta in L4).
    RosterChanged {
        /// The new roster revision.
        rev: u64,
    },
    /// An approval is pending operator action.
    ApprovalPending {
        /// The session it belongs to.
        session: SessionId,
        /// The approval's request id.
        request_id: String,
    },
    /// A model download advanced (replaces the client's poll).
    DownloadProgress {
        /// The download job id.
        id: DownloadId,
        /// Percent complete (0..=100).
        pct: u32,
        /// The job state string.
        state: String,
    },
    /// The feed could not serve from the client's cursor (aged out / lagged); the client must
    /// re-baseline the named scope ("roster" / "all" / ...).
    ResyncNeeded {
        /// The scope to refetch.
        scope: String,
    },
}

/// A page of the node-wide event feed (`EventsSince` -> `EventsPage`): a batch of [`NodeEvent`]s past
/// the requested cursor plus the feed cursors. Non-destructive; repeated reads from the same cursor
/// return the same page (until it ages out of the retained ring, which surfaces as `ResyncNeeded`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventsPage {
    /// The events in cursor order.
    pub events: Vec<NodeEvent>,
    /// The cursor to pass as the next `cursor` (the last event's cursor, or the input when empty).
    pub next_cursor: u64,
    /// The highest cursor the feed has assigned (how far a reader can advance now).
    pub head_cursor: u64,
}

// ---------------------------------------------------------------------------
// The serializable mirror (1:1 with the interface methods)
// ---------------------------------------------------------------------------

/// The serializable reflection of a call into the interface — what every non-in-process transport
/// marshals onto the wire.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiRequest {
    /// [`SessionApi::submit`] / [`SessionApi::submit_from`] / [`SessionApi::submit_as`].
    Submit {
        /// Target session.
        session: SessionId,
        /// The §17 command.
        command: AgentCommand,
        /// Optional per-event attribution. `None` (old encodings) drops to the host-local default;
        /// `Some` routes through [`SessionApi::submit_from`] so the origin is recorded on the log.
        #[serde(default)]
        origin: Option<Origin>,
        /// Optional explicit profile to bind on open ("open chat as agent X", I9). `Some` routes
        /// through [`SessionApi::submit_as`]; `None` keeps routing-config / default binding.
        #[serde(default)]
        profile: Option<ProfileRef>,
    },
    /// [`SessionApi::submit_routed`]: submit by [`Origin`] and let the host's routing capability pick
    /// the session + profile + delivery. The reply is [`ApiResponse::Routed`] carrying the session.
    SubmitRouted {
        /// The inbound origin to route.
        origin: Origin,
        /// The §17 command.
        command: AgentCommand,
    },
    /// [`SessionApi::poll`].
    Poll {
        /// Target session.
        session: SessionId,
        /// Maximum items to drain.
        max: u32,
    },
    /// [`SessionApi::respond`].
    Respond {
        /// Target session.
        session: SessionId,
        /// The correlated host response.
        response: HostResponse,
    },
    /// [`ControlApi::health`].
    Health,
    /// [`ControlApi::stats`].
    Stats,
    /// [`ControlApi::telemetry`].
    Telemetry,
    /// [`ControlApi::sessions`].
    Sessions,
    /// [`ControlApi::assign`].
    Assign {
        /// Session to assign/wake.
        session: SessionId,
    },
    /// [`ControlApi::cancel`].
    Cancel {
        /// Session to cancel.
        session: SessionId,
    },
    /// [`ControlApi::fleet`].
    Fleet,
    /// [`ControlApi::tree`].
    Tree,
    /// [`ControlApi::unit`].
    Unit {
        /// The unit to view.
        unit: UnitId,
    },
    /// [`ControlApi::unit_events`].
    UnitEvents {
        /// The unit to drain events for.
        unit: UnitId,
        /// Maximum events to drain.
        max: u32,
    },
    /// [`ControlApi::unit_outbound`].
    UnitOutbound {
        /// The unit to drain §17 outbound items for.
        unit: UnitId,
        /// Maximum items to drain.
        max: u32,
    },
    /// [`SessionApi::session_history`].
    SessionHistory {
        /// The session whose durable history to read.
        session: SessionId,
        /// The exclusive lower-bound cursor (0 from the start).
        after_cursor: u64,
        /// Maximum entries to return (0 = all available).
        max: u32,
    },
    /// [`SessionApi::log_after`] — the one-shot / long-poll cursor read of the merged live event log
    /// (the wire-marshaled form of `subscribe`; true push streaming stays a transport capability).
    Subscribe {
        /// The session whose merged live log to read.
        session: SessionId,
        /// The exclusive lower-bound `seq` (0 from the start).
        after_seq: u64,
        /// Maximum entries to return (0 = all available).
        max: u32,
    },
    /// [`ControlApi::events_page`] / [`ControlApi::events_subscribe`] — the node-wide event feed
    /// (L3). Served as a push stream over `Open` (streaming, [`is_streaming`]) or a one-shot/long-poll
    /// page over `Call`.
    EventsSince {
        /// The exclusive lower-bound feed cursor (0 from the start of the retained ring).
        cursor: u64,
        /// One-shot long-poll hold (ms); `None`/`0` returns immediately. Ignored by the push path.
        #[serde(default)]
        wait_ms: Option<u32>,
    },
    /// [`SessionApi::delivery_targets`].
    DeliveryTargets {
        /// The session whose delivery targets to read.
        session: SessionId,
    },
    /// [`SessionApi::delivery_sessions`] — the live sessions a transport instance owns for delivery.
    DeliverySessions {
        /// The transport instance whose owned sessions to enumerate.
        transport: TransportId,
    },
    /// [`SessionApi::handover`].
    Handover {
        /// The session whose `Primary` reply sink to re-point.
        session: SessionId,
        /// The new `Primary` target.
        target: DeliveryTarget,
    },
    /// [`SessionApi::record_meta`] — record an observability-only transport/meta event.
    RecordMeta {
        /// The session whose merged log to append to.
        session: SessionId,
        /// The attribution for the meta event.
        origin: Origin,
        /// The renderer/router discriminator (e.g. `"presence"` / `"attach"`).
        kind: String,
        /// The opaque encoded body, decoded by the consumer per `kind`.
        #[serde(with = "serde_bytes")]
        body: Vec<u8>,
    },
    /// [`ControlApi::unit_history`].
    UnitHistory {
        /// The unit whose durable history to read.
        unit: UnitId,
        /// The exclusive lower-bound cursor (0 from the start).
        after_cursor: u64,
        /// Maximum entries to return (0 = all available).
        max: u32,
    },
    /// [`ControlApi::pause`].
    Pause {
        /// The unit to pause.
        unit: UnitId,
    },
    /// [`ControlApi::resume`].
    Resume {
        /// The unit to resume.
        unit: UnitId,
    },
    /// [`ControlApi::scale`].
    Scale {
        /// The unit (sub-fleet) to scale.
        unit: UnitId,
        /// The target member count.
        n: u32,
    },
    /// [`ControlApi::verifying_key`].
    VerifyingKey,
    /// [`ModelApi::model_search`].
    ModelSearch {
        /// The search request.
        query: SearchQuery,
    },
    /// [`ModelApi::model_files`].
    ModelFiles {
        /// The `org/name` repo id.
        repo: String,
        /// The git revision to list (`None` = `main`).
        revision: Option<String>,
        /// The engine the listed files must be loadable by.
        engine: ModelEngine,
    },
    /// [`ModelApi::model_download`].
    ModelDownload {
        /// The model to acquire.
        model: ModelRef,
    },
    /// [`ModelApi::model_downloads`].
    ModelDownloads,
    /// [`ModelApi::model_cancel`].
    ModelCancel {
        /// The download job to cancel.
        id: DownloadId,
    },
    /// [`ModelApi::model_pause`].
    ModelPause {
        /// The download job to pause.
        id: DownloadId,
    },
    /// [`ModelApi::model_resume`].
    ModelResume {
        /// The download job to resume.
        id: DownloadId,
    },
    /// [`ModelApi::model_catalog`].
    ModelCatalog,
    /// [`ModelApi::model_delete`].
    ModelDelete {
        /// The installed model to delete.
        id: ModelId,
    },
    /// [`ModelApi::model_activate`].
    ModelActivate {
        /// The installed model to activate.
        id: ModelId,
        /// The profile to activate it for (`None` = the default local profile).
        profile: Option<String>,
    },
    /// [`ModelApi::model_recommend`].
    ModelRecommend {
        /// The `org/name` repo id.
        repo: String,
        /// The git revision (`None` = `main`).
        revision: Option<String>,
        /// The engine the recommendation targets.
        engine: ModelEngine,
        /// An explicit memory budget in bytes (`None` = auto-detect VRAM/RAM).
        budget_bytes: Option<u64>,
    },
    /// [`ModelApi::model_quantize`].
    ModelQuantize {
        /// The `org/name` repo id whose GGUF is quantized.
        repo: String,
        /// The git revision (`None` = `main`).
        revision: Option<String>,
        /// The target quant label (e.g. `Q4_K_M`).
        target_quant: String,
        /// The source GGUF file (`None` = the highest-precision one in the repo).
        source_file: Option<String>,
    },
    /// [`ModelApi::model_quantizes`].
    ModelQuantizes,
    /// [`ModelApi::model_inspect`].
    ModelInspect {
        /// The installed model to introspect.
        id: ModelId,
    },
    /// [`ProfileApi::profile_list`].
    ProfileList,
    /// [`ProfileApi::profile_get`].
    ProfileGet {
        /// The profile id to fetch.
        id: String,
    },
    /// [`ProfileApi::profile_create`].
    ProfileCreate {
        /// The new profile bundle.
        spec: ProfileSpec,
    },
    /// [`ProfileApi::profile_update`].
    ProfileUpdate {
        /// The replacement profile bundle (keyed by its id).
        spec: ProfileSpec,
    },
    /// [`ProfileApi::profile_delete`].
    ProfileDelete {
        /// The profile id to delete.
        id: String,
    },
    /// [`ProfileApi::profile_select`].
    ProfileSelect {
        /// The profile id to make the active default.
        id: String,
    },
    /// [`ProfileApi::profile_clone`].
    ProfileClone {
        /// The source profile to copy.
        source: String,
        /// The new profile id.
        new_id: String,
    },
    /// [`ProfileApi::profile_export`].
    ProfileExport {
        /// The profile id to export as a distribution.
        id: String,
    },
    /// [`ProfileApi::profile_import`].
    ProfileImport {
        /// The distribution to import.
        dist: Distribution,
        /// Optional id override (`None` = the distribution's own id).
        #[serde(default)]
        new_id: Option<String>,
    },
    /// [`ProfileApi::profile_history`].
    ProfileHistory {
        /// The profile id whose history to list.
        id: String,
    },
    /// [`ProfileApi::profile_at`].
    ProfileAt {
        /// The profile id.
        id: String,
        /// The revision sequence.
        seq: u64,
    },
    /// [`ProfileApi::profile_revert`].
    ProfileRevert {
        /// The profile id.
        id: String,
        /// The revision sequence to revert to.
        seq: u64,
    },
    /// [`ProfileApi::skill_history`].
    SkillHistory {
        /// The skill (bundle) name whose history to list.
        name: String,
    },
    /// [`ProfileApi::skill_at`].
    SkillAt {
        /// The skill (bundle) name.
        name: String,
        /// The revision sequence.
        seq: u64,
    },
    /// [`ProfileApi::skill_revert`].
    SkillRevert {
        /// The skill (bundle) name.
        name: String,
        /// The revision sequence to revert to.
        seq: u64,
    },
    /// [`ProfileApi::curator_list`].
    CuratorList {
        /// The profile whose skill library to list (`None` = the active default).
        #[serde(default)]
        profile: Option<String>,
    },
    /// [`ProfileApi::curator_pin`].
    CuratorPin {
        /// The profile owning the skill (`None` = the active default).
        #[serde(default)]
        profile: Option<String>,
        /// The skill (bundle) name to pin (protect from auto-archiving).
        name: String,
    },
    /// [`ProfileApi::curator_unpin`].
    CuratorUnpin {
        /// The profile owning the skill (`None` = the active default).
        #[serde(default)]
        profile: Option<String>,
        /// The skill (bundle) name to unpin.
        name: String,
    },
    /// [`ProfileApi::curator_archive`].
    CuratorArchive {
        /// The profile owning the skill (`None` = the active default).
        #[serde(default)]
        profile: Option<String>,
        /// The skill (bundle) name to archive (move out of discovery).
        name: String,
    },
    /// [`ProfileApi::curator_restore`].
    CuratorRestore {
        /// The profile owning the skill (`None` = the active default).
        #[serde(default)]
        profile: Option<String>,
        /// The skill (bundle) name to restore from the archive.
        name: String,
    },
    /// [`ProfileApi::curator_run`].
    CuratorRun {
        /// The profile whose library to curate (`None` = the active default).
        #[serde(default)]
        profile: Option<String>,
    },
    /// [`CredentialApi::credential_set`].
    CredentialSet {
        /// The profile / credential-ref to key the secret by.
        profile: String,
        /// The secret value (provider API key / token).
        secret: String,
    },
    /// [`CredentialApi::credential_list`].
    CredentialList,
    /// [`CredentialApi::credential_remove`].
    CredentialRemove {
        /// The profile / credential-ref to clear.
        profile: String,
    },
    /// [`AuthApi::auth_begin`].
    AuthBegin(AuthBeginRequest),
    /// [`AuthApi::auth_complete`].
    AuthComplete(AuthCompleteRequest),
    /// [`AuthApi::auth_cancel`].
    AuthCancel {
        /// The flow id to drop.
        flow_id: String,
    },
    /// [`AuthApi::auth_providers`].
    AuthProviders,
    /// [`ModelApi::models`].
    Models,
    /// [`ModelApi::model_current`].
    ModelCurrent {
        /// The profile to resolve (`None` = the active default).
        profile: Option<String>,
    },
    /// [`SessionApi::set_session_model`].
    SetSessionModel {
        /// The live session to switch.
        session: SessionId,
        /// The new model id.
        model: String,
        /// Optionally re-bind the provider (`None` = keep the session's current provider).
        #[serde(default)]
        provider: Option<ProviderSelector>,
    },
    /// [`SessionApi::set_session_mode`].
    SetSessionMode {
        /// The live session whose edit-approval mode to switch.
        session: SessionId,
        /// The new edit-approval session mode.
        mode: ApprovalMode,
    },
    /// [`SessionApi::set_session_overlay`].
    SetSessionOverlay {
        /// The session whose per-session overlay to replace.
        session: SessionId,
        /// The new overlay (model / provider / tool allowlist / approval mode).
        overlay: SessionOverlay,
    },
    /// [`ControlApi::approvals_pending`].
    ApprovalsPending {
        /// Filter to one session, or `None` for the node-wide HITL inbox.
        #[serde(default)]
        session: Option<SessionId>,
    },
    /// [`ControlApi::approval_decide`].
    ApprovalDecide {
        /// The session that parked the request.
        session: SessionId,
        /// The opaque parked-request id (from [`ApprovalInfo`]).
        request_id: String,
        /// The operator's decision (allow / deny).
        allow: bool,
    },
    /// [`ControlApi::checkpoints`].
    CheckpointList {
        /// Filter to one session, or `None` for the node-wide checkpoint list.
        #[serde(default)]
        session: Option<SessionId>,
    },
    /// [`ControlApi::checkpoint_rewind`].
    CheckpointRewind {
        /// The session the checkpoint belongs to.
        session: SessionId,
        /// The opaque checkpoint id (from [`CheckpointInfo`]).
        checkpoint_id: String,
    },
    /// [`ControlApi::sessions_query`] — the scoped, paginated roster.
    SessionsQuery {
        /// The roster query (scope + cursor + limit).
        query: SessionQuery,
    },
    /// [`ControlApi::session_get`] — one session's full detail.
    SessionGet {
        /// The session to detail.
        session: SessionId,
    },
    /// [`ControlApi::sessions_by_profile`] — the roster grouped by owning profile.
    SessionsByProfile,
    /// [`ControlApi::session_search`] — full-text session search.
    SessionSearch {
        /// The search query.
        query: String,
        /// Max hits (`0` = a server default).
        limit: u32,
    },
    /// [`ControlApi::session_update_meta`] — rename/pin/archive a session (roster session actions).
    SessionUpdateMeta {
        /// The session to update.
        session: SessionId,
        /// The partial metadata patch.
        patch: SessionMetaPatch,
    },
    /// [`ControlApi::rewind`] — unified conversation + workspace rewind.
    Rewind {
        /// The session to rewind.
        session: SessionId,
        /// Where to rewind to.
        point: RewindPoint,
    },
    /// [`ControlApi::acp_discover`] — trigger an ACP discovery scan.
    AcpDiscover,
    /// [`ControlApi::acp_catalog`] — the persisted ACP agent catalog.
    AcpCatalog,
    /// [`ControlApi::acp_register`] — register an ACP launch recipe.
    AcpRegister {
        /// The recipe to persist.
        entry: AcpAgentEntry,
    },
    /// [`ControlApi::acp_remove`] — remove a cataloged/registered ACP agent.
    AcpRemove {
        /// The agent name to remove.
        name: String,
    },
    /// [`ProfileApi::skill_get`] — read a skill bundle at head.
    SkillGet {
        /// The skill (bundle) name.
        name: String,
    },
    /// [`ProfileApi::skill_put`] — create/replace a skill bundle body.
    SkillPut {
        /// The bundle to write.
        bundle: SkillBundle,
    },
    /// [`ControlApi::provider_list`].
    ProviderList,
    /// [`ControlApi::provider_register`].
    ProviderRegister {
        /// The provider to register.
        provider: ProviderInfo,
    },
    /// [`ControlApi::tool_list`].
    ToolList,
    /// [`ControlApi::tool_register`].
    ToolRegister {
        /// The tool to register.
        tool: ToolInfo,
    },
    /// [`ControlApi::command_list`] — the daemon-authoritative command catalog.
    CommandList,
    /// [`ControlApi::command_invoke`] — run a command by name.
    CommandInvoke {
        /// The command + args + session/origin context.
        invocation: CommandInvocation,
    },
    /// [`ControlApi::config_get`].
    ConfigGet,
    /// [`ControlApi::config_set`].
    ConfigSet {
        /// The replacement config.
        config: NodeConfigView,
    },
    /// [`ControlApi::cron_list`].
    CronList,
    /// [`ControlApi::cron_create`].
    CronCreate {
        /// The job spec.
        spec: CronSpec,
    },
    /// [`ControlApi::cron_update`].
    CronUpdate {
        /// The job id.
        id: String,
        /// The replacement spec.
        spec: CronSpec,
    },
    /// [`ControlApi::cron_delete`].
    CronDelete {
        /// The job id.
        id: String,
    },
    /// [`ControlApi::cron_trigger`].
    CronTrigger {
        /// The job id.
        id: String,
    },
    /// [`ControlApi::cron_runs`].
    CronRuns {
        /// The job id.
        id: String,
    },
    /// [`ControlApi::cron_pause`].
    CronPause {
        /// The job id.
        id: String,
        /// `true` to pause, `false` to resume.
        paused: bool,
    },
    /// [`ControlApi::cron_suggestions`].
    CronSuggestions,
    /// [`ControlApi::cron_accept_suggestion`].
    CronAcceptSuggestion {
        /// The suggestion id.
        id: String,
    },
    /// [`ControlApi::cron_dismiss_suggestion`].
    CronDismissSuggestion {
        /// The suggestion id.
        id: String,
    },
    /// [`ControlApi::routing_list_chats`] — all chat→session routing pins.
    RoutingListChats,
    /// [`ControlApi::routing_get`] — the pin for an origin.
    RoutingGet {
        /// The origin to look up.
        origin: Origin,
    },
    /// [`ControlApi::routing_set`] — upsert a full routing pin.
    RoutingSet {
        /// The pin to persist.
        route: ChatRoute,
    },
    /// [`ControlApi::routing_bind_chat`] — pin an origin to a session (+ optional profile).
    RoutingBindChat {
        /// The origin to pin.
        origin: Origin,
        /// The session to pin it to.
        session: SessionId,
        /// An optional profile override.
        #[serde(default)]
        profile: Option<ProfileRef>,
    },
    /// [`ControlApi::routing_unbind_chat`] — remove an origin's pin.
    RoutingUnbindChat {
        /// The origin to unpin.
        origin: Origin,
    },
    /// [`ControlApi::transport_rooms`] — enumerate a transport instance's rooms.
    TransportRooms {
        /// The transport instance.
        transport: TransportId,
    },
    /// [`ControlApi::transport_adapters`] — the available adapter families + capabilities + schema.
    TransportAdapters,
    /// [`ControlApi::transport_instances`] — the configured instances + live connection/presence.
    TransportInstances,
    /// [`ControlApi::conv_list`] — a transport's conversations.
    ConvList {
        /// The owning transport.
        transport: TransportId,
    },
    /// [`ControlApi::conv_get`] — one conversation by id.
    ConvGet {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
    },
    /// [`ControlApi::conv_create_details`] — the typed create form.
    ConvCreateDetails {
        /// The owning transport.
        transport: TransportId,
    },
    /// [`ControlApi::conv_create`] — create a conversation.
    ConvCreate {
        /// The owning transport.
        transport: TransportId,
        /// The filled create details.
        details: CreateConversationDetails,
    },
    /// [`ControlApi::conv_join_details`] — the typed join form.
    ConvJoinDetails {
        /// The owning transport.
        transport: TransportId,
    },
    /// [`ControlApi::conv_join`] — join a channel.
    ConvJoin {
        /// The owning transport.
        transport: TransportId,
        /// The filled join details.
        details: ChannelJoinDetails,
    },
    /// [`ControlApi::conv_leave`] — leave a conversation.
    ConvLeave {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
    },
    /// [`ControlApi::conv_send`] — send into a conversation.
    ConvSend {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
        /// The author (`None` = the account/operator).
        #[serde(default)]
        from: Option<Participant>,
        /// The message.
        message: UserMsg,
    },
    /// [`ControlApi::conv_set_topic`].
    ConvSetTopic {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
        /// The new topic (`None` clears).
        #[serde(default)]
        topic: Option<String>,
    },
    /// [`ControlApi::conv_set_title`].
    ConvSetTitle {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
        /// The new title.
        #[serde(default)]
        title: Option<String>,
    },
    /// [`ControlApi::conv_set_description`].
    ConvSetDescription {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
        /// The new description.
        #[serde(default)]
        description: Option<String>,
    },
    /// [`ControlApi::conv_delete`] — delete/destroy a conversation.
    ConvDelete {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
    },
    /// [`ControlApi::conv_history`] — the conversation's durable verifiable transcript.
    ConvHistory {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
        /// Return entries with cursor strictly greater than this (`0` from the start).
        #[serde(default)]
        after_cursor: u64,
        /// Max entries (`0` = all).
        #[serde(default)]
        max: u32,
    },
    /// [`ControlApi::member_invite`] — invite/add a participant.
    MemberInvite {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
        /// Who to invite.
        who: Participant,
        /// An optional invite message.
        #[serde(default)]
        message: Option<String>,
    },
    /// [`ControlApi::member_remove`] — remove/kick a participant.
    MemberRemove {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
        /// Who to remove.
        who: Participant,
        /// An optional reason.
        #[serde(default)]
        reason: Option<String>,
    },
    /// [`ControlApi::member_ban`] — ban a participant.
    MemberBan {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
        /// Who to ban.
        who: Participant,
        /// An optional reason.
        #[serde(default)]
        reason: Option<String>,
    },
    /// [`ControlApi::member_set_role`] — set a participant's role.
    MemberSetRole {
        /// The owning transport.
        transport: TransportId,
        /// The conversation id.
        conv: String,
        /// Whose role to set.
        who: Participant,
        /// The new role.
        role: MemberRole,
    },
    /// [`ControlApi::contact_get_profile`] — fetch a remote contact's profile.
    ContactGetProfile {
        /// The owning transport.
        transport: TransportId,
        /// The contact whose profile to fetch.
        contact: ContactInfo,
    },
    /// [`ControlApi::contact_set_alias`] — set a local alias for a contact.
    ContactSetAlias {
        /// The owning transport.
        transport: TransportId,
        /// The contact to alias.
        contact: ContactInfo,
        /// The new alias (`None` clears).
        #[serde(default)]
        alias: Option<String>,
    },
    /// [`ControlApi::contact_action_menu`] — the contact's action menu.
    ContactActionMenu {
        /// The owning transport.
        transport: TransportId,
        /// The contact.
        contact: ContactInfo,
    },
    /// [`ControlApi::directory_search`] — search the transport's contact/user directory.
    DirectorySearch {
        /// The owning transport.
        transport: TransportId,
        /// The search query (`None`/empty = an unfiltered listing where the transport allows it).
        #[serde(default)]
        query: Option<String>,
    },
    /// [`ControlApi::fs_roots`].
    FsRoots,
    /// [`ControlApi::fs_list`].
    FsList {
        /// The root to list within.
        root: FsRootId,
        /// Root-relative directory ("" = the root).
        dir: String,
        /// Include ignored entries (they are marked either way).
        #[serde(default)]
        show_ignored: bool,
    },
    /// [`ControlApi::fs_stat`].
    FsStat {
        /// The root.
        root: FsRootId,
        /// Root-relative path.
        path: String,
    },
    /// [`ControlApi::fs_read`].
    FsRead {
        /// The root.
        root: FsRootId,
        /// Root-relative path.
        path: String,
        /// Max bytes (`0` = a server default).
        #[serde(default)]
        max_bytes: u64,
    },
    /// [`ControlApi::fs_write`].
    FsWrite {
        /// The root (Workspace/Session only).
        root: FsRootId,
        /// Root-relative path.
        path: String,
        /// The bytes to write.
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
        /// The base etag for optimistic concurrency (`None` = create-or-overwrite).
        #[serde(default)]
        base_revision: Option<FsRevision>,
        /// Override the sensitive-path / `Deny` gate.
        #[serde(default)]
        force: bool,
    },
    /// [`ControlApi::fs_search`].
    FsSearch {
        /// The root to search within.
        root: FsRootId,
        /// The search query.
        query: FsSearchQuery,
    },
    /// [`ControlApi::fs_watch_after`] — the cursor / long-poll form of the change stream.
    FsWatchPoll {
        /// The root.
        root: FsRootId,
        /// Root-relative directory being watched.
        dir: String,
        /// Drain changes after this cursor.
        after_seq: u64,
        /// Max events to drain.
        max: u32,
    },
    /// [`ControlApi::blob_put`].
    BlobPut {
        /// The bytes to store.
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
    },
    /// [`ControlApi::blob_get`].
    BlobGet {
        /// The content hash to fetch.
        hash: ContentHash,
        /// An optional byte range (a ranged read is returned unverified).
        #[serde(default)]
        range: Option<ByteRange>,
    },
    /// [`ControlApi::blob_stat`].
    BlobStat {
        /// The content hash to stat.
        hash: ContentHash,
    },
    /// [`ControlApi::fs_write_from_blob`].
    FsWriteFromBlob {
        /// The target root (Workspace/Session only).
        root: FsRootId,
        /// Root-relative destination path.
        path: String,
        /// The blob to materialize.
        hash: ContentHash,
        /// The base etag for optimistic concurrency (`None` = create-or-overwrite).
        #[serde(default)]
        base_revision: Option<FsRevision>,
        /// Override the sensitive-path / `Deny` gate.
        #[serde(default)]
        force: bool,
    },
}

/// The serializable reflection of an interface result.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiResponse {
    /// A successful unit reply (submit/respond/assign/cancel).
    Ok,
    /// The session a routed submit ([`ApiRequest::SubmitRouted`]) resolved to and opened.
    Routed {
        /// The derived session id (subscribe/poll it for the reply).
        session: SessionId,
    },
    /// Drained outbound items (poll).
    Drained(Vec<Outbound>),
    /// A health report.
    Health(HealthReport),
    /// A stats report.
    Stats(StatsReport),
    /// A telemetry dump (folded usage/cost + events + health + queue depths).
    Telemetry(TelemetryDump),
    /// A session list.
    Sessions(Vec<SessionInfo>),
    /// A list of parked §12 edit-approval requests awaiting an operator decision.
    Approvals(Vec<ApprovalInfo>),
    /// A list of recorded §12 tool checkpoints (rewind points), newest first.
    Checkpoints(Vec<CheckpointInfo>),
    /// A fleet report.
    Fleet(FleetReport),
    /// A tree report.
    Tree(TreeReport),
    /// One unit's node view (`None` rendered as the absent variant).
    Unit(Option<UnitNode>),
    /// Drained per-unit management events.
    UnitEvents(Vec<ManageEventView>),
    /// A page of decoded + verified journal history (session/unit history).
    Journal(JournalPageView),
    /// A page of the merged live session event log (the cursor read of `subscribe`).
    LogPage(LogPageView),
    /// A page of the node-wide event feed (the cursor read of `events_since`; L3).
    EventsPage(EventsPage),
    /// A session's outbound delivery targets (the reply sinks of `delivery_targets`).
    DeliveryTargets(Vec<DeliveryTarget>),
    /// The live sessions a transport instance owns for delivery (`delivery_sessions`).
    DeliverySessions(Vec<SessionId>),
    /// The node's journal verifying key (hex dCBOR), or `None` if it exposes no signer.
    VerifyingKey(Option<String>),
    /// A page of model search results.
    ModelSearch(SearchPage),
    /// A repo's loadable files.
    ModelFiles(Vec<ModelFile>),
    /// A started download's job handle.
    ModelDownloadStarted(DownloadId),
    /// Download job statuses.
    ModelDownloads(Vec<DownloadStatus>),
    /// The installed-model catalog.
    ModelCatalog(Vec<InstalledModel>),
    /// A quantization recommendation.
    ModelRecommend(QuantRecommendation),
    /// A started quantization's job handle.
    ModelQuantizeStarted(QuantizeId),
    /// Quantization job statuses.
    ModelQuantizes(Vec<QuantizeStatus>),
    /// A model's GGUF metadata.
    ModelInspect(GgufInfo),
    /// A profile listing (the active default marked).
    Profiles(Vec<ProfileInfo>),
    /// One profile's full spec, or `None` if unknown / no active default (profile_get).
    Profile(Option<ProfileSpec>),
    /// A redacted credential listing.
    Credentials(Vec<CredentialInfo>),
    /// A begun interactive-auth flow handle (`auth_begin`).
    AuthBegun(AuthBeginResponse),
    /// A completed interactive-auth flow outcome (`auth_complete`).
    AuthCompleted(AuthCompleteResponse),
    /// The registered interactive-auth providers (`auth_providers`).
    AuthProviders(Vec<AuthProviderInfo>),
    /// A discoverable model catalog (cloud + local).
    Models(Vec<ModelDescriptor>),
    /// The model a profile currently resolves to (`None` = none resolvable).
    ModelCurrent(Option<ModelDescriptor>),
    /// A profile distribution (profile_export).
    Distribution(Distribution),
    /// A created profile id (profile_import).
    ProfileId(String),
    /// A revision history (profile_history / skill_history), oldest first.
    Revisions(Vec<Revision>),
    /// A skill bundle as recorded at a revision (skill_at).
    SkillBundle(SkillBundle),
    /// A profile's curator listing (curator_list): discovered + archived skills with usage.
    CuratorSkills(Vec<CuratorEntry>),
    /// The lifecycle changes a curator run applied (curator_run).
    CuratorRun(Vec<CuratorChange>),
    /// A page of the scoped roster (sessions_query).
    SessionPage(SessionPage),
    /// One session's full detail, or `None` if unknown (session_get).
    SessionDetail(Option<SessionDetail>),
    /// The roster grouped by owning profile (sessions_by_profile).
    SessionsByProfile(Vec<(ProfileRef, Vec<SessionInfo>)>),
    /// Full-text session-search hits (session_search).
    SessionSearch(Vec<SessionSearchHit>),
    /// The ACP agent catalog (acp_discover / acp_catalog).
    AcpCatalog(Vec<AcpAgentEntry>),
    /// The runtime provider registry (provider_list).
    Providers(Vec<ProviderInfo>),
    /// The node tool list (tool_list).
    Tools(Vec<ToolInfo>),
    /// The daemon-authoritative command catalog (command_list).
    Commands(Vec<CommandSpec>),
    /// A command invocation's rendered result (command_invoke).
    CommandOutput(CommandOutput),
    /// The node runtime config (config_get).
    Config(NodeConfigView),
    /// The scheduled cron jobs (cron_list).
    CronJobs(Vec<CronJob>),
    /// A created cron job id (cron_create).
    CronId(String),
    /// Recent runs of a scheduled job (cron_runs).
    CronRuns(Vec<CronRun>),
    /// Pending cron-job suggestions (cron_suggestions).
    CronSuggestions(Vec<CronSuggestion>),
    /// The chat→session routing pins (routing_list_chats).
    ChatRoutes(Vec<ChatRoute>),
    /// One origin's routing pin, if set (routing_get).
    ChatRoute(Option<ChatRoute>),
    /// A transport instance's rooms (transport_rooms).
    Rooms(Vec<RoomInfo>),
    /// A transport's conversations (conv_list).
    Conversations(Vec<ConversationInfo>),
    /// One conversation, if present (conv_get / conv_create / conv_join).
    Conversation(Option<ConversationInfo>),
    /// A remote contact's profile text (contact_get_profile).
    ContactProfile(String),
    /// A list of contacts (directory_search).
    Contacts(Vec<ContactInfo>),
    /// A contact's action menu, if any (contact_action_menu).
    ActionMenu(Option<ActionMenu>),
    /// The typed create-conversation form (conv_create_details).
    ConvCreateDetails(CreateConversationDetails),
    /// The typed channel-join form (conv_join_details).
    ConvJoinDetails(ChannelJoinDetails),
    /// The available transport adapters (transport_adapters).
    Adapters(Vec<AdapterInfo>),
    /// The configured transport instances + live status (transport_instances).
    TransportInstances(Vec<TransportInstanceInfo>),
    /// A failure (the interface's `ApiError`, round-tripped faithfully).
    Error(ApiError),
    /// The browsable filesystem roots (fs_roots).
    FsRoots(Vec<FsRoot>),
    /// A directory listing (fs_list).
    FsList(Vec<FsEntry>),
    /// One entry's metadata (fs_stat).
    FsStat(FsEntry),
    /// A file's bytes + etag (fs_read).
    FsRead(FsContent),
    /// A write's new etag (fs_write).
    FsWrite(FsRevision),
    /// A page of project-search hits (fs_search).
    FsSearch(FsSearchPage),
    /// A page of watch change events (fs_watch_after).
    FsWatch(FsWatchPageView),
    /// A stored blob's ref (blob_put).
    BlobPut(BlobRef),
    /// A blob's bytes (blob_get).
    BlobGet(#[serde(with = "serde_bytes")] Vec<u8>),
    /// A blob's metadata (blob_stat).
    BlobStat(BlobStat),
}

// ---------------------------------------------------------------------------
// Multiplexed / server-streaming socket envelope (wire L0; daemon-sync-protocol-spec.md §2)
// ---------------------------------------------------------------------------

/// The wire protocol version a `Hello` negotiates. Bumped when the envelope shape changes.
pub const WIRE_VERSION: u32 = 1;
/// Feature flag: the connection speaks the multiplexed `Call`/`Reply` envelope.
pub const WIRE_FEATURE_MUX: &str = "mux";
/// Feature flag: the server can push `Item`/`End` frames for streaming requests.
pub const WIRE_FEATURE_STREAM: &str = "stream";
/// Feature flag: the node hosts profile/skill versioning (a bound revision log), so the
/// `Profile{History,At,Revert}` (+ skill) ops are available rather than `Unsupported`.
pub const WIRE_FEATURE_VERSIONING: &str = "versioning";

/// A client -> server multiplexed frame. Wraps an [`ApiRequest`] so one connection can carry many
/// correlated exchanges. Absent on the legacy path: a connection whose first frame decodes as a
/// bare [`ApiRequest`] (no `Hello`) is served one-shot exactly as before, preserving the FFI/CLI.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireC2S {
    /// Opt into the multiplexed/streaming envelope; the server answers with [`WireS2C::Hello`].
    Hello {
        /// The highest [`WIRE_VERSION`] the client speaks.
        wire_version: u32,
        /// Requested capabilities (e.g. [`WIRE_FEATURE_MUX`], [`WIRE_FEATURE_STREAM`]).
        features: Vec<String>,
    },
    /// A one-shot request, answered by exactly one [`WireS2C::Reply`]. `Subscribe` over `Call` is the
    /// non-destructive cursor read (`log_after`), so a polling client keeps working under mux.
    Call {
        /// Client-chosen, per-connection, monotonically increasing correlation id.
        id: u64,
        /// The wrapped request.
        req: ApiRequest,
    },
    /// Open a server-stream for a streaming-capable request ([`is_streaming`]), answered by zero or
    /// more [`WireS2C::Item`]s then [`WireS2C::End`]. The client (not the request variant alone)
    /// chooses streaming, so the same `Subscribe` can be polled (`Call`) or streamed (`Open`).
    Open {
        /// Client-chosen correlation id for the stream.
        id: u64,
        /// The wrapped streaming request.
        req: ApiRequest,
    },
    /// Tear an `Open` stream down early (distinct from [`ApiRequest::Cancel`], which cancels a
    /// turn). No-op for an already-closed `id`.
    Cancel {
        /// The exchange to abort.
        id: u64,
    },
}

/// A server -> client multiplexed frame.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireS2C {
    /// Handshake ack: the capabilities the server actually supports (the usable set is the
    /// intersection with the client's requested `features`).
    Hello {
        /// The server's [`WIRE_VERSION`].
        wire_version: u32,
        /// Supported capabilities.
        features: Vec<String>,
    },
    /// The single result of a one-shot `Call` (closes `id`).
    Reply {
        /// The `Call` id this answers.
        id: u64,
        /// The wrapped response.
        res: ApiResponse,
    },
    /// One chunk of a streaming `Call`; `id` stays open until `End`.
    Item {
        /// The `Call` id this belongs to.
        id: u64,
        /// The wrapped response chunk.
        res: ApiResponse,
    },
    /// A stream closed (clean iff `error` is `None`).
    End {
        /// The `Call` id that closed.
        id: u64,
        /// `Some` if the stream ended in error (e.g. the live broadcast lagged).
        error: Option<ApiError>,
    },
    /// The stream's cursor is no longer trustworthy (lag / re-activation); the client must
    /// re-baseline. Carried here from L0 on; the epoch/head_seq semantics are finalized in L2.
    Reset {
        /// The affected `Call` id.
        id: u64,
        /// The current session-activation epoch.
        epoch: u64,
        /// The current high-water `seq`.
        head_seq: u64,
    },
}

/// Whether a request is served as a server-stream (`Item`* then `End`) rather than a single `Reply`.
/// L0 streams only the live log subscription; later layers add the node-wide events feed.
pub fn is_streaming(req: &ApiRequest) -> bool {
    matches!(
        req,
        ApiRequest::Subscribe { .. } | ApiRequest::EventsSince { .. }
    )
}

// ---------------------------------------------------------------------------
// Filesystem / workspace surface DTOs (daemon-fs-surface-spec.md)
// ---------------------------------------------------------------------------

/// Which root a filesystem op addresses.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsRootId {
    /// Browse the node's own machine for discovery, bounded by the node browse policy (home +
    /// operator allowlist). Read-only. The `String` names which advertised browse root.
    Host(String),
    /// The node's configured workspace root.
    Workspace,
    /// A session/unit's workspace sandbox (its execution-environment root).
    Session(SessionId),
}

/// The kind of an advertised root.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsRootKind {
    /// A host browse root (read-only discovery).
    Host,
    /// The node workspace root.
    Workspace,
    /// A session sandbox root.
    Session,
}

/// A browsable root the node advertises (`fs_roots`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsRoot {
    /// The root id to pass to the other fs ops.
    pub id: FsRootId,
    /// A human label (basename / home / session title).
    pub label: String,
    /// What kind of root this is.
    pub kind: FsRootKind,
    /// The owning session, when `kind == Session`.
    #[serde(default)]
    pub session: Option<SessionId>,
}

/// What kind of directory entry a listing row is.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsEntryKind {
    /// A regular file.
    File,
    /// A directory.
    Dir,
    /// A symbolic link.
    Symlink,
}

/// One directory child (fs_list / fs_stat). `path` is root-relative with POSIX separators.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsEntry {
    /// The entry's base name.
    pub name: String,
    /// Root-relative path (POSIX separators).
    pub path: String,
    /// File / dir / symlink.
    pub kind: FsEntryKind,
    /// Size in bytes (0 for directories).
    pub size: u64,
    /// Last-modified wall-clock milliseconds since the Unix epoch (0 if unknown).
    pub mtime_ms: u64,
    /// Whether the node's ignore rules matched this entry (marked, not hidden — the client decides
    /// whether to show it). Shipped: a built-in artifact/VCS name set (`.git`, `node_modules`,
    /// `target`, ...); full `.gitignore` evaluation is future.
    #[serde(default)]
    pub ignored: bool,
}

/// A cheap opaque content etag for optimistic-concurrency writes. NOT [`Revision`] (which is
/// profile/skill versioning); this avoids re-reading a file to validate a write base.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsRevision {
    /// Last-modified wall-clock milliseconds at read time.
    pub mtime_ms: u64,
    /// Size in bytes at read time.
    pub size: u64,
}

/// A file's bytes + etag (fs_read).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsContent {
    /// The (possibly truncated) file bytes.
    #[serde(with = "serde_bytes")]
    pub bytes: Vec<u8>,
    /// The content etag (pass as `base_revision` to fs_write).
    pub revision: FsRevision,
    /// Whether the bytes were truncated at `max_bytes`.
    #[serde(default)]
    pub truncated: bool,
    /// A content-addressed ref for the served bytes, when the node has a content store and the read
    /// was **not** truncated (so the ref identifies the whole file). Lets a client hand the same
    /// content to an agent without re-uploading. `None` when truncated or no blob store is bound.
    #[serde(default)]
    pub blob_ref: Option<BlobRef>,
}

/// Metadata for a blob in the node content store (`blob_stat`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobStat {
    /// The blob's byte length (0 when absent).
    pub size: u64,
    /// Whether the blob is present in the store.
    pub present: bool,
}

/// A server-side project-search query (fs_search).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsSearchQuery {
    /// The search text (or regex when `regex`).
    pub query: String,
    /// Treat `query` as a regular expression.
    #[serde(default)]
    pub regex: bool,
    /// Case-sensitive match (default: insensitive).
    #[serde(default)]
    pub case_sensitive: bool,
    /// Max hits to return (`0` = a server default).
    #[serde(default)]
    pub max_results: u32,
    /// Zero-based page index for pagination.
    #[serde(default)]
    pub page: u32,
}

/// One project-search hit.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsSearchHit {
    /// Root-relative path of the matching file.
    pub path: String,
    /// 1-based line number of the match.
    pub line: u32,
    /// 1-based column of the match.
    pub col: u32,
    /// The matching line (trimmed) for preview.
    pub preview: String,
}

/// A page of project-search hits (fs_search).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct FsSearchPage {
    /// The hits in this page.
    pub hits: Vec<FsSearchHit>,
    /// Whether more hits exist beyond this page.
    #[serde(default)]
    pub has_more: bool,
}

/// What changed under a watched directory.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsChangeKind {
    /// A path appeared.
    Created,
    /// A path's contents changed.
    Modified,
    /// A path was removed.
    Removed,
}

/// One change event under a watched directory (fs_watch).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsChange {
    /// Root-relative path that changed.
    pub path: String,
    /// The kind of change.
    pub kind: FsChangeKind,
}

/// A page of change events drained by the watch cursor (fs_watch_after), modeled on the session
/// log's cursor read: `next_seq` is the cursor to pass on the next poll.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct FsWatchPageView {
    /// The change events since the requested cursor.
    pub events: Vec<FsChange>,
    /// The cursor to pass as `after_seq` on the next poll.
    pub next_seq: u64,
    /// The highest change `seq` the watch ring currently holds (how far a reader can advance now).
    /// Lets the client detect it is behind the live edge. `#[serde(default)]` keeps old (head-less)
    /// encodings decodable. (Cursored-stream contract; daemon-event-io-spec §5.4.1.)
    #[serde(default)]
    pub head_seq: u64,
    /// `true` when the reader's `after_seq` aged out of the ring (events were evicted past it), so
    /// this page is NOT a complete delta — the client must re-list the watched dir to reconcile
    /// (the fs analogue of the merged log's `Lagged -> Reset`). `#[serde(default)]` = `false`.
    #[serde(default)]
    pub reset: bool,
}

/// A live push stream of filesystem changes (a transport capability, like [`LogStream`]; the
/// one-shot/long-poll cursor form every transport marshals is `fs_watch_after`).
pub type FsWatchStream = BoxStream<'static, FsChange>;

/// Why an api call failed (serializable so it round-trips over any transport).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum ApiError {
    /// No such live/durable session.
    #[error("unknown session: {0}")]
    UnknownSession(String),
    /// The operation is not available (e.g. a control op over a session-only FFI transport, or an
    /// unsupported §17 command).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The session is already owned by the *other* lifecycle: a `SessionId` is durable-managed
    /// (control surface, `assign`) **or** live-interactive (session surface, `submit`), never both.
    #[error("conflict: {0}")]
    Conflict(String),
    /// Any other failure.
    #[error("{0}")]
    Other(String),
}

// ---------------------------------------------------------------------------
// Dispatch — the shared core every non-in-process transport calls
// ---------------------------------------------------------------------------

fn unit_or_err(r: Result<(), ApiError>) -> ApiResponse {
    match r {
        Ok(()) => ApiResponse::Ok,
        Err(e) => ApiResponse::Error(e),
    }
}

async fn serve_session(api: &dyn SessionApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::Submit {
            session,
            command,
            origin,
            profile,
        } => {
            if profile.is_some() {
                unit_or_err(api.submit_as(session, origin, command, profile).await)
            } else {
                match origin {
                    Some(origin) => unit_or_err(api.submit_from(session, origin, command).await),
                    None => unit_or_err(api.submit(session, command).await),
                }
            }
        }
        ApiRequest::SubmitRouted { origin, command } => {
            match api.submit_routed(origin, command).await {
                Ok(session) => ApiResponse::Routed { session },
                Err(e) => ApiResponse::Error(e),
            }
        }
        ApiRequest::Poll { session, max } => match api.poll(session, max).await {
            Ok(items) => ApiResponse::Drained(items),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::Respond { session, response } => {
            unit_or_err(api.respond(session, response).await)
        }
        ApiRequest::SessionHistory {
            session,
            after_cursor,
            max,
        } => ApiResponse::Journal(api.session_history(session, after_cursor, max).await),
        ApiRequest::Subscribe {
            session,
            after_seq,
            max,
        } => match api.log_after(session, after_seq, max).await {
            Ok(page) => ApiResponse::LogPage(page),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::DeliveryTargets { session } => {
            ApiResponse::DeliveryTargets(api.delivery_targets(session).await)
        }
        ApiRequest::DeliverySessions { transport } => {
            ApiResponse::DeliverySessions(api.delivery_sessions(transport).await)
        }
        ApiRequest::Handover { session, target } => {
            unit_or_err(api.handover(session, target).await)
        }
        ApiRequest::RecordMeta {
            session,
            origin,
            kind,
            body,
        } => unit_or_err(api.record_meta(session, origin, kind, body).await),
        ApiRequest::SetSessionModel {
            session,
            model,
            provider,
        } => unit_or_err(api.set_session_model(session, model, provider).await),
        ApiRequest::SetSessionMode { session, mode } => {
            unit_or_err(api.set_session_mode(session, mode).await)
        }
        ApiRequest::SetSessionOverlay { session, overlay } => {
            unit_or_err(api.set_session_overlay(session, overlay).await)
        }
        _ => return None,
    })
}

/// Dispatch a request against a full [`NodeApi`] — the entry point the socket/TCP/JSON-RPC node
/// transports call.
pub async fn dispatch(api: &dyn NodeApi, req: ApiRequest) -> ApiResponse {
    if let Some(resp) = serve_session(api, req.clone()).await {
        return resp;
    }
    match req {
        ApiRequest::Health => ApiResponse::Health(api.health().await),
        ApiRequest::Stats => ApiResponse::Stats(api.stats().await),
        ApiRequest::Telemetry => ApiResponse::Telemetry(api.telemetry().await),
        ApiRequest::Sessions => ApiResponse::Sessions(api.sessions().await),
        ApiRequest::ApprovalsPending { session } => {
            ApiResponse::Approvals(api.approvals_pending(session).await)
        }
        ApiRequest::ApprovalDecide {
            session,
            request_id,
            allow,
        } => unit_or_err(api.approval_decide(session, request_id, allow).await),
        ApiRequest::CheckpointList { session } => {
            ApiResponse::Checkpoints(api.checkpoints(session).await)
        }
        // L3 node-wide event feed: the one-shot/long-poll page (the push form rides `Open` ->
        // `events_subscribe` in the socket pump, not `dispatch`).
        ApiRequest::EventsSince { cursor, .. } => {
            ApiResponse::EventsPage(api.events_page(cursor, 0).await)
        }
        ApiRequest::CheckpointRewind {
            session,
            checkpoint_id,
        } => unit_or_err(api.checkpoint_rewind(session, checkpoint_id).await),
        ApiRequest::Assign { session } => unit_or_err(api.assign(session).await),
        ApiRequest::Cancel { session } => unit_or_err(api.cancel(session).await),
        ApiRequest::Fleet => ApiResponse::Fleet(api.fleet().await),
        ApiRequest::Tree => ApiResponse::Tree(api.tree().await),
        ApiRequest::Unit { unit } => ApiResponse::Unit(api.unit(unit).await),
        ApiRequest::UnitEvents { unit, max } => {
            ApiResponse::UnitEvents(api.unit_events(unit, max).await)
        }
        ApiRequest::UnitOutbound { unit, max } => {
            // Reuses the `Drained(Vec<Outbound>)` response — the same rich §17 drain shape as `poll`.
            ApiResponse::Drained(api.unit_outbound(unit, max).await)
        }
        ApiRequest::UnitHistory {
            unit,
            after_cursor,
            max,
        } => ApiResponse::Journal(api.unit_history(unit, after_cursor, max).await),
        ApiRequest::Pause { unit } => unit_or_err(api.pause(unit).await),
        ApiRequest::Resume { unit } => unit_or_err(api.resume(unit).await),
        ApiRequest::Scale { unit, n } => unit_or_err(api.scale(unit, n).await),
        ApiRequest::VerifyingKey => ApiResponse::VerifyingKey(api.verifying_key().await),
        ApiRequest::ModelSearch { query } => match api.model_search(query).await {
            Ok(page) => ApiResponse::ModelSearch(page),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ModelFiles {
            repo,
            revision,
            engine,
        } => match api.model_files(repo, revision, engine).await {
            Ok(files) => ApiResponse::ModelFiles(files),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ModelDownload { model } => match api.model_download(model).await {
            Ok(id) => ApiResponse::ModelDownloadStarted(id),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ModelDownloads => ApiResponse::ModelDownloads(api.model_downloads().await),
        ApiRequest::ModelCancel { id } => unit_or_err(api.model_cancel(id).await),
        ApiRequest::ModelPause { id } => unit_or_err(api.model_pause(id).await),
        ApiRequest::ModelResume { id } => unit_or_err(api.model_resume(id).await),
        ApiRequest::ModelCatalog => ApiResponse::ModelCatalog(api.model_catalog().await),
        ApiRequest::ModelDelete { id } => unit_or_err(api.model_delete(id).await),
        ApiRequest::ModelActivate { id, profile } => {
            unit_or_err(api.model_activate(id, profile).await)
        }
        ApiRequest::ModelRecommend {
            repo,
            revision,
            engine,
            budget_bytes,
        } => match api
            .model_recommend(repo, revision, engine, budget_bytes)
            .await
        {
            Ok(rec) => ApiResponse::ModelRecommend(rec),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ModelQuantize {
            repo,
            revision,
            target_quant,
            source_file,
        } => match api
            .model_quantize(repo, revision, target_quant, source_file)
            .await
        {
            Ok(id) => ApiResponse::ModelQuantizeStarted(id),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ModelQuantizes => ApiResponse::ModelQuantizes(api.model_quantizes().await),
        ApiRequest::ModelInspect { id } => match api.model_inspect(id).await {
            Ok(info) => ApiResponse::ModelInspect(info),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ProfileList => ApiResponse::Profiles(api.profile_list().await),
        ApiRequest::ProfileGet { id } => match api.profile_get(id).await {
            Ok(spec) => ApiResponse::Profile(spec),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ProfileCreate { spec } => unit_or_err(api.profile_create(spec).await),
        ApiRequest::ProfileUpdate { spec } => unit_or_err(api.profile_update(spec).await),
        ApiRequest::ProfileDelete { id } => unit_or_err(api.profile_delete(id).await),
        ApiRequest::ProfileSelect { id } => unit_or_err(api.profile_select(id).await),
        ApiRequest::ProfileClone { source, new_id } => {
            unit_or_err(api.profile_clone(source, new_id).await)
        }
        ApiRequest::ProfileExport { id } => match api.profile_export(id).await {
            Ok(dist) => ApiResponse::Distribution(dist),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ProfileImport { dist, new_id } => {
            match api.profile_import(dist, new_id).await {
                Ok(id) => ApiResponse::ProfileId(id),
                Err(e) => ApiResponse::Error(e),
            }
        }
        ApiRequest::ProfileHistory { id } => match api.profile_history(id).await {
            Ok(revs) => ApiResponse::Revisions(revs),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ProfileAt { id, seq } => match api.profile_at(id, seq).await {
            Ok(spec) => ApiResponse::Profile(Some(spec)),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ProfileRevert { id, seq } => unit_or_err(api.profile_revert(id, seq).await),
        ApiRequest::SkillHistory { name } => match api.skill_history(name).await {
            Ok(revs) => ApiResponse::Revisions(revs),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::SkillAt { name, seq } => match api.skill_at(name, seq).await {
            Ok(bundle) => ApiResponse::SkillBundle(bundle),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::SkillRevert { name, seq } => unit_or_err(api.skill_revert(name, seq).await),
        ApiRequest::CuratorList { profile } => match api.curator_list(profile).await {
            Ok(entries) => ApiResponse::CuratorSkills(entries),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::CuratorPin { profile, name } => {
            unit_or_err(api.curator_pin(profile, name).await)
        }
        ApiRequest::CuratorUnpin { profile, name } => {
            unit_or_err(api.curator_unpin(profile, name).await)
        }
        ApiRequest::CuratorArchive { profile, name } => {
            unit_or_err(api.curator_archive(profile, name).await)
        }
        ApiRequest::CuratorRestore { profile, name } => {
            unit_or_err(api.curator_restore(profile, name).await)
        }
        ApiRequest::CuratorRun { profile } => match api.curator_run(profile).await {
            Ok(changes) => ApiResponse::CuratorRun(changes),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::CredentialSet { profile, secret } => {
            unit_or_err(api.credential_set(profile, secret).await)
        }
        ApiRequest::CredentialList => ApiResponse::Credentials(api.credential_list().await),
        ApiRequest::CredentialRemove { profile } => {
            unit_or_err(api.credential_remove(profile).await)
        }
        ApiRequest::AuthBegin(req) => match api.auth_begin(req).await {
            Ok(r) => ApiResponse::AuthBegun(r),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::AuthComplete(req) => match api.auth_complete(req).await {
            Ok(r) => ApiResponse::AuthCompleted(r),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::AuthCancel { flow_id } => unit_or_err(api.auth_cancel(flow_id).await),
        ApiRequest::AuthProviders => ApiResponse::AuthProviders(api.auth_providers().await),
        ApiRequest::Models => ApiResponse::Models(api.models().await),
        ApiRequest::ModelCurrent { profile } => match api.model_current(profile).await {
            Ok(m) => ApiResponse::ModelCurrent(m),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::SessionsQuery { query } => {
            ApiResponse::SessionPage(api.sessions_query(query).await)
        }
        ApiRequest::SessionGet { session } => {
            ApiResponse::SessionDetail(api.session_get(session).await)
        }
        ApiRequest::SessionsByProfile => {
            ApiResponse::SessionsByProfile(api.sessions_by_profile().await)
        }
        ApiRequest::SessionSearch { query, limit } => {
            ApiResponse::SessionSearch(api.session_search(query, limit).await)
        }
        ApiRequest::SessionUpdateMeta { session, patch } => {
            unit_or_err(api.session_update_meta(session, patch).await)
        }
        ApiRequest::Rewind { session, point } => unit_or_err(api.rewind(session, point).await),
        ApiRequest::AcpDiscover => ApiResponse::AcpCatalog(api.acp_discover().await),
        ApiRequest::AcpCatalog => ApiResponse::AcpCatalog(api.acp_catalog().await),
        ApiRequest::AcpRegister { entry } => unit_or_err(api.acp_register(entry).await),
        ApiRequest::AcpRemove { name } => unit_or_err(api.acp_remove(name).await),
        ApiRequest::SkillGet { name } => match api.skill_get(name).await {
            Ok(bundle) => ApiResponse::SkillBundle(bundle),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::SkillPut { bundle } => unit_or_err(api.skill_put(bundle).await),
        ApiRequest::ProviderList => ApiResponse::Providers(api.provider_list().await),
        ApiRequest::ProviderRegister { provider } => {
            unit_or_err(api.provider_register(provider).await)
        }
        ApiRequest::ToolList => ApiResponse::Tools(api.tool_list().await),
        ApiRequest::ToolRegister { tool } => unit_or_err(api.tool_register(tool).await),
        ApiRequest::CommandList => ApiResponse::Commands(api.command_list().await),
        ApiRequest::CommandInvoke { invocation } => match api.command_invoke(invocation).await {
            Ok(out) => ApiResponse::CommandOutput(out),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ConfigGet => match api.config_get().await {
            Ok(c) => ApiResponse::Config(c),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ConfigSet { config } => unit_or_err(api.config_set(config).await),
        ApiRequest::CronList => ApiResponse::CronJobs(api.cron_list().await),
        ApiRequest::CronCreate { spec } => match api.cron_create(spec).await {
            Ok(id) => ApiResponse::CronId(id),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::CronUpdate { id, spec } => unit_or_err(api.cron_update(id, spec).await),
        ApiRequest::CronDelete { id } => unit_or_err(api.cron_delete(id).await),
        ApiRequest::CronTrigger { id } => unit_or_err(api.cron_trigger(id).await),
        ApiRequest::CronRuns { id } => ApiResponse::CronRuns(api.cron_runs(id).await),
        ApiRequest::CronPause { id, paused } => unit_or_err(api.cron_pause(id, paused).await),
        ApiRequest::CronSuggestions => ApiResponse::CronSuggestions(api.cron_suggestions().await),
        ApiRequest::CronAcceptSuggestion { id } => match api.cron_accept_suggestion(id).await {
            Ok(id) => ApiResponse::CronId(id),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::CronDismissSuggestion { id } => {
            unit_or_err(api.cron_dismiss_suggestion(id).await)
        }
        ApiRequest::RoutingListChats => ApiResponse::ChatRoutes(api.routing_list_chats().await),
        ApiRequest::RoutingGet { origin } => ApiResponse::ChatRoute(api.routing_get(origin).await),
        ApiRequest::RoutingSet { route } => unit_or_err(api.routing_set(route).await),
        ApiRequest::RoutingBindChat {
            origin,
            session,
            profile,
        } => unit_or_err(api.routing_bind_chat(origin, session, profile).await),
        ApiRequest::RoutingUnbindChat { origin } => {
            unit_or_err(api.routing_unbind_chat(origin).await)
        }
        ApiRequest::TransportRooms { transport } => {
            ApiResponse::Rooms(api.transport_rooms(transport).await)
        }
        ApiRequest::ConvList { transport } => {
            ApiResponse::Conversations(api.conv_list(transport).await)
        }
        ApiRequest::ConvGet { transport, conv } => {
            ApiResponse::Conversation(api.conv_get(transport, conv).await)
        }
        ApiRequest::ConvCreateDetails { transport } => {
            ApiResponse::ConvCreateDetails(api.conv_create_details(transport).await)
        }
        ApiRequest::ConvCreate { transport, details } => {
            match api.conv_create(transport, details).await {
                Ok(info) => ApiResponse::Conversation(Some(info)),
                Err(e) => ApiResponse::Error(e),
            }
        }
        ApiRequest::ConvJoinDetails { transport } => {
            ApiResponse::ConvJoinDetails(api.conv_join_details(transport).await)
        }
        ApiRequest::ConvJoin { transport, details } => {
            match api.conv_join(transport, details).await {
                Ok(info) => ApiResponse::Conversation(Some(info)),
                Err(e) => ApiResponse::Error(e),
            }
        }
        ApiRequest::ConvLeave { transport, conv } => {
            unit_or_err(api.conv_leave(transport, conv).await)
        }
        ApiRequest::ConvSend {
            transport,
            conv,
            from,
            message,
        } => unit_or_err(api.conv_send(transport, conv, from, message).await),
        ApiRequest::ConvSetTopic {
            transport,
            conv,
            topic,
        } => unit_or_err(api.conv_set_topic(transport, conv, topic).await),
        ApiRequest::ConvSetTitle {
            transport,
            conv,
            title,
        } => unit_or_err(api.conv_set_title(transport, conv, title).await),
        ApiRequest::ConvSetDescription {
            transport,
            conv,
            description,
        } => unit_or_err(api.conv_set_description(transport, conv, description).await),
        ApiRequest::ConvDelete { transport, conv } => {
            unit_or_err(api.conv_delete(transport, conv).await)
        }
        ApiRequest::ConvHistory {
            transport,
            conv,
            after_cursor,
            max,
        } => ApiResponse::Journal(api.conv_history(transport, conv, after_cursor, max).await),
        ApiRequest::MemberInvite {
            transport,
            conv,
            who,
            message,
        } => unit_or_err(api.member_invite(transport, conv, who, message).await),
        ApiRequest::MemberRemove {
            transport,
            conv,
            who,
            reason,
        } => unit_or_err(api.member_remove(transport, conv, who, reason).await),
        ApiRequest::MemberBan {
            transport,
            conv,
            who,
            reason,
        } => unit_or_err(api.member_ban(transport, conv, who, reason).await),
        ApiRequest::MemberSetRole {
            transport,
            conv,
            who,
            role,
        } => unit_or_err(api.member_set_role(transport, conv, who, role).await),
        ApiRequest::ContactGetProfile { transport, contact } => {
            match api.contact_get_profile(transport, contact).await {
                Ok(profile) => ApiResponse::ContactProfile(profile),
                Err(e) => ApiResponse::Error(e),
            }
        }
        ApiRequest::ContactSetAlias {
            transport,
            contact,
            alias,
        } => unit_or_err(api.contact_set_alias(transport, contact, alias).await),
        ApiRequest::ContactActionMenu { transport, contact } => {
            ApiResponse::ActionMenu(api.contact_action_menu(transport, contact).await)
        }
        ApiRequest::DirectorySearch { transport, query } => {
            match api.directory_search(transport, query).await {
                Ok(contacts) => ApiResponse::Contacts(contacts),
                Err(e) => ApiResponse::Error(e),
            }
        }
        ApiRequest::TransportAdapters => ApiResponse::Adapters(api.transport_adapters().await),
        ApiRequest::TransportInstances => {
            ApiResponse::TransportInstances(api.transport_instances().await)
        }
        ApiRequest::FsRoots => ApiResponse::FsRoots(api.fs_roots().await),
        ApiRequest::FsList {
            root,
            dir,
            show_ignored,
        } => match api.fs_list(root, dir, show_ignored).await {
            Ok(entries) => ApiResponse::FsList(entries),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::FsStat { root, path } => match api.fs_stat(root, path).await {
            Ok(entry) => ApiResponse::FsStat(entry),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::FsRead {
            root,
            path,
            max_bytes,
        } => match api.fs_read(root, path, max_bytes).await {
            Ok(content) => ApiResponse::FsRead(content),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::FsWrite {
            root,
            path,
            bytes,
            base_revision,
            force,
        } => match api.fs_write(root, path, bytes, base_revision, force).await {
            Ok(rev) => ApiResponse::FsWrite(rev),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::FsSearch { root, query } => match api.fs_search(root, query).await {
            Ok(page) => ApiResponse::FsSearch(page),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::FsWatchPoll {
            root,
            dir,
            after_seq,
            max,
        } => match api.fs_watch_after(root, dir, after_seq, max).await {
            Ok(page) => ApiResponse::FsWatch(page),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::BlobPut { bytes } => match api.blob_put(bytes).await {
            Ok(blob_ref) => ApiResponse::BlobPut(blob_ref),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::BlobGet { hash, range } => match api.blob_get(hash, range).await {
            Ok(bytes) => ApiResponse::BlobGet(bytes),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::BlobStat { hash } => ApiResponse::BlobStat(api.blob_stat(hash).await),
        ApiRequest::FsWriteFromBlob {
            root,
            path,
            hash,
            base_revision,
            force,
        } => match api
            .fs_write_from_blob(root, path, hash, base_revision, force)
            .await
        {
            Ok(rev) => ApiResponse::FsWrite(rev),
            Err(e) => ApiResponse::Error(e),
        },
        // Session variants were handled by `serve_session`.
        ApiRequest::Submit { .. }
        | ApiRequest::SubmitRouted { .. }
        | ApiRequest::Poll { .. }
        | ApiRequest::Respond { .. }
        | ApiRequest::SessionHistory { .. }
        | ApiRequest::Subscribe { .. }
        | ApiRequest::DeliveryTargets { .. }
        | ApiRequest::DeliverySessions { .. }
        | ApiRequest::Handover { .. }
        | ApiRequest::RecordMeta { .. }
        | ApiRequest::SetSessionModel { .. }
        | ApiRequest::SetSessionMode { .. }
        | ApiRequest::SetSessionOverlay { .. } => {
            unreachable!("session variants handled above")
        }
    }
}

/// Dispatch against a **session-only** surface — the entry point the `daemon-core-ffi` transport
/// calls. Control-surface requests resolve to [`ApiError::Unsupported`] (this transport is the §17
/// brain seam, not the node control plane).
pub async fn dispatch_session(api: &dyn SessionApi, req: ApiRequest) -> ApiResponse {
    match serve_session(api, req).await {
        Some(resp) => resp,
        None => ApiResponse::Error(ApiError::Unsupported(
            "control surface is not available on this transport".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// CBOR codec helpers (the one encoding shared by every framed transport)
// ---------------------------------------------------------------------------

/// CBOR-encode a mirror value (request or response).
pub fn to_cbor<T: Serialize>(value: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    // Mirror types are always serializable; a failure here is a programming error.
    ciborium::into_writer(value, &mut buf).expect("encode api mirror as CBOR");
    buf
}

/// CBOR-decode a mirror value, mapping a decode failure to [`ApiError::Other`].
pub fn from_cbor<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ApiError> {
    ciborium::from_reader(bytes).map_err(|e| ApiError::Other(format!("CBOR decode: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::ReqId;
    use daemon_protocol::{AgentEvent, EndReason, TurnSummary, UserMsg};

    #[test]
    fn request_cbor_round_trips() {
        let reqs = vec![
            ApiRequest::Health,
            ApiRequest::Assign {
                session: SessionId::new("s1"),
            },
            ApiRequest::Submit {
                session: SessionId::new("s1"),
                command: AgentCommand::StartTurn {
                    input: UserMsg::new("hi"),
                    request_id: ReqId(7),
                },
                origin: None,
                profile: None,
            },
            // The origin-carrying form (per-event attribution) must round-trip too.
            ApiRequest::Submit {
                session: SessionId::new("s1"),
                command: AgentCommand::Steer {
                    request_id: ReqId(8),
                    text: "go on".into(),
                },
                origin: Some(daemon_protocol::Origin::new(
                    "telegram",
                    daemon_protocol::OriginScope::Dm { user: "u1".into() },
                )),
                profile: Some(ProfileRef::new("agent-x")),
            },
            ApiRequest::Poll {
                session: SessionId::new("s1"),
                max: 16,
            },
        ];
        for req in reqs {
            let bytes = to_cbor(&req);
            let back: ApiRequest = from_cbor(&bytes).unwrap();
            assert_eq!(req, back);
        }
    }

    #[test]
    fn messaging_requests_and_responses_round_trip() {
        let transport = TransportId::new("room");
        let who = Participant::Agent {
            profile: ProfileRef::new("opus"),
            member: "@bot".into(),
        };
        let reqs = vec![
            ApiRequest::ConvList {
                transport: transport.clone(),
            },
            ApiRequest::ConvGet {
                transport: transport.clone(),
                conv: "r1".into(),
            },
            ApiRequest::ConvCreateDetails {
                transport: transport.clone(),
            },
            ApiRequest::ConvCreate {
                transport: transport.clone(),
                details: CreateConversationDetails::default(),
            },
            ApiRequest::ConvJoinDetails {
                transport: transport.clone(),
            },
            ApiRequest::ConvJoin {
                transport: transport.clone(),
                details: ChannelJoinDetails::default(),
            },
            ApiRequest::ConvLeave {
                transport: transport.clone(),
                conv: "r1".into(),
            },
            ApiRequest::ConvSend {
                transport: transport.clone(),
                conv: "r1".into(),
                from: Some(who.clone()),
                message: UserMsg::new("hi"),
            },
            ApiRequest::ConvSetTopic {
                transport: transport.clone(),
                conv: "r1".into(),
                topic: Some("t".into()),
            },
            ApiRequest::ConvSetTitle {
                transport: transport.clone(),
                conv: "r1".into(),
                title: None,
            },
            ApiRequest::ConvSetDescription {
                transport: transport.clone(),
                conv: "r1".into(),
                description: Some("d".into()),
            },
            ApiRequest::ConvDelete {
                transport: transport.clone(),
                conv: "r1".into(),
            },
            ApiRequest::ConvHistory {
                transport: transport.clone(),
                conv: "r1".into(),
                after_cursor: 0,
                max: 16,
            },
            ApiRequest::MemberInvite {
                transport: transport.clone(),
                conv: "r1".into(),
                who: who.clone(),
                message: None,
            },
            ApiRequest::MemberRemove {
                transport: transport.clone(),
                conv: "r1".into(),
                who: who.clone(),
                reason: Some("bye".into()),
            },
            ApiRequest::MemberBan {
                transport: transport.clone(),
                conv: "r1".into(),
                who: who.clone(),
                reason: None,
            },
            ApiRequest::MemberSetRole {
                transport: transport.clone(),
                conv: "r1".into(),
                who: who.clone(),
                role: MemberRole::Op,
            },
            ApiRequest::ContactGetProfile {
                transport: transport.clone(),
                contact: ContactInfo {
                    id: "@alice:hs".into(),
                    ..ContactInfo::default()
                },
            },
            ApiRequest::ContactSetAlias {
                transport: transport.clone(),
                contact: ContactInfo {
                    id: "@alice:hs".into(),
                    ..ContactInfo::default()
                },
                alias: Some("Ali".into()),
            },
            ApiRequest::ContactActionMenu {
                transport: transport.clone(),
                contact: ContactInfo {
                    id: "@alice:hs".into(),
                    ..ContactInfo::default()
                },
            },
            ApiRequest::DirectorySearch {
                transport: transport.clone(),
                query: Some("ali".into()),
            },
            ApiRequest::TransportAdapters,
            ApiRequest::TransportInstances,
        ];
        for req in reqs {
            let bytes = to_cbor(&req);
            let back: ApiRequest = from_cbor(&bytes).unwrap();
            assert_eq!(req, back);
        }

        let info = ConversationInfo {
            transport: transport.clone(),
            id: "r1".into(),
            kind: ConversationType::GroupDm,
            title: Some("Room 1".into()),
            topic: None,
            description: None,
            members: vec![ConversationMember {
                contact: ContactInfo {
                    id: "@bot".into(),
                    display_name: None,
                    presence: Presence::default(),
                    permission: ContactPermission::Unset,
                },
                alias: None,
                nickname: None,
                typing: TypingState::None,
                role: MemberRole::Op,
                session: Some(SessionId::new("sess-1")),
            }],
        };
        let resps = vec![
            ApiResponse::Conversations(vec![info.clone()]),
            ApiResponse::Conversation(Some(info.clone())),
            ApiResponse::ConvCreateDetails(CreateConversationDetails::default()),
            ApiResponse::ConvJoinDetails(ChannelJoinDetails::default()),
            ApiResponse::TransportInstances(vec![TransportInstanceInfo {
                transport: transport.clone(),
                family: "room".into(),
                display_name: "Rooms".into(),
                connection: ConnectionState::Connected,
                presence: PresenceState::Unknown,
                bound_profile: None,
            }]),
            ApiResponse::Adapters(vec![AdapterInfo {
                family: "room".into(),
                display_name: "Rooms".into(),
                capabilities: AdapterCapabilities::default(),
                account_schema: AccountSettingsSchema::default(),
            }]),
            ApiResponse::ContactProfile("display_name: Alice".into()),
            ApiResponse::Contacts(vec![info.members[0].contact.clone()]),
            ApiResponse::ActionMenu(Some(ActionMenu {
                items: vec!["Block".into()],
            })),
            ApiResponse::ActionMenu(None),
        ];
        for resp in resps {
            let bytes = to_cbor(&resp);
            let back: ApiResponse = from_cbor(&bytes).unwrap();
            assert_eq!(resp, back);
        }
    }

    #[test]
    fn delivery_and_meta_requests_round_trip() {
        let reqs = vec![
            ApiRequest::DeliveryTargets {
                session: SessionId::new("s1"),
            },
            ApiRequest::Handover {
                session: SessionId::new("s1"),
                target: daemon_protocol::DeliveryTarget::new(
                    "telegram",
                    "chat-9",
                    daemon_protocol::SinkKind::Primary,
                ),
            },
            ApiRequest::RecordMeta {
                session: SessionId::new("s1"),
                origin: daemon_protocol::Origin::new(
                    "gui",
                    daemon_protocol::OriginScope::Api {
                        key: "owner".into(),
                    },
                ),
                kind: "attach".into(),
                body: vec![9, 9, 9],
            },
        ];
        for req in reqs {
            let bytes = to_cbor(&req);
            let back: ApiRequest = from_cbor(&bytes).unwrap();
            assert_eq!(req, back);
        }

        let resp = ApiResponse::DeliveryTargets(vec![daemon_protocol::DeliveryTarget::new(
            "telegram",
            "chat-9",
            daemon_protocol::SinkKind::Primary,
        )]);
        let bytes = to_cbor(&resp);
        let back: ApiResponse = from_cbor(&bytes).unwrap();
        assert_eq!(resp, back);
    }

    #[test]
    fn cron_requests_and_responses_round_trip() {
        // A fully-populated spec exercises every additive v15/v16/v17 field through CBOR.
        let spec = CronSpec {
            name: "nightly-digest".into(),
            schedule: "0 9 * * *".into(),
            target: Some("opus".into()),
            payload: b"summarize today".to_vec(),
            enabled: true,
            timezone: Some("Europe/Berlin".into()),
            repeat: Some(7),
            jitter_secs: Some(30),
            overlap: OverlapPolicy::Queue,
            catch_up: CatchUpPolicy::Always,
            script: Some("scripts/collect.sh".into()),
            no_agent: false,
            context_from: vec!["job-a".into(), "job-b".into()],
            deliver: Some("origin".into()),
            enabled_toolsets: Some(vec!["fs".into()]),
            workdir: Some("/srv/proj".into()),
            model: Some("gpt-5".into()),
            provider: Some("genai".into()),
            skills: vec!["briefing".into(), "calendar".into()],
            origin: Some(Origin::new(
                "slack",
                daemon_protocol::OriginScope::Group {
                    chat: "C123".into(),
                    thread: None,
                },
            )),
        };
        let reqs = vec![
            ApiRequest::CronList,
            ApiRequest::CronCreate { spec: spec.clone() },
            ApiRequest::CronUpdate {
                id: "j1".into(),
                spec: spec.clone(),
            },
            ApiRequest::CronPause {
                id: "j1".into(),
                paused: true,
            },
            ApiRequest::CronTrigger { id: "j1".into() },
            ApiRequest::CronRuns { id: "j1".into() },
            ApiRequest::CronSuggestions,
            ApiRequest::CronAcceptSuggestion { id: "s1".into() },
            ApiRequest::CronDismissSuggestion { id: "s1".into() },
        ];
        for req in reqs {
            let bytes = to_cbor(&req);
            let back: ApiRequest = from_cbor(&bytes).unwrap();
            assert_eq!(req, back);
        }

        let job = CronJob {
            id: "j1".into(),
            spec: spec.clone(),
            next_fire_unix: Some(1_900_000_000),
            paused: false,
            last_run_unix: Some(1_800_000_000),
            last_ok: Some(true),
            last_detail: Some("ok".into()),
            fire_count: 3,
        };
        let run = CronRun {
            started_unix: 1_800_000_000,
            ok: false,
            detail: Some("boom".into()),
            finished_unix: Some(1_800_000_050),
            session: Some(SessionId::new("cron_j1_20260624")),
            trigger: RunTrigger::Manual,
        };
        let suggestion = CronSuggestion {
            id: "s1".into(),
            title: "Daily briefing".into(),
            description: "A morning digest".into(),
            source: "catalog".into(),
            spec,
            dedup_key: "catalog:daily-briefing".into(),
            status: SuggestionStatus::Pending,
        };
        let resps = vec![
            ApiResponse::CronJobs(vec![job]),
            ApiResponse::CronId("j1".into()),
            ApiResponse::CronRuns(vec![run]),
            ApiResponse::CronSuggestions(vec![suggestion]),
        ];
        for resp in resps {
            let bytes = to_cbor(&resp);
            let back: ApiResponse = from_cbor(&bytes).unwrap();
            assert_eq!(resp, back);
        }
    }

    #[test]
    fn fs_requests_and_responses_round_trip() {
        let reqs = vec![
            ApiRequest::FsRoots,
            ApiRequest::FsList {
                root: FsRootId::Host("home".into()),
                dir: "projects".into(),
                show_ignored: true,
            },
            ApiRequest::FsStat {
                root: FsRootId::Workspace,
                path: "src/main.rs".into(),
            },
            ApiRequest::FsRead {
                root: FsRootId::Session(SessionId::new("s1")),
                path: "README.md".into(),
                max_bytes: 1024,
            },
            ApiRequest::FsWrite {
                root: FsRootId::Session(SessionId::new("s1")),
                path: "a.txt".into(),
                bytes: vec![1, 2, 3],
                base_revision: Some(FsRevision {
                    mtime_ms: 10,
                    size: 3,
                }),
                force: false,
            },
            ApiRequest::FsSearch {
                root: FsRootId::Workspace,
                query: FsSearchQuery {
                    query: "TODO".into(),
                    regex: false,
                    case_sensitive: false,
                    max_results: 50,
                    page: 0,
                },
            },
            ApiRequest::FsWatchPoll {
                root: FsRootId::Workspace,
                dir: String::new(),
                after_seq: 4,
                max: 32,
            },
        ];
        for req in reqs {
            let bytes = to_cbor(&req);
            let back: ApiRequest = from_cbor(&bytes).unwrap();
            assert_eq!(req, back);
        }

        let resps = vec![
            ApiResponse::FsRoots(vec![FsRoot {
                id: FsRootId::Workspace,
                label: "workspace".into(),
                kind: FsRootKind::Workspace,
                session: None,
            }]),
            ApiResponse::FsList(vec![FsEntry {
                name: "src".into(),
                path: "src".into(),
                kind: FsEntryKind::Dir,
                size: 0,
                mtime_ms: 1,
                ignored: false,
            }]),
            ApiResponse::FsRead(FsContent {
                bytes: vec![9, 9],
                revision: FsRevision {
                    mtime_ms: 2,
                    size: 2,
                },
                truncated: true,
                blob_ref: None,
            }),
            ApiResponse::FsWrite(FsRevision {
                mtime_ms: 3,
                size: 7,
            }),
            ApiResponse::FsWatch(FsWatchPageView {
                events: vec![FsChange {
                    path: "a.txt".into(),
                    kind: FsChangeKind::Modified,
                }],
                next_seq: 5,
                head_seq: 5,
                reset: false,
            }),
        ];
        for resp in resps {
            let bytes = to_cbor(&resp);
            let back: ApiResponse = from_cbor(&bytes).unwrap();
            assert_eq!(resp, back);
        }
    }

    #[test]
    fn blob_requests_and_responses_round_trip() {
        let hash = ContentHash::new([7u8; 32]);
        let reqs = vec![
            ApiRequest::BlobPut {
                bytes: vec![1, 2, 3],
            },
            ApiRequest::BlobGet {
                hash,
                range: Some(ByteRange { offset: 1, len: 2 }),
            },
            ApiRequest::BlobStat { hash },
            ApiRequest::FsWriteFromBlob {
                root: FsRootId::Session(SessionId::new("s1")),
                path: "out/x.bin".into(),
                hash,
                base_revision: None,
                force: false,
            },
        ];
        for req in reqs {
            let bytes = to_cbor(&req);
            let back: ApiRequest = from_cbor(&bytes).unwrap();
            assert_eq!(req, back);
        }

        let resps = vec![
            ApiResponse::BlobPut(BlobRef::new(hash, 3)),
            ApiResponse::BlobGet(vec![2, 3]),
            ApiResponse::BlobStat(BlobStat {
                size: 3,
                present: true,
            }),
        ];
        for resp in resps {
            let bytes = to_cbor(&resp);
            let back: ApiResponse = from_cbor(&bytes).unwrap();
            assert_eq!(resp, back);
        }
    }

    #[test]
    fn overlay_workspace_binding_round_trips() {
        let overlay = SessionOverlay {
            workspace: Some(WorkspaceBinding::Bound("/srv/projects/foo".into())),
            ..SessionOverlay::default()
        };
        let bytes = to_cbor(&overlay);
        let back: SessionOverlay = from_cbor(&bytes).unwrap();
        assert_eq!(overlay, back);
        assert!(!overlay.is_empty());
    }

    #[test]
    fn submit_origin_field_defaults_when_absent() {
        // An old-shape Submit encoded WITHOUT `origin` must still decode (serde default -> None),
        // proving the additive field is backward compatible on the v2 wire.
        #[derive(Serialize)]
        enum LegacyRequest {
            Submit {
                session: SessionId,
                command: AgentCommand,
            },
        }
        let legacy = LegacyRequest::Submit {
            session: SessionId::new("s1"),
            command: AgentCommand::Shutdown,
        };
        let bytes = to_cbor(&legacy);
        let back: ApiRequest = from_cbor(&bytes).unwrap();
        assert_eq!(
            back,
            ApiRequest::Submit {
                session: SessionId::new("s1"),
                command: AgentCommand::Shutdown,
                origin: None,
                profile: None,
            }
        );
    }

    #[test]
    fn response_cbor_round_trips() {
        let resp = ApiResponse::Drained(vec![Outbound::Event(AgentEvent::TurnFinished {
            seq: 3,
            summary: TurnSummary::ended(EndReason::Completed),
        })]);
        let bytes = to_cbor(&resp);
        let back: ApiResponse = from_cbor(&bytes).unwrap();
        assert_eq!(resp, back);
    }

    fn sample_info() -> SessionInfo {
        SessionInfo {
            session: SessionId::new("s1"),
            state: SessionState::Active,
            rewindable: true,
            bound_profile: Some(ProfileRef::new("agent-x")),
            title: Some("hello world".into()),
            last_activity_ms: Some(1_700_000_000_000),
            lifecycle: Lifecycle::Live,
            role: SessionRole::ManagedChild,
            parent: Some(SessionId::new("p1")),
            pinned: false,
            archived: false,
        }
    }

    #[test]
    fn roster_requests_and_responses_round_trip() {
        let reqs = vec![
            ApiRequest::SessionsQuery {
                query: SessionQuery {
                    scope: SessionScope::ByProfile(ProfileRef::new("agent-x")),
                    after: Some(SessionId::new("s0")),
                    limit: 25,
                    since_rev: Some(12),
                },
            },
            ApiRequest::SessionGet {
                session: SessionId::new("s1"),
            },
            ApiRequest::SessionsByProfile,
            ApiRequest::SessionSearch {
                query: "build".into(),
                limit: 10,
            },
            ApiRequest::Rewind {
                session: SessionId::new("s1"),
                point: RewindPoint {
                    anchor: RewindAnchor::UserTurn { ordinal: 3 },
                    restore_workspace: true,
                },
            },
            ApiRequest::AcpDiscover,
            ApiRequest::AcpRegister {
                entry: AcpAgentEntry {
                    name: "gemini".into(),
                    recipe: AcpRecipe {
                        program: Some("gemini".into()),
                        args: vec!["--acp".into()],
                        env: vec![("KEY".into(), "v".into())],
                        endpoint: None,
                    },
                    source: AcpSource::Builtin,
                    installed: true,
                    version: Some("0.1".into()),
                    capabilities: vec![("fs".into(), "true".into())],
                },
            },
        ];
        for req in reqs {
            assert_eq!(req, from_cbor::<ApiRequest>(&to_cbor(&req)).unwrap());
        }

        let resps = vec![
            ApiResponse::SessionPage(SessionPage {
                sessions: vec![sample_info()],
                next_cursor: Some(SessionId::new("s1")),
                rev: 7,
                removed: vec![SessionId::new("gone")],
            }),
            ApiResponse::SessionDetail(Some(SessionDetail {
                info: sample_info(),
                overlay: None,
                model: Some("groq::llama".into()),
                delivery_targets: vec![],
                children: vec![SessionId::new("c1")],
                checkpoints: 2,
            })),
            ApiResponse::SessionsByProfile(vec![(ProfileRef::new("agent-x"), vec![sample_info()])]),
            ApiResponse::SessionSearch(vec![SessionSearchHit {
                session: SessionId::new("s1"),
                title: "hello".into(),
                snippet: "…[hello]…".into(),
            }]),
        ];
        for resp in resps {
            assert_eq!(resp, from_cbor::<ApiResponse>(&to_cbor(&resp)).unwrap());
        }
    }

    #[test]
    fn session_info_defaults_when_enrichment_absent() {
        // An old-shape SessionInfo (only session/state) must still decode: the additive roster
        // fields fall back to their serde defaults (no profile/title/activity, durable, primary).
        #[derive(Serialize)]
        struct LegacyInfo {
            session: SessionId,
            state: SessionState,
        }
        let legacy = LegacyInfo {
            session: SessionId::new("s1"),
            state: SessionState::Ready,
        };
        let back: SessionInfo = from_cbor(&to_cbor(&legacy)).unwrap();
        assert_eq!(back.role, SessionRole::Primary);
        assert_eq!(back.lifecycle, Lifecycle::Durable);
        assert!(back.rewindable);
        assert!(back.bound_profile.is_none());
        assert!(back.parent.is_none());
    }

    #[test]
    fn tree_event_round_trips() {
        let ev = TreeEvent::Snapshot(TreeReport {
            root: Some(UnitId::new("root")),
            nodes: vec![UnitNode {
                id: UnitId::new("root"),
                kind: UnitKind::Orchestrator,
                state: UnitState::Running,
                work: None,
                usage: UsageDelta::default(),
                children: vec![],
                profile: None,
                session: None,
                title: None,
                role: None,
            }],
        });
        assert_eq!(ev, from_cbor::<TreeEvent>(&to_cbor(&ev)).unwrap());
    }

    #[test]
    fn subscribe_request_and_log_page_round_trip() {
        use daemon_protocol::{Direction, Disposition, Origin, OriginScope, SessionPayload};

        let req = ApiRequest::Subscribe {
            session: SessionId::new("s1"),
            after_seq: 12,
            max: 64,
        };
        assert_eq!(req, from_cbor::<ApiRequest>(&to_cbor(&req)).unwrap());

        let page = ApiResponse::LogPage(LogPageView {
            entries: vec![SessionLogEntry {
                seq: 13,
                direction: Direction::Inbound,
                origin: Origin::new("telegram", OriginScope::Dm { user: "u1".into() }),
                disposition: Disposition::Context,
                payload: SessionPayload::Command(AgentCommand::StartTurn {
                    input: UserMsg::new("hi"),
                    request_id: ReqId(1),
                }),
            }],
            next_seq: 13,
            head_seq: 13,
            epoch: 0,
        });
        assert_eq!(page, from_cbor::<ApiResponse>(&to_cbor(&page)).unwrap());
    }

    /// The default `SessionApi` implementations of the live-log reads degrade gracefully: an empty
    /// page and an empty stream, so a transport with no live log still satisfies the surface.
    #[tokio::test]
    async fn default_log_reads_are_empty() {
        use futures::StreamExt;

        struct Bare;
        #[async_trait]
        impl SessionApi for Bare {
            async fn submit(&self, _: SessionId, _: AgentCommand) -> Result<(), ApiError> {
                Ok(())
            }
            async fn poll(&self, _: SessionId, _: u32) -> Result<Vec<Outbound>, ApiError> {
                Ok(Vec::new())
            }
            async fn respond(&self, _: SessionId, _: HostResponse) -> Result<(), ApiError> {
                Ok(())
            }
        }

        let page = Bare.log_after(SessionId::new("s"), 0, 16).await.unwrap();
        assert_eq!(page, LogPageView::default());
        let mut s = Bare.subscribe(SessionId::new("s"), 0).await.unwrap();
        assert!(s.next().await.is_none());
    }
}
