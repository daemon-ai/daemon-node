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
use daemon_common::{
    DownloadId, DownloadStatus, GgufInfo, InstalledModel, ModelEngine, ModelFile, ModelId,
    ModelRef, ProfileRef, QuantRecommendation, QuantizeId, QuantizeStatus, SearchPage, SearchQuery,
    SessionId, UnitId, UsageDelta, WireVersion,
};
use std::collections::BTreeMap;
use daemon_protocol::{
    session_id_for, AgentCommand, DeliveryTarget, HostResponse, IsolationPolicy, Origin,
    RewindAnchor, TranscriptBlock, TransportId,
};
pub use daemon_common::{Author, Revision, RevisionKind, SkillBundle};
pub use daemon_protocol::{Outbound, SessionLogEntry};
use futures::stream::{self, BoxStream, StreamExt};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub mod profile;
pub use profile::{
    BoundAccount, BudgetSpec, ContextEngineSel, CredentialInfo, CuratorChange, CuratorEntry,
    Distribution, EngineTunables, MemoryProviderSel, ModelDescriptor, ProfileInfo, ProfileSpec,
    ProviderSelector, SessionOverlay, ToolsOverride,
};
pub use daemon_common::{SkillCreator, SkillState, SkillUsage};

/// A live, push-based stream of merged [`SessionLogEntry`] items (inbound + outbound), the delivery
/// shape a streaming transport (in-process, socket, HTTP/WS) returns from [`SessionApi::subscribe`].
/// Streaming is a *transport capability*, not a wire-mirror variant: the cursor read
/// ([`SessionApi::log_after`] / [`ApiRequest::Subscribe`]) is the one-shot/long-poll form every
/// transport marshals.
pub type LogStream = BoxStream<'static, SessionLogEntry>;

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

    /// Drain up to `max` outbound items (events + raised host requests) for `session`.
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

    /// Drain up to `max` recent management events for one unit (GUI drill-down). Default: empty.
    async fn unit_events(&self, _id: UnitId, _max: u32) -> Vec<ManageEventView> {
        Vec::new()
    }

    /// Drain up to `max` recent §17 [`Outbound`] items (streamed events + raised host requests) for
    /// one unit — the rich, transcript-fidelity drill-down a GUI reads to render a full transcript
    /// for *any* unit in the tree (not just a top-level interactive session). The coarse
    /// [`Self::unit_events`] is the fleet-dashboard view; this is the drill-down-to-transcript view,
    /// carrying the full §17 vocabulary (text, reasoning, tool I/O with opaque structured `detail`,
    /// opaque `ContentDelta`, usage, errors) plus blocking host requests, untouched. A destructive
    /// drain like [`Self::poll`] (each call consumes what it returns; `max == 0` drains all).
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthFlowKind {
    /// Matrix SSO: the redirect carries a single-use `loginToken`.
    MatrixSso,
    /// OAuth2 / OIDC authorization-code + PKCE: the redirect carries `code` + `state`.
    OAuth2Pkce,
}

/// Optionally bind the freshly-authenticated account to a profile on success.
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthCompleteRequest {
    /// The flow id from `auth_begin`.
    pub flow_id: String,
    /// The captured callback: the full redirect URL or just its query string (carries the
    /// `loginToken`, or `code` + `state`).
    pub callback: String,
}

/// The outcome of a completed flow.
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthReport {
    /// Whether every resident service is currently `Ok`.
    pub all_ok: bool,
    /// Per-service health.
    pub services: Vec<ServiceHealth>,
}

/// One resident service's health line.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lifecycle {
    /// Durable-managed (a `session_record` row; driven via [`ControlApi::assign`]).
    Durable,
    /// Live-interactive (an in-memory submit/poll chat; driven via [`SessionApi::submit`]).
    Live,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self::Durable
    }
}

/// A session's identity + lifecycle state + roster metadata. Enriched for the GUI roster: it carries
/// the bound profile (agent identity), an optional title, last-activity (for sort), the
/// durable-vs-live [`Lifecycle`], and the hierarchy [`SessionRole`] + `parent` so a client can keep
/// the `Primary` inbox separate from drill-down children.
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
}

/// `serde` default for [`SessionInfo::rewindable`] on older wire payloads that predate the field:
/// daemon-core sessions are rewindable, so the safe default is `true`.
fn default_rewindable() -> bool {
    true
}

/// The scope filter for [`ControlApi::sessions_query`] — the GUI roster query. The tree is the lazy
/// drill-down for children, so the default `TopLevel` returns only `Primary` conversations (the
/// inbox); the by-profile / by-transport scopes back the per-agent / per-transport views.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionScope {
    /// Only top-level (`Primary`) conversations — the inbox. Children are reached via `tree()`.
    TopLevel,
    /// Sessions bound to a specific profile (the per-agent view).
    ByProfile(ProfileRef),
    /// Sessions whose `Primary` delivery target names a specific transport instance.
    ByTransport(TransportId),
    /// Every session regardless of role (explicit opt-in; can be large in a fleets-of-fleets node).
    All,
}

impl Default for SessionScope {
    fn default() -> Self {
        Self::TopLevel
    }
}

/// A scoped, paginated roster query. The cursor is the last session id from the previous page
/// (`None` for the first page); `limit == 0` means a sensible server default.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionQuery {
    /// The roster scope filter.
    #[serde(default)]
    pub scope: SessionScope,
    /// The exclusive cursor: the last [`SessionId`] returned by the previous page (`None` = start).
    #[serde(default)]
    pub after: Option<SessionId>,
    /// Maximum sessions to return (`0` = a server default).
    #[serde(default)]
    pub limit: u32,
}

/// A page of the scoped roster: the matching sessions plus the cursor to fetch the next page
/// (`None` when the page is the last).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionPage {
    /// The sessions in this page (already scope-filtered + ordered).
    pub sessions: Vec<SessionInfo>,
    /// The cursor to pass as [`SessionQuery::after`] on the next read; `None` => no more pages.
    #[serde(default)]
    pub next_cursor: Option<SessionId>,
}

/// The full detail of one session — the single round-trip a GUI detail pane reads: roster `info`
/// plus the resolved overlay/model/provider, delivery targets, parent/children ids, and a
/// checkpoint count.
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewindPoint {
    /// The conversation anchor to truncate the transcript at.
    pub anchor: RewindAnchor,
    /// Whether to also roll the workspace back to the checkpoint captured at/just before `anchor`.
    #[serde(default)]
    pub restore_workspace: bool,
}

/// Where a cataloged ACP agent's launch recipe came from.
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInfo {
    /// The tool name (as used in `ProfileSpec.tool_allowlist`).
    pub name: String,
    /// A short human description, when known.
    #[serde(default)]
    pub description: Option<String>,
}

/// An opaque view of the node's runtime config (I13 stub DTO). Carried as a serialized blob so the
/// concrete `NodeConfig` (a binary-layer type) need not leak into the contract; the shape is the
/// stable wire envelope, the encoding fills in with the runtime.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeConfigView {
    /// The config encoding discriminator (e.g. `"toml"` / `"json"`).
    pub format: String,
    /// The serialized config body.
    pub body: String,
}

/// A scheduled-job spec (I15 stub DTO): when to fire and what to do.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronSpec {
    /// A human name for the job.
    pub name: String,
    /// The schedule expression (cron syntax / ISO interval — runtime-defined).
    pub schedule: String,
    /// The session/profile the job drives, when it targets one.
    #[serde(default)]
    pub target: Option<String>,
    /// The opaque payload/command the job submits when it fires.
    #[serde(default)]
    pub payload: Vec<u8>,
}

/// A scheduled job (I15 stub DTO): a [`CronSpec`] plus its id and next-fire time.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronJob {
    /// The opaque job id.
    pub id: String,
    /// The job spec.
    pub spec: CronSpec,
    /// Unix seconds of the next scheduled fire, when computed.
    #[serde(default)]
    pub next_fire_unix: Option<u64>,
}

/// One recorded run of a scheduled job (I15 stub DTO).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronRun {
    /// Unix seconds the run started.
    pub started_unix: u64,
    /// Whether the run succeeded.
    pub ok: bool,
    /// A rendered outcome detail, when present.
    #[serde(default)]
    pub detail: Option<String>,
}

/// A chat→session routing pin (I5; daemon-event-io-spec §5.9): an explicit binding of an inbound
/// [`Origin`] to a specific session (+ optional profile + session-naming isolation), surfaced by the
/// `routing_*` ops so a GUI/operator can pin a chat to a named conversation. The host consults pins
/// **resolve-first**, overriding the deterministic `session_id_for` derivation.
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
// Verifiable journal read DTOs (the non-destructive reconnect / scroll-back surface)
// ---------------------------------------------------------------------------

/// The decoded payload of one journal entry: a coarse management lifecycle record or a coalesced
/// finished chat block (a `daemon-protocol` [`TranscriptBlock`], already decoded for the consumer —
/// the GUI renders it directly, an auditor reads one timeline). Streaming deltas never appear here;
/// they stay on the ephemeral live drains.
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogPageView {
    /// The merged-log entries (inbound + outbound) in `seq` order.
    pub entries: Vec<SessionLogEntry>,
    /// The cursor to pass as the next `after_seq` (the last entry's `seq`, or the input `after_seq`
    /// when the page is empty).
    pub next_seq: u64,
    /// The highest `seq` currently retained for the session (how far a reader can advance now).
    pub head_seq: u64,
}

// ---------------------------------------------------------------------------
// The serializable mirror (1:1 with the interface methods)
// ---------------------------------------------------------------------------

/// The serializable reflection of a call into the interface — what every non-in-process transport
/// marshals onto the wire.
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
}

/// The serializable reflection of an interface result.
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
    /// The node runtime config (config_get).
    Config(NodeConfigView),
    /// The scheduled cron jobs (cron_list).
    CronJobs(Vec<CronJob>),
    /// A created cron job id (cron_create).
    CronId(String),
    /// Recent runs of a scheduled job (cron_runs).
    CronRuns(Vec<CronRun>),
    /// The chat→session routing pins (routing_list_chats).
    ChatRoutes(Vec<ChatRoute>),
    /// One origin's routing pin, if set (routing_get).
    ChatRoute(Option<ChatRoute>),
    /// A transport instance's rooms (transport_rooms).
    Rooms(Vec<RoomInfo>),
    /// A failure (the interface's `ApiError`, round-tripped faithfully).
    Error(ApiError),
}

/// Why an api call failed (serializable so it round-trips over any transport).
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
        ApiRequest::ProfileImport { dist, new_id } => match api.profile_import(dist, new_id).await {
            Ok(id) => ApiResponse::ProfileId(id),
            Err(e) => ApiResponse::Error(e),
        },
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
            }),
            ApiResponse::SessionDetail(Some(SessionDetail {
                info: sample_info(),
                overlay: None,
                model: Some("groq::llama".into()),
                delivery_targets: vec![],
                children: vec![SessionId::new("c1")],
                checkpoints: 2,
            })),
            ApiResponse::SessionsByProfile(vec![(
                ProfileRef::new("agent-x"),
                vec![sample_info()],
            )]),
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
