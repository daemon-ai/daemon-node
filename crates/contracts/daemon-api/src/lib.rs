// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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
    SearchQuery, SessionId, TraceId, UnitId, UsageDelta, WireVersion,
};
use daemon_protocol::{
    session_id_for, AgentCommand, DeliveryTarget, HostResponse, IsolationPolicy, Origin,
    RewindAnchor, ToolDetail, TranscriptBlock, TransportId, UserMsg,
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
    CustomProvider, CustomProviderSource, Distribution, EngineSelector, EngineTunables,
    ForeignBackend, MemoryProviderSel, ModelDescriptor, ProfileInfo, ProfileSpec,
    ProviderDescriptor, ProviderKindWire, ProviderSelector, ProviderSignIn, SessionOverlay,
    ToolsOverride,
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

    /// Whether this mode *widens* autonomy — auto-approves gated actions (`AcceptEdits` auto-allows
    /// workspace edits; `AutoAllow` auto-allows nearly everything) — vs. the safe directions (`Ask`,
    /// the default that prompts; `Deny`, the strictest that blocks). Widening a live session's
    /// autonomy is an operator-tier act (Cluster E): a non-operator may narrow (`Ask`/`Deny`) or
    /// switch model/provider on its own session, but not widen its approval posture.
    pub fn widens_autonomy(self) -> bool {
        matches!(self, ApprovalMode::AcceptEdits | ApprovalMode::AutoAllow)
    }

    /// The full set of advertisable modes (for a GUI mode picker / session-state advertisement).
    pub const ALL: [ApprovalMode; 4] = [
        ApprovalMode::Ask,
        ApprovalMode::AcceptEdits,
        ApprovalMode::AutoAllow,
        ApprovalMode::Deny,
    ];
}

// ---------------------------------------------------------------------------
// User feedback over OpenTelemetry (N1: API contract + node-owned consent)
// ---------------------------------------------------------------------------

/// The server-side cap on a feedback `comment`'s length, in bytes. The node rejects an over-long
/// comment ([`ApiError::Other`]) rather than truncating, so a client sees the failure.
pub const FEEDBACK_COMMENT_MAX: usize = 4096;

/// Which flavor of feedback a [`ApiRequest::FeedbackSubmit`] carries (wire v31): a reaction to a
/// specific agent response, or general free-form feedback about the app.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackKind {
    /// Feedback on a specific assistant response/turn (requires `target` + `rating`).
    Response,
    /// General app feedback (requires a `comment` or a `rating`).
    App,
}

/// A thumbs up / down reaction (wire v31).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackRating {
    /// Thumbs up (positive).
    Up,
    /// Thumbs down (negative).
    Down,
}

/// What a [`FeedbackKind::Response`] feedback points at (wire v31): the rated assistant
/// message/turn, addressed by its durable journal `cursor` within a session, plus an optional
/// trace-context handle for correlation with the emitted OTel log event.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackTarget {
    /// The session the rated response belongs to.
    pub session: SessionId,
    /// The durable journal cursor of the rated assistant message/turn.
    pub cursor: u64,
    /// The trace context of the rated turn, when the client has it (`None` otherwise).
    #[serde(default)]
    pub trace: Option<TraceId>,
}

/// Optional client-supplied diagnostics attached to a feedback submission (wire v31). Every field
/// is optional so a privacy-conscious client can omit it.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackDiagnostics {
    /// The submitting app's version string.
    #[serde(default)]
    pub app_version: Option<String>,
    /// The submitting app's OS/platform string.
    #[serde(default)]
    pub os: Option<String>,
}

/// The interface arg bundle for [`ControlApi::feedback_submit`] (C1: multi-field ops carry a struct
/// rather than a long positional list). Mirrors the [`ApiRequest::FeedbackSubmit`] wire fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedbackSubmitArgs {
    /// The feedback flavor (response vs. app).
    pub kind: FeedbackKind,
    /// The rated response, for [`FeedbackKind::Response`] (`None` for app feedback).
    pub target: Option<FeedbackTarget>,
    /// The thumbs up/down rating, when given.
    pub rating: Option<FeedbackRating>,
    /// A free-form comment, when given.
    pub comment: Option<String>,
    /// Whether the client consents to including the rated response content in the exported event.
    pub include_content: bool,
    /// Optional client diagnostics (app version / OS).
    pub diagnostics: Option<FeedbackDiagnostics>,
    /// The UI surface the feedback was given from (free-form label, e.g. `"transcript"`).
    pub surface: String,
}

/// The acknowledgement returned by [`ControlApi::feedback_submit`] and carried on the wire by
/// [`ApiResponse::FeedbackAck`]. `accepted`/`queued` mean the node validated the submission and
/// persisted it to the durable feedback outbox; it does **not** mean the OTel event was delivered
/// (export is a separate, best-effort drain).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackAck {
    /// The node validated + accepted the submission.
    pub accepted: bool,
    /// The submission was persisted to the durable feedback outbox.
    pub queued: bool,
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
    async fn submit_as(&self, args: SubmitAsArgs) -> Result<(), ApiError> {
        let SubmitAsArgs {
            session,
            origin,
            command,
            profile: _profile,
        } = args;
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

    /// Node-authoritative session creation: create a **blank, profile-bound, UN-RUN** session (no
    /// turn, no engine wake), persist it so it appears in the roster + the ByProfile query, emit the
    /// existing `RosterChanged`, and return the id. The node mints the id when `session` is `None`,
    /// or accepts a caller-supplied one; `profile` binds `bound_profile` (or the active default when
    /// `None`). This is the node-authority replacement for a client-minted session id: the client
    /// **requests**, the node **creates**, the node **events**, the client **updates** from the event.
    ///
    /// Default: unsupported (a session-only transport with no roster/profile store), so existing
    /// implementors keep compiling unchanged.
    async fn session_create(
        &self,
        _session: Option<SessionId>,
        _profile: Option<ProfileRef>,
    ) -> Result<SessionId, ApiError> {
        Err(ApiError::Unsupported("session_create".into()))
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
    /// `submit_routed`-returns-an-id path alone cannot cover after a restart). Paged at
    /// [`WIRE_PAGE_MAX`] in session-id order; `after` resumes past the previous page's `next`.
    /// Default: empty (a transport with no live delivery state).
    async fn delivery_sessions(
        &self,
        _transport: TransportId,
        _after: Option<String>,
    ) -> WirePage<SessionId> {
        WirePage::default()
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
    async fn record_meta(&self, _args: RecordMetaArgs) -> Result<(), ApiError> {
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

    /// Full-text search over indexed session text (title + coalesced body), most-relevant first,
    /// capped at `limit` (`0` => a server default). Default: empty (no text index).
    async fn session_search(&self, _query: String, _limit: u32) -> Vec<SessionSearchHit> {
        Vec::new()
    }

    /// A pure-local [`SessionRecap`] of one session's recent activity (no LLM call): scope counts,
    /// top tools, recently-touched files, and the last ask/reply — the hermes `/recap` analogue,
    /// computed node-side from the session's conversation. For a durable session the source is its
    /// **last checkpointed snapshot** (a resident mid-turn session recaps its last durable state);
    /// a resident live session is served from its live conversation view. `None` when the session
    /// is unknown, not visible to the caller, or has no recoverable conversation. Default: `None`.
    async fn session_recap(&self, _session: SessionId) -> Option<SessionRecap> {
        None
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
    /// fleets-of-fleets) with its parent/child structure, state, work, and folded usage. `nodes`
    /// is paged at [`WIRE_PAGE_MAX`] in unit-id order (`after` resumes past the report's `next`;
    /// `root` rides every page); the id-linked structure reassembles client-side regardless of
    /// page boundaries. The default is an empty tree (a transport with no fleet projection, e.g.
    /// the session-only FFI).
    async fn tree(&self, _after: Option<String>) -> TreeReport {
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
    /// given, else across all sessions (the operator HITL inbox). Paged at [`WIRE_PAGE_MAX`] in
    /// `request_id` order. Default: empty (a transport with no durable approval store).
    async fn approvals_pending(
        &self,
        _session: Option<SessionId>,
        _after: Option<String>,
    ) -> WirePage<ApprovalInfo> {
        WirePage::default()
    }

    /// Answer a parked §12 edit-approval request: record the operator's decision and wake the dormant
    /// session so it resumes (allow -> the gated tool runs; deny -> the tool returns an error). The
    /// `request_id` is the opaque id from [`Self::approvals_pending`]. `allow_permanent` (Cluster B)
    /// additionally remembers the approved command's fingerprint on the session allow-list when the
    /// parked approval carries one (so an identical in-session re-request auto-approves); it degrades
    /// to a single allow otherwise. `reason` (wire v29) is an optional operator justification: on a
    /// deny it becomes the gated tool's error content in the agent's conversation, so the model can
    /// adapt its next attempt; ignored on allow. Idempotent (a redelivered decision is a no-op).
    /// Default: unsupported (a transport with no durable approval store).
    async fn approval_decide(
        &self,
        _session: SessionId,
        _request_id: String,
        _allow: bool,
        _allow_permanent: bool,
        _reason: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("approval_decide".into()))
    }

    /// List a session's remembered exec-approval command fingerprints (wire v29; the
    /// `allow_permanent` allow-list `Effect::RememberApproval` / a durable allow-permanent decision
    /// records on the session snapshot), truncated at [`WIRE_PAGE_MAX`] (the allow-list is
    /// operator-curated and small). Reads the DURABLE snapshot: a live-resident session's
    /// in-memory list is not readable here (it is ephemeral — it dies with the residency).
    /// Default: unsupported (a transport with no durable session store).
    async fn fingerprint_list(
        &self,
        _session: SessionId,
    ) -> Result<Vec<RememberedFingerprint>, ApiError> {
        Err(ApiError::Unsupported("fingerprint_list".into()))
    }

    /// Revoke one remembered fingerprint from a session's `allow_permanent` allow-list (wire v29):
    /// the exact command stops auto-approving and the next identical request re-prompts. Applies to
    /// the DURABLE snapshot of a dormant session; the revoke takes effect at the session's next
    /// activation (a turn already running when the revoke lands still honors the old list — the
    /// documented one-round latency). A live-resident or actively-running session refuses with a
    /// clear error instead of silently losing the edit. Default: unsupported.
    async fn fingerprint_revoke(
        &self,
        _session: SessionId,
        _fingerprint: String,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("fingerprint_revoke".into()))
    }

    /// The node's journal **verifying** key (hex-encoded dCBOR), so an auditor can independently
    /// verify the sealed segments returned by the history reads. `None` when the node exposes no
    /// journal signer. Default: `None`.
    async fn verifying_key(&self) -> Option<String> {
        None
    }

    /// List recorded §12 tool checkpoints (workspace snapshots taken before a mutating tool ran) —
    /// for one `session` when given, else node-wide. Paged at [`WIRE_PAGE_MAX`] in checkpoint-id
    /// order (the uniform ascending-by-key cursor; consumers re-sort for a newest-first render).
    /// Default: empty (a node with no checkpoint store). The GUI renders these as rewind points.
    async fn checkpoints(
        &self,
        _session: Option<SessionId>,
        _after: Option<String>,
    ) -> WirePage<CheckpointInfo> {
        WirePage::default()
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

    /// List all chat→session routing pins, paged at [`WIRE_PAGE_MAX`] in origin-pin-key order (the
    /// store's `ORDER BY key`; the cursor is recomputed from each route's origin). Default: empty.
    async fn routing_list_chats(&self, _after: Option<String>) -> WirePage<ChatRoute> {
        WirePage::default()
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
    /// session for each when one exists. Paged at [`WIRE_PAGE_MAX`] in `room` order. Default:
    /// empty (a transport with no room enumeration).
    async fn transport_rooms(
        &self,
        _transport: TransportId,
        _after: Option<String>,
    ) -> WirePage<RoomInfo> {
        WirePage::default()
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

    /// Disconnect a transport instance (wire v30): stop its serve loop and mark it `Offline`,
    /// KEEPING its credential/config/bound_profile so a later reconnect resumes it — the reversible
    /// `purple_account_disconnect` analogue. Default: unsupported.
    async fn transport_disconnect(&self, _transport: TransportId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("transport_disconnect".into()))
    }

    /// Reconnect a transport instance (wire v35): the reversible counterpart of
    /// [`transport_disconnect`](Self::transport_disconnect) — re-spawn the owning adapter family's
    /// supervised serve loop and clear any fatal-disconnect marker, so a disconnected (or
    /// fatally-dropped) account resumes without a node restart. Errors if no adapter owns the
    /// transport. Idempotent: a no-op (returns `Ok`) when the family's serve loop is already
    /// running, and — honoring the persisted desired state — when every instance of the family is
    /// disabled. The serve loop emits its own serve-start `TransportChanged`, so this does not
    /// double-emit. Default: unsupported.
    async fn transport_connect(&self, _transport: TransportId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("transport_connect".into()))
    }

    /// Persist a transport instance's desired enabled/disabled state (wire v35). `enabled = false`
    /// disconnects it now (via [`transport_disconnect`](Self::transport_disconnect)) AND persists
    /// the desire so it is skipped at boot/spawn; `enabled = true` persists the desire and attempts
    /// to (re)connect (via [`transport_connect`](Self::transport_connect)). Because the serve loop
    /// is per-adapter-FAMILY (the coarsest granularity), a family serves unless EVERY one of its
    /// instances is disabled; the per-instance desire is always surfaced on
    /// [`TransportInstanceInfo::enabled`] regardless. Default: unsupported.
    async fn transport_set_enabled(
        &self,
        _transport: TransportId,
        _enabled: bool,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("transport_set_enabled".into()))
    }

    /// Set (or clear, with `None`) a transport instance's human label/rename (wire v35). Persisted
    /// by the node and overlaid onto [`TransportInstanceInfo::label`] in
    /// [`transport_instances`](Self::transport_instances). Default: unsupported.
    async fn transport_set_label(
        &self,
        _transport: TransportId,
        _label: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("transport_set_label".into()))
    }

    /// Remove a transport instance (wire v30): disconnect it, then perform the single node-owned
    /// teardown — close its conversations, unbind its routing pins, and drop its credential +
    /// config (the `purple_account_delete` analogue). The client issues one intent; the node
    /// sequences the steps. Default: unsupported.
    async fn transport_remove(&self, _transport: TransportId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("transport_remove".into()))
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

    /// List the conversations a transport owns (`SupportsConversations::list`), paged at
    /// [`WIRE_PAGE_MAX`] in conversation-id order. Default: empty.
    async fn conv_list(
        &self,
        _transport: TransportId,
        _after: Option<String>,
    ) -> WirePage<ConversationInfo> {
        WirePage::default()
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
    async fn conv_send(&self, _args: ConvSendArgs) -> Result<(), ApiError> {
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
    async fn conv_history(&self, _args: ConvHistoryArgs) -> JournalPageView {
        JournalPageView::default()
    }

    /// Invite/add a participant to a conversation (`SupportsMembership::invite`). Default: unsupported.
    async fn member_invite(&self, _args: MemberInviteArgs) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_invite".into()))
    }

    /// Remove/kick a participant. Default: unsupported.
    async fn member_remove(&self, _args: MemberRemoveArgs) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_remove".into()))
    }

    /// Ban a participant. Default: unsupported.
    async fn member_ban(&self, _args: MemberBanArgs) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_ban".into()))
    }

    /// Set a participant's role/affiliation. Default: unsupported.
    async fn member_set_role(&self, _args: MemberSetRoleArgs) -> Result<(), ApiError> {
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

    /// List a transport's server-side contact roster (`SupportsRoster::list`), paged at
    /// [`WIRE_PAGE_MAX`] in contact-id order (mirrors [`ControlApi::conv_list`]; the adapter
    /// returns the unbounded roster, the host sorts + pages it once). Default: empty.
    async fn roster_list(
        &self,
        _transport: TransportId,
        _after: Option<String>,
    ) -> WirePage<ContactInfo> {
        WirePage::default()
    }

    /// Add a contact to a transport's server-side roster (`SupportsRoster::add`). Default:
    /// unsupported.
    async fn roster_add(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("roster_add".into()))
    }

    /// Update a contact already on a transport's server-side roster (`SupportsRoster::update`).
    /// Default: unsupported.
    async fn roster_update(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("roster_update".into()))
    }

    /// Remove a contact from a transport's server-side roster (`SupportsRoster::remove`). Default:
    /// unsupported.
    async fn roster_remove(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("roster_remove".into()))
    }

    /// The node's live notification list (wire v37), newest first — the node-authoritative
    /// [`NotificationInfo`] collection a client renders and re-lists on a
    /// [`NodeEvent::NotificationsChanged`] pointer (ported from libpurple's `PurpleNotificationManager`).
    /// Default: empty (a node assembled without a notification manager).
    async fn notification_list(&self) -> Vec<NotificationInfo> {
        Vec::new()
    }

    /// Send a file out over a transport (`SupportsFileTransfer::send`; wire v37). Default:
    /// unsupported.
    async fn ft_send(
        &self,
        _transport: TransportId,
        _transfer: FileTransfer,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("ft_send".into()))
    }

    /// Receive a file over a transport (`SupportsFileTransfer::receive`; wire v37). Default:
    /// unsupported.
    async fn ft_receive(
        &self,
        _transport: TransportId,
        _transfer: FileTransfer,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("ft_receive".into()))
    }

    /// The node's person/metacontact registry (wire v37), insertion order — the
    /// node-authoritative [`Person`] collection a client renders and re-lists on a
    /// [`NodeEvent::PersonsChanged`] pointer (ported from the person half of libpurple's
    /// `PurpleContactManager`). Default: empty (a node assembled without a person registry).
    async fn person_list(&self) -> Vec<Person> {
        Vec::new()
    }

    // -- Foreign-agent discovery + registry (catalog-style; the daemon probes its own PATH) --

    /// Trigger a server-side foreign-agent discovery scan (PATH + well-known locations + the
    /// curated known-agent recipe table + configured endpoints), confirming each ACP candidate via
    /// the `initialize` handshake (stream-json entries are PATH-probed only). Operator-triggered
    /// (spawns subprocesses), like `model_search`. Default: empty.
    async fn agent_discover(&self) -> Vec<AgentEntry> {
        Vec::new()
    }

    /// The last discovery results plus any manually-registered recipes (the persisted catalog a
    /// GUI renders). Default: empty.
    async fn agent_catalog(&self) -> Vec<AgentEntry> {
        Vec::new()
    }

    /// Manually register (persist) a foreign-agent launch recipe — for a local path auto-detect
    /// missed or a remote endpoint. Default: unsupported.
    async fn agent_register(&self, _entry: AgentEntry) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("agent_register".into()))
    }

    /// Remove a registered/cataloged foreign agent by name. Default: unsupported.
    async fn agent_remove(&self, _name: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("agent_remove".into()))
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

    /// Persist a node-wide tool enable/disable override (wire v30). The override overlays the bound
    /// inventory served by [`Self::tool_list`] AND is consulted by per-session tool wiring, so a
    /// disabled tool disappears from new turns. Force-disable is always honored; a force-enable can
    /// never conjure a tool missing its build feature (it stays disabled with its `requires`
    /// string). Default: unsupported.
    async fn tool_set_enabled(&self, _tool: String, _enabled: bool) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("tool_set_enabled".into()))
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

    /// The node's read-only delegation guardrail caps (wire v29): the effective `orchestrate`
    /// depth/fanout ceilings, so a client can render them without probing. Default: zeros (a node
    /// that wired no orchestration caps).
    async fn caps(&self) -> CapsReport {
        CapsReport::default()
    }

    /// Read the node's runtime config (I13). Default: unsupported (config is env/TOML startup-only).
    async fn config_get(&self) -> Result<NodeConfigView, ApiError> {
        Err(ApiError::Unsupported("config_get".into()))
    }

    /// Write the node's runtime config (I13). Default: unsupported.
    async fn config_set(&self, _config: NodeConfigView) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("config_set".into()))
    }

    /// Read the node-owned OpenAI-compatible gateway's runtime status (enabled/addr/listening/
    /// last_error). The gateway is a node-managed resident service (also visible as the `"gateway"`
    /// [`ServiceHealth`] entry); this is its typed control view. Default: unsupported (a transport
    /// with no gateway seam wired).
    async fn gateway_get(&self) -> Result<GatewayStatus, ApiError> {
        Err(ApiError::Unsupported("gateway_get".into()))
    }

    /// Enable/disable the gateway and optionally rebind its listener. The new state is persisted to
    /// the durable store (config is the default/fallback) and the listener is hot-(re)bound, then
    /// the resulting [`GatewayStatus`] is returned. `addr = None` keeps the current/boot address.
    /// Default: unsupported.
    async fn gateway_set(
        &self,
        _enabled: bool,
        _addr: Option<String>,
    ) -> Result<GatewayStatus, ApiError> {
        Err(ApiError::Unsupported("gateway_set".into()))
    }

    /// Submit user feedback (N1; wire v31) — thumbs up/down + optional comment on an agent
    /// response, or general app feedback. The node validates the submission server-side and, on
    /// success, persists it to the durable feedback outbox (the acknowledgement means
    /// accepted+queued, never delivered). Explicit feedback is per-event consent: it is queued even
    /// when the global telemetry toggle is off. Default: unsupported (a transport with no store).
    async fn feedback_submit(&self, _args: FeedbackSubmitArgs) -> Result<FeedbackAck, ApiError> {
        Err(ApiError::Unsupported("feedback_submit".into()))
    }

    /// Read the node-owned global telemetry consent toggle (N1; wire v31). Default OFF (opt-in).
    /// Default: unsupported (a transport with no store).
    async fn telemetry_consent_get(&self) -> Result<bool, ApiError> {
        Err(ApiError::Unsupported("telemetry_consent_get".into()))
    }

    /// Set the node-owned global telemetry consent toggle (N1; wire v31); returns the new state.
    /// Default: unsupported (a transport with no store).
    async fn telemetry_consent_set(&self, _enabled: bool) -> Result<bool, ApiError> {
        Err(ApiError::Unsupported("telemetry_consent_set".into()))
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

    // -- Saved presences (W2-F; wire v37): the node-authoritative list of named, reusable
    //    presences the app renders + drives. Backed by the host `PresenceManager` over the durable
    //    store; a node built without it inherits the defaulted empty list / `Unsupported`. --

    /// List every saved presence (wire v37), in the manager's insertion order. Default: empty.
    async fn presence_list(&self) -> Vec<SavedPresence> {
        Vec::new()
    }

    /// Create or update a saved presence (wire v37): mints an id when unset, else replaces the
    /// existing presence by id. Default: unsupported.
    async fn presence_save(&self, _presence: SavedPresence) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("presence_save".into()))
    }

    /// Delete a saved presence by id (wire v37; idempotent). Default: unsupported.
    async fn presence_delete(&self, _id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("presence_delete".into()))
    }

    /// Set the active saved presence by id (wire v37), bumping its use-count + last-used.
    /// Default: unsupported.
    async fn presence_set_active(&self, _id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("presence_set_active".into()))
    }

    // -- Filesystem / workspace surface (daemon-fs-surface-spec.md). Grouped (defaulted) here, not a
    //    new sub-trait, so every NodeApi implementor inherits the surface; a node with a workspace
    //    binds the real impl (backed by daemon-host's WorkspaceFs). --

    /// The browsable roots this node exposes: host browse roots (home + operator allowlist) +
    /// the workspace root + any opened session sandboxes. Default: empty (no workspace surface).
    async fn fs_roots(&self) -> Vec<FsRoot> {
        Vec::new()
    }

    /// One directory's children (root-relative `dir`, "" = the root), paged at [`WIRE_PAGE_MAX`]
    /// entries per response: `after` resumes past the previous page's `next` cursor. Ignored
    /// entries are *marked* (`FsEntry.ignored`), not hidden, when `show_ignored` is false the
    /// caller may still hide them. Default: unsupported.
    async fn fs_list(
        &self,
        _root: FsRootId,
        _dir: String,
        _show_ignored: bool,
        _after: Option<String>,
    ) -> Result<FsListPage, ApiError> {
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
    async fn fs_write(&self, _args: FsWriteArgs) -> Result<FsRevision, ApiError> {
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
    async fn fs_watch_after(&self, _args: FsWatchAfterArgs) -> Result<FsWatchPageView, ApiError> {
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
    async fn fs_write_from_blob(&self, _args: FsWriteFromBlobArgs) -> Result<FsRevision, ApiError> {
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

    /// Step 2 — list a repo's loadable files for `engine` (the set a client selects to download),
    /// paged at [`WIRE_PAGE_MAX`] in `path` order.
    async fn model_files(
        &self,
        _repo: String,
        _revision: Option<String>,
        _engine: ModelEngine,
        _after: Option<String>,
    ) -> Result<WirePage<ModelFile>, ApiError> {
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
        _args: ModelRecommendArgs,
    ) -> Result<QuantRecommendation, ApiError> {
        Err(ApiError::Unsupported("model_recommend".into()))
    }

    /// Start an offline quantization of a repo's GGUF to `target_quant` (e.g. `Q4_K_M`); returns the
    /// job handle. `source_file` selects the source GGUF (`None` = the highest-precision one).
    async fn model_quantize(&self, _args: ModelQuantizeArgs) -> Result<QuantizeId, ApiError> {
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
    /// `claude-opus-4-8`) merged with locally-installed models. Paged at [`WIRE_PAGE_MAX`] in
    /// descriptor-id order. Default: the built-in cloud catalog.
    async fn models(&self, after: Option<String>) -> WirePage<ModelDescriptor> {
        let mut catalog = ModelDescriptor::builtin_cloud_catalog();
        catalog.sort_by(|a, b| a.id.cmp(&b.id));
        paginate(catalog, after.as_deref(), WIRE_PAGE_MAX, |m| m.id.clone())
    }

    /// The model a profile currently resolves to (`None` profile = the active default). `None` when
    /// no profile/model is resolvable. Default: `None`.
    async fn model_current(
        &self,
        _profile: Option<String>,
    ) -> Result<Option<ModelDescriptor>, ApiError> {
        Ok(None)
    }

    /// The discoverable provider catalog the setup picker renders: local engines + every genai cloud
    /// vendor + Daemon Cloud. Independent of the launch default, so an unconfigured node still lists
    /// providers. Default: empty (a transport with no discovery seam wired).
    async fn provider_catalog(&self) -> Vec<ProviderDescriptor> {
        Vec::new()
    }

    /// One provider's discoverable models. Credential-aware for genai vendors (authenticate the LIST
    /// call with the `transient_key`, else the stored `credential_ref`); Daemon Cloud lists keyless;
    /// local providers return the installed models. Paged at [`WIRE_PAGE_MAX`] in descriptor-id
    /// order. Default: empty.
    async fn provider_models(
        &self,
        _provider: String,
        _credential_ref: Option<String>,
        _transient_key: Option<String>,
        _after: Option<String>,
    ) -> WirePage<ModelDescriptor> {
        WirePage::default()
    }

    /// The persisted user-defined custom OpenAI-compatible providers (the editor's read-your-writes
    /// view — the raw [`CustomProvider`] write model, distinct from the merged `provider_catalog`
    /// read). Default: empty (no custom-provider store wired).
    async fn custom_provider_list(&self) -> Vec<CustomProvider> {
        Vec::new()
    }

    /// Create/update a user-defined custom provider (keyed by [`CustomProvider::id`]); the node
    /// forces `source = User` and re-validates `base_url`/`wire_selector`. It then appears as a
    /// normal `provider_catalog` row. Default: unsupported (no store wired).
    async fn custom_provider_set(&self, _provider: CustomProvider) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("custom_provider_set".into()))
    }

    /// Remove a user-defined custom provider by id (idempotent; config-seeded entries are not
    /// user-removable). Default: unsupported (no store wired).
    async fn custom_provider_remove(&self, _id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("custom_provider_remove".into()))
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

    /// The revision history of a profile (oldest first), paged at [`WIRE_PAGE_MAX`]; the cursor is
    /// the stringified revision `seq`.
    async fn profile_history(
        &self,
        _id: String,
        _after: Option<String>,
    ) -> Result<WirePage<Revision>, ApiError> {
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

    /// Read a profile's persona (SOUL.md) text (wire v36). This is the persona SOURCE text only —
    /// the composed system prompt is never wire-visible. Unknown ids fail with the profile-op
    /// not-found error rather than seeding a persona doc. Default: unsupported.
    async fn soul_get(&self, _id: String) -> Result<String, ApiError> {
        Err(ApiError::Unsupported("soul_get".into()))
    }

    /// Replace a profile's persona (SOUL.md) text (wire v36). The node validates/scans/caps and
    /// revision-logs the write; rejected typed for a Foreign-engine profile (its agent owns its
    /// own prompt — there is no persona to set). Default: unsupported.
    async fn soul_set(&self, _id: String, _text: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("soul_set".into()))
    }

    /// The revision history of a skill (oldest first), paged at [`WIRE_PAGE_MAX`]; the cursor is
    /// the stringified revision `seq`.
    async fn skill_history(
        &self,
        _name: String,
        _after: Option<String>,
    ) -> Result<WirePage<Revision>, ApiError> {
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

    /// Set (or clear, with `None`) a credential/account's human label/rename (wire v35). Persisted
    /// by the node and overlaid onto [`CredentialInfo::label`] in
    /// [`credential_list`](Self::credential_list) — this backs the app's AccountsPage rename.
    /// Default: unsupported.
    async fn credential_set_label(
        &self,
        _profile: String,
        _label: Option<String>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("credential_set_label".into()))
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

/// The kind of interactive auth flow (informs the client how to render + drive it). A flow is a
/// challenge/response state machine (see [`AuthChallenge`] / [`AuthStepInput`]); the kind is a hint
/// for capability discovery + the first challenge the client should expect.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthFlowKind {
    /// Matrix SSO: a browser-redirect flow whose redirect carries a single-use `loginToken`.
    MatrixSso,
    /// OAuth2 / OIDC authorization-code + PKCE: a browser-redirect flow whose redirect carries
    /// `code` + `state`.
    OAuth2Pkce,
    /// A bot/service token pasted into a form (no browser hop) — e.g. a Discord/Telegram bot token.
    BotToken,
    /// A user access token pasted into a form (no browser hop) — e.g. a personal API token.
    UserToken,
    /// A phone-number + one-time-code exchange (a `Form` phone prompt, then a `Form` OTP prompt).
    PhoneOtp,
    /// A QR-code device-link/pairing flow: the client renders a `Qr` challenge and polls until the
    /// other device approves (e.g. WhatsApp/Signal linked-device pairing).
    QrPairing,
    /// A username + password exchanged at sign-in (wire vNEXT). A masked [`AuthChallenge::Form`]
    /// collects a `username` ([`AuthFieldKind::Text`]) + `password` ([`AuthFieldKind::Password`]);
    /// the factory validates them and **exchanges** them for an opaque session token/blob, which
    /// [`crate::CredentialApi`]'s store persists as the RESULT. The password is transient — it drives
    /// the exchange and is never itself stored. (Contrast [`BotToken`](Self::BotToken) /
    /// [`UserToken`](Self::UserToken), where the pasted secret IS the stored credential.)
    UserPassword,
}

/// A single challenge a flow presents to the client at some step — how the client should collect the
/// next [`AuthStepInput`]. Modeled on libpurple's `PurpleRequest` request-fields (redirect / form /
/// QR / message prompts). Serialized externally-tagged (`{ "Redirect": { .. } }`, ...).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthChallenge {
    /// Open `authorization_url` in a browser and capture the redirect; reply with
    /// [`AuthStepInput::Callback`] carrying the captured URL/query (the SSO / OAuth2 hop).
    Redirect {
        /// The URL the client opens in a browser.
        authorization_url: String,
    },
    /// Collect the named fields from the user (phone number, OTP, bot/user token, homeserver, …) and
    /// reply with [`AuthStepInput::Fields`]. `fields` reuses the discovery [`AuthParamField`] shape.
    Form {
        /// A human title for the form (e.g. "Enter the code we texted you").
        title: String,
        /// The fields to collect (keyed by [`AuthParamField::key`]).
        fields: Vec<AuthParamField>,
    },
    /// Render `payload` as a QR code (and/or display the pre-rendered `image` bytes) and poll with
    /// [`AuthStepInput::Poll`] every `poll_interval_ms` until the flow completes (device pairing).
    Qr {
        /// The QR payload the peer device scans (e.g. a device-link URI).
        payload: String,
        /// An optional pre-rendered QR image (raw bytes, e.g. PNG); `None` = the client renders
        /// `payload` itself. Inline bytes rather than a content-addressed [`Image`] because the QR is
        /// ephemeral per-flow and the client renders it immediately (no blob store round-trip).
        #[serde(with = "serde_bytes")]
        image: Option<Vec<u8>>,
        /// How often the client should re-poll with [`AuthStepInput::Poll`] (milliseconds).
        poll_interval_ms: u64,
    },
    /// A purely informational message to display (e.g. "Approve the login on your other device");
    /// the client typically follows it with a [`AuthStepInput::Poll`] or a terminal state.
    Message {
        /// The message text to show the user.
        text: String,
    },
}

/// The client's response to an [`AuthChallenge`] — the input driving one step of the flow.
/// Serialized externally-tagged (`{ "Fields": { .. } }`, `{ "Callback": ".." }`, `"Poll"`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthStepInput {
    /// The filled `key -> value` pairs answering a [`AuthChallenge::Form`] (keyed by
    /// [`AuthParamField::key`]).
    Fields(BTreeMap<String, String>),
    /// The captured redirect (full URL or its query string) answering a [`AuthChallenge::Redirect`].
    Callback(String),
    /// A no-payload poll answering a [`AuthChallenge::Qr`] / [`AuthChallenge::Message`] — "has the
    /// pairing/approval landed yet?".
    Poll,
}

/// The result of advancing a flow one step: either the next challenge to present, or the completed
/// outcome. Serialized externally-tagged (`{ "Challenge": .. }` / `{ "Completed": .. }`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthStepResult {
    /// The flow needs more input: present this challenge and call `auth_step` again.
    Challenge(AuthChallenge),
    /// The flow finished: the credential was persisted (and any bind honored), described by the
    /// same [`AuthCompleteResponse`] the single-step `auth_complete` returns.
    Completed(AuthCompleteResponse),
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

/// The parked-flow handle returned by `auth_begin`: the flow id + its initial [`AuthChallenge`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthBeginResponse {
    /// The single-use flow id to pass to `auth_step` / `auth_complete` / `auth_cancel`.
    pub flow_id: String,
    /// The first challenge to present (a redirect URL to open, a form to fill, a QR to render, …).
    pub challenge: AuthChallenge,
    /// Flow TTL (unix seconds); the flow is evicted after this.
    pub expires_at: u64,
}

/// Advance a flow one step: feed the [`AuthStepInput`] the client collected for the current
/// [`AuthChallenge`]. The reply is an [`AuthStepResult`] (the next challenge, or completion).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthStepRequest {
    /// The flow id from `auth_begin`.
    pub flow_id: String,
    /// The client's response to the current challenge.
    pub input: AuthStepInput,
}

/// Finish a flow from the captured redirect (the single-step compatibility shape over `auth_step`).
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

/// How a client should render + validate an [`AuthParamField`] (wire vNEXT). Defaults to [`Text`],
/// a plain single-line entry, so a pre-vNEXT peer that omits `kind` keeps today's behavior.
///
/// [`Text`]: AuthFieldKind::Text
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthFieldKind {
    /// A plain single-line text entry (the default) — e.g. a username, homeserver, or handle.
    #[default]
    Text,
    /// A secret entry (token/password): the client MUST mask input and must not echo or persist it.
    Password,
    /// A numeric entry (e.g. a one-time code); the client may render a numeric keypad.
    Number,
    /// A single choice from [`AuthParamField::choices`]; the client renders a picker.
    Choice,
}

/// One field of a family's `params` form (capability discovery).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthParamField {
    /// The `params` key.
    pub key: String,
    /// A human label for the field.
    pub label: String,
    /// Whether the field is required.
    pub required: bool,
    /// How the client should render + validate this field (wire vNEXT). Defaults to
    /// [`AuthFieldKind::Text`]; a secret (token/password) uses [`AuthFieldKind::Password`] so the
    /// client masks it.
    #[serde(default)]
    pub kind: AuthFieldKind,
    /// A prefill/default value to seed the input with (wire vNEXT); `None` = start empty.
    #[serde(default)]
    pub default: Option<String>,
    /// Placeholder/hint text shown in an empty input (wire vNEXT); `None` = no hint.
    #[serde(default)]
    pub placeholder: Option<String>,
    /// The allowed values when `kind == `[`AuthFieldKind::Choice`] (wire vNEXT); empty for every
    /// other kind.
    #[serde(default)]
    pub choices: Vec<String>,
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
    /// Begin a flow: park the continuation and return its id + initial [`AuthChallenge`].
    async fn auth_begin(&self, _req: AuthBeginRequest) -> Result<AuthBeginResponse, ApiError> {
        Err(ApiError::Unsupported("auth_begin".into()))
    }

    /// Advance a parked flow one step with the client's [`AuthStepInput`]. On completion the node
    /// persists the credential (and honors any bind) and returns the [`AuthCompleteResponse`] inside
    /// [`AuthStepResult::Completed`]; otherwise it returns the next [`AuthChallenge`].
    async fn auth_step(&self, _req: AuthStepRequest) -> Result<AuthStepResult, ApiError> {
        Err(ApiError::Unsupported("auth_step".into()))
    }

    /// Finish a single-redirect flow from the captured callback — a thin compatibility wrapper that
    /// drives [`auth_step`](AuthApi::auth_step) with an [`AuthStepInput::Callback`] and expects the
    /// flow to complete in one step. A flow that needs further interactive steps errors here (use
    /// `auth_step`). Kept so existing single-step callers keep working over one code path.
    async fn auth_complete(
        &self,
        req: AuthCompleteRequest,
    ) -> Result<AuthCompleteResponse, ApiError> {
        match self
            .auth_step(AuthStepRequest {
                flow_id: req.flow_id,
                input: AuthStepInput::Callback(req.callback),
            })
            .await?
        {
            AuthStepResult::Completed(resp) => Ok(resp),
            AuthStepResult::Challenge(_) => Err(ApiError::Unsupported(
                "auth_complete: flow needs interactive steps beyond a single callback; use auth_step"
                    .into(),
            )),
        }
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

/// The admin access-control surface (Auth 5): user/role/session administration over the node's
/// identity store. Every op except [`who_am_i`](AccessControlApi::who_am_i) requires the
/// `access_admin` capability (enforced by the request-context gate and re-checked in the host
/// implementation); `who_am_i` is allowed for any authenticated principal. The reserved
/// `resource_grant_*` ops return [`ApiError::Unsupported`] until per-resource grants are enforced.
///
/// Default methods resolve to [`ApiError::Unsupported`] / empty so a node assembled without an
/// identity store (the FFI / conformance harness) still satisfies [`NodeApi`].
#[async_trait]
pub trait AccessControlApi: Send + Sync {
    /// Create a user with an initial password + role set; returns the created record (no secrets).
    async fn user_create(
        &self,
        _username: String,
        _password: String,
        _roles: Vec<String>,
    ) -> Result<AccessUser, ApiError> {
        Err(ApiError::Unsupported("access control not available".into()))
    }

    /// List all users (with resolved roles).
    async fn user_list(&self) -> Result<Vec<AccessUser>, ApiError> {
        Err(ApiError::Unsupported("access control not available".into()))
    }

    /// Enable/disable an account (disable also revokes the user's sessions).
    async fn user_disable(&self, _user_id: String, _disabled: bool) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("access control not available".into()))
    }

    /// Replace a user's role set.
    async fn user_set_roles(&self, _user_id: String, _roles: Vec<String>) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("access control not available".into()))
    }

    /// Replace a user's password (re-derives SCRAM material and revokes the user's sessions).
    async fn user_set_password(&self, _user_id: String, _password: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("access control not available".into()))
    }

    /// The built-in roles and their effective capabilities (for an admin UI's role→cap matrix).
    async fn role_list(&self) -> Result<Vec<RoleInfo>, ApiError> {
        Err(ApiError::Unsupported("access control not available".into()))
    }

    /// The caller's own [`PrincipalView`] (any authenticated principal; no `access_admin` needed).
    async fn who_am_i(&self) -> Result<PrincipalView, ApiError> {
        Err(ApiError::Unsupported("access control not available".into()))
    }

    /// Revoke **all** session tokens for a user.
    async fn session_revoke(&self, _user_id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("access control not available".into()))
    }

    /// Reserved (option B): grant one capability over one resource to one user. Returns
    /// [`ApiError::Unsupported`] until per-resource grants are enforced.
    async fn resource_grant_create(
        &self,
        _user_id: String,
        _resource_kind: String,
        _resource_id: String,
        _capability: String,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("resource grants are reserved".into()))
    }

    /// Reserved: list per-resource grants. Returns [`ApiError::Unsupported`].
    async fn resource_grant_list(&self, _user_id: Option<String>) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("resource grants are reserved".into()))
    }

    /// Reserved: revoke a per-resource grant. Returns [`ApiError::Unsupported`].
    async fn resource_grant_revoke(&self, _id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("resource grants are reserved".into()))
    }
}

/// The whole node surface: the session, control, model-management, profile/config, credential,
/// interactive-auth, and access-control sub-surfaces.
pub trait NodeApi:
    SessionApi + ControlApi + ModelApi + ProfileApi + CredentialApi + AuthApi + AccessControlApi
{
}
impl<
        T: SessionApi
            + ControlApi
            + ModelApi
            + ProfileApi
            + CredentialApi
            + AuthApi
            + AccessControlApi,
    > NodeApi for T
{
}

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

/// The node-owned OpenAI-compatible gateway's runtime status (the reply to both
/// [`ControlApi::gateway_get`] and [`ControlApi::gateway_set`]). The gateway is a node-managed
/// resident service (also surfaced as a [`ServiceHealth`] entry named `"gateway"`); this is its
/// typed control view: whether it should be serving (`enabled`), the effective bind `addr`
/// (runtime override on top of boot config), whether the listener is actually bound (`listening`),
/// and the last bind/serve error if any.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayStatus {
    /// Whether the gateway is configured to serve (the persisted enable state).
    pub enabled: bool,
    /// The effective bind address (`None` when no addr is configured yet).
    pub addr: Option<String>,
    /// Whether the listener is currently bound and serving.
    pub listening: bool,
    /// The last bind/serve error, if the most recent (re)bind failed.
    pub last_error: Option<String>,
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

/// One choice a foreign agent's [`ModelSelector`] offers: the opaque value id the set intent
/// carries plus a human-readable label.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelChoice {
    /// The choice's value id — what a [`SessionApi::set_session_model`] carries (opaque to the node).
    pub id: String,
    /// The choice's human-readable label (the agent's display name for the model).
    pub label: String,
}

/// A foreign (ACP) session's advertised model selector, surfaced as live session state (wire v30,
/// Phase 3): the agent's own `Model`-category config option, captured by `daemon-acp` at
/// `session/new`, after a `set_config_option`, and on a `config_option_update` notification. Present
/// only for a resident foreign session whose agent advertises a Model selector — a native (`Core`)
/// session and a gateway-routed `NodeProvider` session (whose model is chosen node-side) have none.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSelector {
    /// The ACP config-option id of the Model selector (opaque; echoed back on a set intent).
    pub option_id: String,
    /// The currently-selected model value id (one of `choices`' ids).
    pub current: String,
    /// The models the agent offers, flattened across any groups.
    pub choices: Vec<ModelChoice>,
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
    /// Which execution engine this session runs on (wire v30): the native `Core` engine, or a
    /// `Foreign { agent }` backed by a catalog agent. Denormalized from the bound profile so a
    /// detail pane renders the backend without a second `profile_get`. Defaults to `Core`.
    #[serde(default)]
    pub engine: EngineSelector,
    /// For a `Foreign` engine: how the agent sources its model backend (its own, model-steered; or
    /// routed through the node gateway to a node provider). Denormalized from the bound profile;
    /// `AgentNative { model: None }` for `Core` sessions and pre-v30 encodings.
    #[serde(default)]
    pub foreign_backend: ForeignBackend,
    /// For a resident foreign (ACP) session whose agent advertises a `Model` selector: its live
    /// model choices + current selection (wire v30, Phase 3), so a thin client can render + drive a
    /// foreign model picker. `None` for native sessions, foreign agents with no Model selector, and
    /// gateway-routed `NodeProvider` sessions (whose model is chosen node-side).
    #[serde(default)]
    pub model_selector: Option<ModelSelector>,
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

/// A pure-local recap of a session's recent activity ([`ControlApi::session_recap`]) — the
/// hermes `build_recap` analogue, computed node-side from the session's conversation with **no LLM
/// call**: scope counts, the most-used tools, recently-touched files, and the last ask/reply.
///
/// The counts are totals over the whole conversation; `top_tools` / `files_touched` / `last_ask` /
/// `last_reply` are derived from the recent-activity window (the last 20 turns).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionRecap {
    /// The session's roster title, when one is set.
    pub title: Option<String>,
    /// Total user turns in the conversation.
    pub user_turns: u32,
    /// Total assistant turns (including tool-calling turns).
    pub assistant_turns: u32,
    /// Total tool results recorded.
    pub tool_results: u32,
    /// The most-used tools in the recent window as `(name, count)`, descending, at most 5.
    pub top_tools: Vec<(String, u32)>,
    /// Distinct file paths recently touched by tools (from `path`/`file_path` args), newest first,
    /// at most 5.
    pub files_touched: Vec<String>,
    /// The latest user prompt in the window, whitespace-collapsed and truncated (~140 chars).
    pub last_ask: Option<String>,
    /// The latest assistant text in the window, whitespace-collapsed and truncated (~200 chars).
    pub last_reply: Option<String>,
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
    /// The §12 exec-approval command fingerprint (wire v28): the lowercase-hex sha256 of the resolved
    /// `(abs-binary, argv, env-delta, cwd, exec-surface)` tuple the operator is approving, promoted
    /// from the free-text `prompt` to a structured field so a GUI can display/correlate it. `None`
    /// for non-command approvals (fs edits) and pre-v28 rows. Display/correlation only — the
    /// approve-then-swap enforcement stays snapshot-side in `daemon-core`.
    #[serde(default)]
    pub fingerprint: Option<String>,
    /// A node-computed structured detail for the gated call (wire v30) — an fs/edit approval
    /// attaches a unified diff as a [`ToolDetail`] with `kind == "fs.diff"` (JSON body
    /// `{ "path", "diff" }`) a rich client renders. `None` for approvals with no computable detail
    /// and pre-v30 rows.
    #[serde(default)]
    pub detail: Option<ToolDetail>,
}

/// One remembered exec-approval command fingerprint on a session's `allow_permanent` allow-list
/// (wire v29; [`ControlApi::fingerprint_list`] / [`ControlApi::fingerprint_revoke`]): the operator
/// answered an approval with "Allow permanently", so this exact resolved command auto-approves for
/// the rest of the session. Least-privilege management surface — list what is trusted, revoke one.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RememberedFingerprint {
    /// The lowercase-hex sha256 of the resolved `(exec-surface, abs-binary, argv, env-delta, cwd)`
    /// tuple — the same value [`ApprovalInfo::fingerprint`] displays at park time.
    pub fingerprint: String,
    /// An optional human-readable label for the remembered command (wire v30; populated from the
    /// engine's command summary at the `allow_permanent` decide path when known). `None` when only
    /// the hash was captured.
    #[serde(default)]
    pub label: Option<String>,
    /// Unix milliseconds when the operator remembered this command (wire v30; the `allow_permanent`
    /// decide path stamps it). `0` for pre-v30 rows that predate provenance.
    #[serde(default)]
    pub remembered_at_ms: u64,
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

/// Where a cataloged foreign agent's launch recipe came from.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentSource {
    /// From the curated builtin known-agent recipe table.
    Builtin,
    /// Manually registered by an operator (via [`ControlApi::agent_register`]).
    Manual,
    /// A network endpoint (TCP / stdio-bus / remote), not a PATH binary.
    Endpoint,
}

/// The wire protocol a cataloged foreign agent speaks — selects the adapter the node drives the
/// spawned agent with (wire v29; the registry mirror of the fleet spawner's `ForeignProtocol`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentProtocol {
    /// Agent Client Protocol: symmetric JSON-RPC 2.0 over stdio (the default; pre-v29 entries
    /// decode as ACP). Probed via the ACP `initialize` handshake.
    #[default]
    Acp,
    /// Claude-Code `stream-json`: NDJSON event envelope over the line transport (also Amp, Cursor).
    /// Probed installed-on-PATH only — there is no `initialize` handshake, so `version` and
    /// `capabilities` stay empty.
    StreamJson,
}

/// A launch recipe for a foreign agent — the catalog's mirror of the host's spawn spec. Either a
/// stdio subprocess (`program` + `args` + `env`) or a network `endpoint`; exactly one is meaningful
/// per `source`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentRecipe {
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

/// The node-derived trust status of a cataloged agent (wire v32) — computed once at catalog
/// assembly from `installed` / `protocol` / `version` so every client renders the same verdict
/// instead of re-deriving it. Build it only via [`AgentEntry::derive_verification`] (never hand-roll
/// the rule at a call site).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentVerification {
    /// Installed and confirmed via the ACP `initialize` handshake (protocol version reported).
    Verified,
    /// Installed but unconfirmed: a stream-json agent (no handshake), or an ACP agent whose
    /// `initialize` probe has not reported a protocol version.
    Unverified,
    /// No candidate binary/endpoint was found on PATH/well-known/endpoint probe (the serde default,
    /// so a pre-v32 encoding without the field decodes as `NotInstalled` until the node re-derives).
    #[default]
    NotInstalled,
}

/// One entry in the foreign-agent catalog ([`ControlApi::agent_catalog`] /
/// [`ControlApi::agent_discover`]): a known/registered agent, the protocol it speaks, whether it is
/// installed, and (for ACP) the `initialize`-verified metadata.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentEntry {
    /// The agent display name (catalog key, e.g. `"gemini"`, `"goose"`, `"claude-via-zed"`).
    pub name: String,
    /// How to launch it.
    pub recipe: AgentRecipe,
    /// Where the recipe came from.
    pub source: AgentSource,
    /// The wire protocol the agent speaks (wire v29; pre-v29 encodings decode as `Acp`).
    #[serde(default)]
    pub protocol: AgentProtocol,
    /// Whether a candidate binary/endpoint was found (PATH/well-known/endpoint probe).
    #[serde(default)]
    pub installed: bool,
    /// The ACP protocol version the agent reported at `initialize`, when probed (`None` for
    /// stream-json agents — they have no handshake).
    #[serde(default)]
    pub version: Option<String>,
    /// Agent capabilities advertised at `initialize` (opaque key/value), when probed (empty for
    /// stream-json agents).
    #[serde(default)]
    pub capabilities: Vec<(String, String)>,
    /// The node-derived trust status (wire v32): the single authoritative verdict a client renders
    /// verbatim. Always recomputed by the node from `installed`/`protocol`/`version` at catalog
    /// assembly (see [`AgentEntry::derive_verification`] / [`AgentEntry::refresh_verification`]); a
    /// caller-supplied value on a registration is not trusted.
    #[serde(default)]
    pub verification: AgentVerification,
}

impl AgentEntry {
    /// The single derivation rule for [`AgentVerification`] — the ONE place `installed` / `protocol`
    /// / `version` are folded into a verdict, so no client (or node) call site re-implements it:
    /// not installed ⇒ `NotInstalled`; an installed ACP agent that reported a version at
    /// `initialize` ⇒ `Verified`; anything else installed (stream-json, or ACP without a handshake
    /// version) ⇒ `Unverified`.
    pub fn derive_verification(&self) -> AgentVerification {
        if !self.installed {
            AgentVerification::NotInstalled
        } else if matches!(self.protocol, AgentProtocol::Acp) && self.version.is_some() {
            AgentVerification::Verified
        } else {
            AgentVerification::Unverified
        }
    }

    /// Recompute [`Self::verification`] from the current `installed`/`protocol`/`version`. Call this
    /// after any mutation of those fields (probe/enrich) and before serving/persisting an entry, so
    /// the wire status stays in lockstep with the raw fields the node derives it from.
    pub fn refresh_verification(&mut self) {
        self.verification = self.derive_verification();
    }
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

/// The node's read-only delegation guardrail caps ([`ControlApi::caps`], wire v29): the EFFECTIVE
/// ceilings the `orchestrate` tool enforces (config policy composed with the assembly recursion
/// budget), so a client can render "why was this spawn declined" next to the structured
/// `guardrail` tool detail without probing by hitting limits.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapsReport {
    /// The delegation-tree depth a `spawn` is declined past.
    pub orchestrate_max_depth: u32,
    /// The concurrent detached-children per parent a `spawn wait:false` is declined past.
    pub orchestrate_max_fanout: u32,
    /// The number of profiles an authoring session may compose (author via `profile_manage`) before
    /// a further `create` is declined (wire v31): the agent-created-agents guardrail, counted over
    /// the session's own `agent/{session}/` profile namespace.
    #[serde(default)]
    pub max_composed_profiles: u32,
    /// The concurrent inline/ephemeral children per session a `spawn` is declined past (wire v31):
    /// bounds a session's live transient-subagent fan, distinct from the persistent detached-fanout
    /// cap.
    #[serde(default)]
    pub max_ephemeral_per_session: u32,
}

/// A node tool-inventory entry ([`ControlApi::tool_list`]; enriched in wire v29 so a client can
/// render "why is this tool unavailable"). One row per registered tool, plus one row per
/// config-gated optional surface that did NOT materialize (`enabled: false` + `requires`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInfo {
    /// The tool name (as used in `ProfileSpec.tool_allowlist`), or the subsystem name for a
    /// disabled dynamic surface whose tool names are only known once it is enabled (e.g. `python`,
    /// `mcp`).
    pub name: String,
    /// A short human description, when known.
    #[serde(default)]
    pub description: Option<String>,
    /// Whether the tool is registered and usable on this node right now (wire v29). Profile
    /// allowlists can still narrow a session's view — this is the node-wide availability.
    pub enabled: bool,
    /// Why a disabled tool is unavailable (wire v29): the missing config key / credential /
    /// build feature (e.g. `[web].enable + a tavily credential`, `browser build feature`).
    /// `None` for enabled tools.
    #[serde(default)]
    pub requires: Option<String>,
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
    /// A disconnect/teardown is in flight (wire v30; transient — the account is on its way to
    /// `Offline`). Emitted per reconnect attempt so a client can render a "reconnecting…" state.
    Disconnecting,
    /// Failed (the adapter logs the specifics; this carries only the coarse state).
    Error,
}

/// Why a transport instance disconnected (wire v30) — a closed, Matrix-scoped set the node maps
/// every adapter-specific failure onto (the GError-shaped analogue). **The node decides
/// [`TransportInstanceInfo::fatal`]**; a thin client keys re-auth affordances off `fatal`/`reason`,
/// never off adapter strings, and never re-derives whether a disconnect is terminal.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisconnectReason {
    /// The operator/user asked to disconnect (`TransportDisconnect`, sign-out).
    UserRequested,
    /// A network/transport error (timeout, connection reset, homeserver unreachable).
    NetworkError,
    /// The stored credential was rejected (login/refresh failed).
    AuthenticationFailed,
    /// The server replaced this session with another client (soft-logout / device conflict).
    ReplacedByOtherClient,
    /// The instance config is invalid (bad homeserver URL, missing required field).
    InvalidSettings,
    /// A TLS/certificate validation failure.
    CertificateError,
    /// Any reason not captured above (the adapter's detail rides `message`).
    Other,
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

/// One display-oriented adapter policy row (wire v30; [`AdapterInfo::policies`]) — a node-labeled,
/// node-valued setting a client renders read-only (the [`AuthParamField::label`] precedent: the
/// node owns the label, the client never keys behavior off `key`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyEntry {
    /// The stable policy key (e.g. `"auto_accept_invites"`).
    pub key: String,
    /// The node-decided human label.
    pub label: String,
    /// The current value, rendered as text (e.g. `"true"`).
    pub value: String,
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
    /// Display-oriented adapter policies the client renders read-only (wire v30). Matrix reports
    /// `auto_accept_invites`; the node decides the labels. Empty for adapters with no reportable
    /// policy.
    #[serde(default)]
    pub policies: Vec<PolicyEntry>,
    /// Per-verb conversation-management capabilities (wire v33; ← libpurple's `implements_*` probes,
    /// here reported once by the node from [`SupportsConversations::supported`]). `None` = this
    /// adapter does not implement the conversation feature trait at all; `Some(ops)` = implemented,
    /// with a bool per verb (create / join / leave / delete / …). A thin client capability-gates the
    /// room affordances off these flags instead of switching on `family`.
    #[serde(default)]
    pub conversation_ops: Option<ConversationOps>,
    /// Per-verb membership-administration capabilities (wire v33; from
    /// [`SupportsMembership::supported`]). `None` = the adapter does not implement the membership
    /// feature trait; `Some(ops)` = implemented, with a bool per verb (invite / remove / ban /
    /// set_role) gating the member-row buttons.
    #[serde(default)]
    pub membership_ops: Option<MembershipOps>,
    /// Per-verb remote-contact capabilities (wire v33; from [`SupportsContacts::supported`]).
    /// `None` = the adapter does not implement the contacts feature trait; `Some(ops)` =
    /// implemented, with a bool per verb (get_profile / action_menu / set_alias).
    #[serde(default)]
    pub contacts_ops: Option<ContactsOps>,
    /// Per-verb server-side roster (contact-list) capabilities (wire v33; from
    /// [`SupportsRoster::supported`]). `None` = the adapter does not implement the roster feature
    /// trait; `Some(ops)` = implemented, with a bool per verb (add / update / remove).
    #[serde(default)]
    pub roster_ops: Option<RosterOps>,
    /// Whether the adapter exposes a contact/user directory search (wire v33; ←
    /// `purpleprotocoldirectory.h` / the libpurple roomlist successor). Unlike the other feature
    /// traits there is no per-verb ops struct — search is the sole verb — so presence + the
    /// adapter's own [`SupportsDirectory::supported`] probe collapse to this single flag. `false`
    /// when the trait is absent or the adapter reports it unsupported.
    #[serde(default)]
    pub directory: bool,
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
    /// Why the instance is offline/disconnected, when known (wire v30). `None` while connected.
    #[serde(default)]
    pub reason: Option<DisconnectReason>,
    /// A human-readable detail for the disconnect (the adapter's error text). `None` when none.
    #[serde(default)]
    pub message: Option<String>,
    /// Node-decided: whether the disconnect is fatal (stop retrying; offer re-auth). Thin clients
    /// MUST NOT re-derive this — the node owns the reconnect/backoff policy. `false` for transient
    /// reasons the node will retry, and while connected.
    #[serde(default)]
    pub fatal: bool,
    /// The operator's DESIRED enabled state (wire v35), overlaid by the node from its durable
    /// store. `false` = disabled: disconnected now and skipped at boot/spawn. Surfaced per-instance
    /// regardless of the family's live serve state (the serve loop is per-family, but the desire is
    /// per-instance). Default `true` (an instance with no stored preference).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// An operator-set human label/rename for this instance (wire v35), overlaid by the node from
    /// its durable store. `None` = no custom label (the client falls back to `display_name`).
    #[serde(default)]
    pub label: Option<String>,
}

/// The node-owned lifecycle sink an events-IO adapter reports coarse lifecycle signals through
/// (wire v30). The single seam that keeps `NodeApi` clean: the assembling binary hands the adapter
/// an `Arc<dyn LifecycleSink>` (the node itself), and the adapter calls it when it observes a
/// disconnect cause or a conversation/membership change. The node owns the consequences: it maps a
/// reported `reason` onto `fatal` and pushes `TransportChanged` (item 2), and on an `is_self`
/// removal it reconciles its own routing (drops the dangling `ChatRoute` pin) before pushing the
/// membership event (item 3). Adapters that never observe these simply never call it.
#[async_trait]
pub trait LifecycleSink: Send + Sync {
    /// Report that `transport` disconnected with a coarse `reason` (+ optional detail `message`).
    /// The NODE decides `fatal` from the reason and pushes the resulting `TransportChanged`; the
    /// adapter never re-derives whether the disconnect is terminal.
    async fn transport_disconnected(
        &self,
        transport: TransportId,
        reason: DisconnectReason,
        message: Option<String>,
    );

    /// Report that a conversation was added/removed on `transport` (coarse tier; retires client
    /// `ConvList` re-polling).
    async fn conversations_changed(&self, transport: TransportId, conv: String, change: ConvChange);

    /// Report a membership transition (granular tier). On a self `Left`/`Kicked`/`Banned` the node
    /// reconciles its routing (drops the now-dangling pin) before emitting.
    #[allow(clippy::too_many_arguments)]
    async fn membership_changed(
        &self,
        transport: TransportId,
        conv: String,
        member: String,
        change: MembershipChange,
        actor: Option<String>,
        reason: Option<String>,
        is_self: bool,
    );
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
    async fn send(&self, _args: ConvSendArgs) -> Result<(), ApiError> {
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
    async fn invite(&self, _args: MemberInviteArgs) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_invite".into()))
    }
    async fn remove(&self, _args: MemberRemoveArgs) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_remove".into()))
    }
    async fn ban(&self, _args: MemberBanArgs) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_ban".into()))
    }
    async fn set_role(&self, _args: MemberSetRoleArgs) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_set_role".into()))
    }
}

/// Per-verb probe for [`SupportsRoster`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterOps {
    /// Enumerate the account's server-side contact list (wire v34; ← `purple_protocol_roster_*`
    /// listing). Gates the client's Contacts section.
    pub list: bool,
    pub add: bool,
    pub update: bool,
    pub remove: bool,
}

/// Account-level server-side contact list (← `purpleprotocolroster.h`). Defined; no adapter yet.
#[async_trait]
pub trait SupportsRoster: Send + Sync {
    fn supported(&self) -> RosterOps;
    /// List the account's server-side contact roster (wire v34). Adapter-ordered + unbounded; the
    /// host sorts + pages it centrally (mirrors [`SupportsConversations::list`]). Default: empty.
    async fn list(&self, _transport: TransportId) -> Vec<ContactInfo> {
        Vec::new()
    }
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

/// File transfer (← `purpleprotocolfiletransfer.h`). The verbs carry `transport` (mirroring every
/// other feature trait, e.g. [`ConvSendArgs::transport`]) so an adapter can resolve its per-account
/// client; the [`FileTransfer`] stays a pure domain object.
#[async_trait]
pub trait SupportsFileTransfer: Send + Sync {
    /// The per-verb capability probe (← `implements_send`/`implements_receive`).
    fn supported(&self) -> FileTransferOps;
    /// Send a file out (← `send_async`/`send_finish`). Default: unsupported.
    async fn send(&self, _transport: TransportId, _transfer: FileTransfer) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("file_transfer_send".into()))
    }
    /// Receive a file (← `receive_async`/`receive_finish`). Default: unsupported.
    async fn receive(
        &self,
        _transport: TransportId,
        _transfer: FileTransfer,
    ) -> Result<(), ApiError> {
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

/// The direction of a [`FileTransfer`] (wire v37). Daemon-native framing of libpurple's
/// initiator-vs-remote asymmetry (`purple_file_transfer_new_send` sets `initiator = account`;
/// `purple_file_transfer_new_receive` sets `initiator = remote`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileTransferDirection {
    /// The node is sending a local file out (`new_send`).
    #[default]
    Send,
    /// The node is receiving a remote file (`new_receive`).
    Receive,
}

/// The state of a [`FileTransfer`] (← `PurpleFileTransferState`, `purplefiletransfer.h`; wire
/// v37). There is no explicit accepted/cancelled state in libpurple — cancellation is a
/// `GCancellable` + error, modeled here as [`FileTransferState::Failed`] plus an `error` message.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileTransferState {
    /// The transfer is in an unknown state (`*_STATE_UNKNOWN`, the default).
    #[default]
    Unknown,
    /// The transfer is still being negotiated (`*_STATE_NEGOTIATING`).
    Negotiating,
    /// The transfer is in progress (`*_STATE_STARTED`).
    Started,
    /// The transfer completed successfully (`*_STATE_FINISHED`).
    Finished,
    /// The transfer failed (`*_STATE_FAILED`); `error` should carry the reason.
    Failed,
}

/// A file transfer (← `PurpleFileTransfer`, `purplefiletransfer.c`). The `name` + content-addressed
/// `blob` are the original (Wave-1) wire shape; every field below is appended additively with a
/// serde default (wire v37), so pre-existing payloads decode unchanged. The behavior logic
/// (constructors, state machine, predicates) lives in [`crate::file_transfer`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileTransfer {
    /// The file name (← `filename`; for `new_send`, the local file's base name).
    pub name: String,
    /// The content-addressed blob. For a `Send` this is the local file's content; for a `Receive`
    /// it is the destination handle (a placeholder hash until the bytes are stored).
    pub blob: BlobRef,
    /// Send vs. receive (wire v37).
    #[serde(default)]
    pub direction: FileTransferDirection,
    /// The lifecycle state (wire v37).
    #[serde(default)]
    pub state: FileTransferState,
    /// The remote participant (← `remote`; wire v37).
    #[serde(default)]
    pub remote: Option<ContactInfo>,
    /// Who initiated the transfer (← `initiator`; wire v37).
    #[serde(default)]
    pub initiator: Option<ContactInfo>,
    /// The advertised file size in bytes (← `file-size`; kept independent of `blob.size`, as in C;
    /// wire v37).
    #[serde(default)]
    pub file_size: u64,
    /// Bytes transferred so far — the node-owned progress a thin client renders (wire v37).
    #[serde(default)]
    pub transferred: u64,
    /// The content/media type hint (← `content-type`; wire v37).
    #[serde(default)]
    pub content_type: Option<String>,
    /// An optional message sent with the transfer (← `message`; wire v37).
    #[serde(default)]
    pub message: Option<String>,
    /// The failure reason when `state == Failed` (← the `error` `GError`, rendered as text; wire
    /// v37).
    #[serde(default)]
    pub error: Option<String>,
    /// The remote content locator a `receive` fetches from (e.g. a Matrix `mxc://` URI);
    /// protocol-opaque, daemon-native (wire v37).
    #[serde(default)]
    pub source: Option<String>,
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
    /// A rich chat message (wire v37): the [`ChatMessage`] representation of a conversation-history
    /// entry, carrying delivery/edit state, author, and attachments the coarser `Block` shape omits.
    /// Additive — clients that only know `Management`/`Block` ignore it. Boxed to keep the enum small
    /// (`ChatMessage` is the largest variant); `Box<T>` serializes identically, so the wire shape is
    /// unchanged.
    Chat {
        /// The decoded chat message.
        message: Box<ChatMessage>,
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
    /// than misapplying a new log's entries onto the old one. `0` for the first activation
    /// (matching `Snapshot::fresh`). `#[serde(default)]` keeps old (epoch-less) encodings decodable.
    #[serde(default)]
    pub epoch: u64,
}

// ---------------------------------------------------------------------------
// Node-wide event feed DTOs (L3; daemon-sync-protocol-spec.md §5)
// ---------------------------------------------------------------------------

/// A coarse conversation-set delta (wire v30; [`NodeEvent::ConversationsChanged`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvChange {
    /// A conversation appeared (joined / created).
    Added,
    /// A conversation disappeared (left / deleted).
    Removed,
}

/// A membership transition (wire v30; [`NodeEvent::MembershipChanged`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MembershipChange {
    /// The member joined.
    Joined,
    /// The member left of their own accord.
    Left,
    /// The member was invited.
    Invited,
    /// The member was kicked (removed by another).
    Kicked,
    /// The member was banned.
    Banned,
}

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
    /// The fleet/subagent tree changed (a unit spawned / changed state / finished); the client
    /// refetches `Tree`. Like `RosterChanged`, a payload-free pointer carrying a coalescing `rev`.
    FleetChanged {
        /// The new fleet revision.
        rev: u64,
    },
    /// The profile set changed (wire v31): a profile was authored/edited/deleted by an operator op
    /// or the agent `profile_manage` tool; the client refetches the profile list (`ProfileList`).
    /// Like `RosterChanged`/`FleetChanged`, a payload-free pointer carrying a coalescing `rev`.
    ProfilesChanged {
        /// The new profiles revision.
        rev: u64,
    },
    /// An approval is pending operator action.
    ApprovalPending {
        /// The session it belongs to.
        session: SessionId,
        /// The approval's request id.
        request_id: String,
    },
    /// A model download advanced (replaces the client's poll). Emitted from the byte-level
    /// transfer sink, throttled node-side (>= 1 percent-point advance or >= 500 ms since the last
    /// emit), plus on every state transition and per-file completion.
    DownloadProgress {
        /// The download job id.
        id: DownloadId,
        /// Percent complete (0..=100).
        pct: u32,
        /// The job state string.
        state: String,
        /// Bytes transferred so far (across the job's files) — the client renders these directly.
        downloaded_bytes: u64,
        /// Total bytes to transfer, when known (0 when the Hub reported no sizes).
        total_bytes: u64,
    },
    /// The installed-model registry changed (a finished download was cataloged / a model was
    /// deleted): the client refetches `ModelCatalog`. Payload-free and globally coalesced in the
    /// backlog (a refetch always reads the whole catalog).
    CatalogChanged,
    /// An events-io transport instance's connection/presence changed (wire v29): emitted at the
    /// coarse real transitions (adapter serve start with the instance's reported state, clean
    /// teardown -> `Offline`, a crashed serve loop -> `Error`), carrying the full new state so a
    /// client updates its channel/presence dots WITHOUT re-polling `TransportInstances`.
    TransportChanged {
        /// The instance-qualified transport id (e.g. `"matrix/@bot:hs.org"`, `"room"`).
        transport: TransportId,
        /// The new connection state.
        connection: ConnectionState,
        /// The new presence (adapters without a presence source report `Unknown`).
        #[serde(default)]
        presence: PresenceState,
        /// Why the instance disconnected, when known (wire v30). `None` on a connect transition.
        #[serde(default)]
        reason: Option<DisconnectReason>,
        /// A human-readable disconnect detail (the adapter's error text), when any (wire v30).
        #[serde(default)]
        message: Option<String>,
        /// Node-decided: whether the disconnect is fatal (stop retrying; offer re-auth). Thin
        /// clients MUST NOT re-derive this (wire v30). `false` for transient reasons + connects.
        #[serde(default)]
        fatal: bool,
    },
    /// A transport's conversation set changed (wire v30): a conversation was added or removed.
    /// Retires client `ConvList` re-polling — a pointer; the client refetches `ConvGet`/`ConvList`.
    ConversationsChanged {
        /// The owning transport instance.
        transport: TransportId,
        /// The affected conversation id.
        conv: String,
        /// Added or removed.
        change: ConvChange,
    },
    /// A conversation's membership changed (wire v30). A granular invalidation pointer; on an
    /// `is_self` removal (`Left`/`Kicked`/`Banned`) the node has ALREADY reconciled its own routing
    /// registry (dropped the now-dangling `ChatRoute` pin for that origin) before emitting this.
    MembershipChanged {
        /// The owning transport instance.
        transport: TransportId,
        /// The affected conversation id.
        conv: String,
        /// The adapter-opaque member handle whose membership changed (e.g. a Matrix MXID). Clients
        /// re-fetch richer detail via `ConvGet`.
        member: String,
        /// What happened to `member`.
        change: MembershipChange,
        /// Who performed the action (the inviter/kicker/banner), when known.
        #[serde(default)]
        actor: Option<String>,
        /// A reason string (kick/ban reason), when the transport supplies one.
        #[serde(default)]
        reason: Option<String>,
        /// Whether `member` is THIS account. On a self `Left`/`Kicked`/`Banned` the node reconciled
        /// its routing for the now-dangling origin before emitting.
        is_self: bool,
    },
    /// A transport's server-side contact roster changed (wire v34): a contact was added, updated, or
    /// removed. A payload-free-per-transport invalidation pointer (named `ContactsChanged` to avoid
    /// colliding with the session-roster [`NodeEvent::RosterChanged`]); the client refetches the
    /// roster (`RosterList`). Retires client roster re-polling.
    ContactsChanged {
        /// The owning transport instance whose roster changed.
        transport: TransportId,
    },
    /// The feed could not serve from the client's cursor (aged out / lagged); the client must
    /// re-baseline the named scope ("roster" / "all" / ...).
    ResyncNeeded {
        /// The scope to refetch.
        scope: String,
    },
    /// The node's notification set changed (wire v37): a notification was added, removed, read, or
    /// deleted. A payload-free node-wide invalidation pointer (clients re-list via `NotificationList`),
    /// mirroring [`NodeEvent::CatalogChanged`] — the whole list is cheap to refetch, so it carries
    /// no per-notification detail.
    NotificationsChanged,
    /// The node's person/metacontact registry changed (wire v37): a person was created or removed,
    /// or a contact endpoint was associated/dissociated. A payload-free node-wide invalidation
    /// pointer (clients re-list via `PersonList`), mirroring [`NodeEvent::NotificationsChanged`].
    PersonsChanged,
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
                after: Some("conv-cursor".into()),
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
            ApiRequest::ConvSend(ConvSendArgs {
                transport: transport.clone(),
                conv: "r1".into(),
                from: Some(who.clone()),
                message: UserMsg::new("hi"),
            }),
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
            ApiRequest::ConvHistory(ConvHistoryArgs {
                transport: transport.clone(),
                conv: "r1".into(),
                after_cursor: 0,
                max: 16,
            }),
            ApiRequest::MemberInvite(MemberInviteArgs {
                transport: transport.clone(),
                conv: "r1".into(),
                who: who.clone(),
                message: None,
            }),
            ApiRequest::MemberRemove(MemberRemoveArgs {
                transport: transport.clone(),
                conv: "r1".into(),
                who: who.clone(),
                reason: Some("bye".into()),
            }),
            ApiRequest::MemberBan(MemberBanArgs {
                transport: transport.clone(),
                conv: "r1".into(),
                who: who.clone(),
                reason: None,
            }),
            ApiRequest::MemberSetRole(MemberSetRoleArgs {
                transport: transport.clone(),
                conv: "r1".into(),
                who: who.clone(),
                role: MemberRole::Op,
            }),
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
            ApiResponse::Conversations(WirePage {
                items: vec![info.clone()],
                next: Some(info.id.clone()),
            }),
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
                reason: None,
                message: None,
                fatal: false,
                enabled: true,
                label: None,
            }]),
            ApiResponse::Adapters(vec![AdapterInfo {
                family: "room".into(),
                display_name: "Rooms".into(),
                capabilities: AdapterCapabilities::default(),
                account_schema: AccountSettingsSchema::default(),
                policies: Vec::new(),
                // wire v33: an adapter that implements no feature trait leaves every ops field None.
                conversation_ops: None,
                membership_ops: None,
                contacts_ops: None,
                roster_ops: None,
                directory: false,
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
            ApiRequest::RecordMeta(RecordMetaArgs {
                session: SessionId::new("s1"),
                origin: daemon_protocol::Origin::new(
                    "gui",
                    daemon_protocol::OriginScope::Api {
                        key: "owner".into(),
                    },
                ),
                kind: "attach".into(),
                body: vec![9, 9, 9],
            }),
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
                after: Some("projects/zzz".into()),
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
            ApiRequest::FsWrite(FsWriteArgs {
                root: FsRootId::Session(SessionId::new("s1")),
                path: "a.txt".into(),
                bytes: vec![1, 2, 3],
                base_revision: Some(FsRevision {
                    mtime_ms: 10,
                    size: 3,
                }),
                force: false,
            }),
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
            ApiRequest::FsWatchPoll(FsWatchAfterArgs {
                root: FsRootId::Workspace,
                dir: String::new(),
                after_seq: 4,
                max: 32,
            }),
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
            ApiResponse::FsList(FsListPage {
                items: vec![FsEntry {
                    name: "src".into(),
                    path: "src".into(),
                    kind: FsEntryKind::Dir,
                    size: 0,
                    mtime_ms: 1,
                    ignored: false,
                }],
                next: Some("src".into()),
            }),
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
            ApiRequest::FsWriteFromBlob(FsWriteFromBlobArgs {
                root: FsRootId::Session(SessionId::new("s1")),
                path: "out/x.bin".into(),
                hash,
                base_revision: None,
                force: false,
            }),
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

    #[test]
    fn adapter_info_per_verb_ops_round_trip() {
        // wire v33: AdapterInfo with every per-verb ops descriptor populated (mixed flags) plus the
        // directory bool, and a second row with every new field absent — both must round-trip so
        // the additive fields agree with the CDDL under `api-response`.
        let full = AdapterInfo {
            family: "matrix".into(),
            display_name: "Matrix".into(),
            capabilities: AdapterCapabilities {
                rooms: true,
                direct_messages: true,
                presence: true,
                room_enumeration: true,
                file_transfer: false,
                interactive_auth: true,
            },
            account_schema: AccountSettingsSchema::default(),
            policies: Vec::new(),
            conversation_ops: Some(ConversationOps {
                create: true,
                join_channel: true,
                leave: true,
                delete: false,
                send: true,
                set_topic: false,
                set_title: true,
                set_description: false,
            }),
            membership_ops: Some(MembershipOps {
                invite: true,
                remove: false,
                ban: false,
                set_role: true,
            }),
            contacts_ops: Some(ContactsOps {
                get_profile: true,
                action_menu: false,
                set_alias: true,
            }),
            roster_ops: Some(RosterOps {
                list: true,
                add: true,
                update: false,
                remove: true,
            }),
            directory: true,
        };
        let bare = AdapterInfo {
            family: "room".into(),
            display_name: "Rooms (internal)".into(),
            capabilities: AdapterCapabilities::default(),
            account_schema: AccountSettingsSchema::default(),
            policies: Vec::new(),
            conversation_ops: None,
            membership_ops: None,
            contacts_ops: None,
            roster_ops: None,
            directory: false,
        };
        let resp = ApiResponse::Adapters(vec![full, bare]);
        let bytes = to_cbor(&resp);
        let back: ApiResponse = from_cbor(&bytes).unwrap();
        assert_eq!(resp, back);
    }

    #[test]
    fn adapter_info_decodes_pre_v33_payload() {
        // Back-compat: a v32-shaped AdapterInfo map (no per-verb ops fields at all) must still decode
        // — the new fields are `#[serde(default)]`, so an older peer's payload deserializes with the
        // ops absent (None) and directory false. This proves the v33 bump is additive on the wire.
        use ciborium::value::Value;
        let legacy = Value::Map(vec![
            (Value::Text("family".into()), Value::Text("room".into())),
            (
                Value::Text("display_name".into()),
                Value::Text("Rooms (internal)".into()),
            ),
            (
                Value::Text("capabilities".into()),
                Value::Map(vec![
                    (Value::Text("rooms".into()), Value::Bool(false)),
                    (Value::Text("direct_messages".into()), Value::Bool(false)),
                    (Value::Text("presence".into()), Value::Bool(false)),
                    (Value::Text("room_enumeration".into()), Value::Bool(false)),
                    (Value::Text("file_transfer".into()), Value::Bool(false)),
                    (Value::Text("interactive_auth".into()), Value::Bool(false)),
                ]),
            ),
        ]);
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&legacy, &mut bytes).unwrap();
        let info: AdapterInfo = from_cbor(&bytes).unwrap();
        assert_eq!(info.family, "room");
        assert_eq!(info.conversation_ops, None);
        assert_eq!(info.membership_ops, None);
        assert_eq!(info.contacts_ops, None);
        assert_eq!(info.roster_ops, None);
        assert!(!info.directory);
    }

    #[test]
    fn messaging_roster_requests_responses_and_event_round_trip() {
        // wire v34: the server-side roster surface — the four RosterList/Add/Update/Remove requests,
        // the paged ContactPage response, and the ContactsChanged node event — must all CBOR
        // round-trip so they agree with the CDDL under `api-request` / `api-response`.
        let contact = ContactInfo {
            id: "@bob:matrix.org".into(),
            display_name: Some("Bob".into()),
            presence: Presence::default(),
            permission: ContactPermission::Allow,
        };
        let transport = TransportId::new("matrix/@me:hs.org");
        let reqs = vec![
            ApiRequest::RosterList {
                transport: transport.clone(),
                after: Some("@aaa:matrix.org".into()),
            },
            ApiRequest::RosterList {
                transport: transport.clone(),
                after: None,
            },
            ApiRequest::RosterAdd {
                transport: transport.clone(),
                contact: contact.clone(),
            },
            ApiRequest::RosterUpdate {
                transport: transport.clone(),
                contact: contact.clone(),
            },
            ApiRequest::RosterRemove {
                transport: transport.clone(),
                contact: contact.clone(),
            },
        ];
        for req in reqs {
            assert_eq!(req, from_cbor::<ApiRequest>(&to_cbor(&req)).unwrap());
        }

        let resps = vec![
            ApiResponse::ContactPage(WirePage {
                items: vec![contact.clone()],
                next: Some("@bob:matrix.org".into()),
            }),
            ApiResponse::ContactPage(WirePage::default()),
            ApiResponse::Ok,
        ];
        for resp in resps {
            assert_eq!(resp, from_cbor::<ApiResponse>(&to_cbor(&resp)).unwrap());
        }

        // The ContactsChanged event round-trips inside an EventsPage (its wire carrier).
        let wrapped = EventsPage {
            events: vec![NodeEvent::ContactsChanged { transport }],
            next_cursor: 1,
            head_cursor: 1,
        };
        assert_eq!(
            wrapped,
            from_cbor::<EventsPage>(&to_cbor(&wrapped)).unwrap()
        );
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
            ApiRequest::SessionSearch {
                query: "build".into(),
                limit: 10,
            },
            ApiRequest::SessionRecap {
                session: SessionId::new("s1"),
            },
            ApiRequest::Rewind {
                session: SessionId::new("s1"),
                point: RewindPoint {
                    anchor: RewindAnchor::UserTurn { ordinal: 3 },
                    restore_workspace: true,
                },
            },
            ApiRequest::AgentDiscover,
            ApiRequest::AgentRegister {
                entry: AgentEntry {
                    name: "gemini".into(),
                    recipe: AgentRecipe {
                        program: Some("gemini".into()),
                        args: vec!["--acp".into()],
                        env: vec![("KEY".into(), "v".into())],
                        endpoint: None,
                    },
                    source: AgentSource::Builtin,
                    protocol: AgentProtocol::Acp,
                    installed: true,
                    version: Some("0.1".into()),
                    capabilities: vec![("fs".into(), "true".into())],
                    verification: AgentVerification::Verified,
                },
            },
            ApiRequest::AgentRegister {
                entry: AgentEntry {
                    name: "claude".into(),
                    recipe: AgentRecipe {
                        program: Some("claude".into()),
                        args: vec!["--output-format".into(), "stream-json".into()],
                        env: Vec::new(),
                        endpoint: None,
                    },
                    source: AgentSource::Manual,
                    protocol: AgentProtocol::StreamJson,
                    installed: true,
                    version: None,
                    capabilities: Vec::new(),
                    verification: AgentVerification::Unverified,
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
                engine: EngineSelector::Foreign {
                    agent: "codex".into(),
                },
                foreign_backend: ForeignBackend::NodeProvider {
                    provider: ProviderSelector::GenAi,
                    model: "gpt-4o".into(),
                    credential_ref: None,
                },
                model_selector: Some(ModelSelector {
                    option_id: "model".into(),
                    current: "mock-model-a".into(),
                    choices: vec![
                        ModelChoice {
                            id: "mock-model-a".into(),
                            label: "Mock Model A".into(),
                        },
                        ModelChoice {
                            id: "mock-model-b".into(),
                            label: "Mock Model B".into(),
                        },
                    ],
                }),
            })),
            ApiResponse::SessionSearch(vec![SessionSearchHit {
                session: SessionId::new("s1"),
                title: "hello".into(),
                snippet: "…[hello]…".into(),
            }]),
            ApiResponse::SessionRecap(Some(SessionRecap {
                title: Some("build fixes".into()),
                user_turns: 4,
                assistant_turns: 5,
                tool_results: 3,
                top_tools: vec![("fs".into(), 2), ("shell".into(), 1)],
                files_touched: vec!["src/lib.rs".into()],
                last_ask: Some("fix the build".into()),
                last_reply: Some("done".into()),
            })),
            ApiResponse::SessionRecap(None),
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
                lifetime: Some(daemon_protocol::DelegationLifetime::Persistent),
                engine: Some(EngineSelector::Foreign {
                    agent: "gemini".into(),
                }),
            }],
            next: Some("root".into()),
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

mod matching;
pub use matching::*;
mod args;
pub use args::*;
mod wire;
pub use wire::*;
mod dispatch;
pub use dispatch::*;
mod details;
pub use details::*;
mod saved_presence;
pub use saved_presence::*;

mod message;
pub use message::*;

mod tags;
pub use tags::*;

mod notify;
pub use notify::*;

mod person;
pub use person::*;

mod request;
pub use request::*;

mod file_transfer;
pub use file_transfer::*;
