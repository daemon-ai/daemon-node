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
    ModelRef, QuantRecommendation, QuantizeId, QuantizeStatus, SearchPage, SearchQuery, SessionId,
    UnitId, UsageDelta, WireVersion,
};
use daemon_protocol::{
    session_id_for, AgentCommand, DeliveryTarget, HostResponse, IsolationPolicy, Origin,
    TranscriptBlock,
};
pub use daemon_protocol::{Outbound, SessionLogEntry};
use futures::stream::{self, BoxStream, StreamExt};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub mod profile;
pub use profile::{
    BudgetSpec, ConfigField, ConfigPatch, ConfigSchema, ContextEngineSel, CredentialInfo,
    EngineTunables, MemoryProviderSel, ModelDescriptor, ProfileInfo, ProfileSpec, ProviderSelector,
};

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

    /// Switch a **live** session's model (and optionally its provider) in place — a transient,
    /// per-session model switch (the bound profile is not mutated, mirroring hermes
    /// `set_session_model`). The host rebuilds a provider for the new model from the session's
    /// profile and swaps it on the running engine; it takes effect at the next turn boundary so an
    /// in-flight turn's prompt cache is preserved. Default: unsupported (a transport with no live
    /// model factory). `provider = None` keeps the profile's current provider.
    async fn set_session_model(
        &self,
        _session: SessionId,
        _model: String,
        _provider: Option<ProviderSelector>,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("set_session_model".into()))
    }

    /// Set a **live** session's edit-approval [`ApprovalMode`] (the §12 session mode) in place — a
    /// transient, per-session switch (the bound profile is not mutated, mirroring hermes
    /// `set_session_mode`). It governs how a gated tool action (an fs edit, a dangerous shell
    /// command) is serviced: auto-allow, deny, or ask (the host parks for a human on the live path
    /// or suspends the turn on the durable path). Takes effect at the next gated action. Default:
    /// unsupported (a transport with no live session policy store).
    async fn set_session_mode(
        &self,
        _session: SessionId,
        _mode: ApprovalMode,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("set_session_mode".into()))
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

    /// The known durable sessions and their statuses.
    async fn sessions(&self) -> Vec<SessionInfo>;

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

/// The profile / runtime-config sub-surface: create, inspect, edit, and select the agent
/// configuration bundles ([`ProfileSpec`]) a session binds to, plus the dynamically-settable
/// runtime config (`DAEMON_MODEL`/`DAEMON_MODEL_PROVIDER`/persona/credential-ref) the GUI drives
/// without restarting the node. Every method defaults to [`ApiError::Unsupported`] / empty so a
/// transport that hosts no profile store (the session-only FFI, test stubs) inherits the surface;
/// the node's [`NodeApi`] binds the real implementation (backed by a `ProfileStore`).
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

    /// The resolved effective config for `profile` (`None` = the active default).
    async fn config_get(&self, _profile: Option<String>) -> Result<Option<ProfileSpec>, ApiError> {
        Err(ApiError::Unsupported("config_get".into()))
    }

    /// Apply a runtime-config patch to `profile` (`None` = the active default).
    async fn config_set(
        &self,
        _profile: Option<String>,
        _patch: ConfigPatch,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("config_set".into()))
    }

    /// The settable-config schema (a GUI renders it as a settings form). Default: the built-in schema.
    async fn config_schema(&self) -> ConfigSchema {
        ConfigSchema::builtin()
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

/// The whole node surface: the session, control, model-management, profile/config, and credential
/// sub-surfaces.
pub trait NodeApi: SessionApi + ControlApi + ModelApi + ProfileApi + CredentialApi {}
impl<T: SessionApi + ControlApi + ModelApi + ProfileApi + CredentialApi> NodeApi for T {}

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

/// A durable session's identity + lifecycle state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionInfo {
    /// The session id.
    pub session: SessionId,
    /// Its durable lifecycle state.
    pub state: SessionState,
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
pub use daemon_protocol::{TreeReport, UnitKind, UnitNode, UnitState};

// `ManageEventView` is defined in `daemon-protocol` (so the `ManagedUnit` projection seam can carry
// it without a surface-crate edge) and re-exported here unchanged.
pub use daemon_protocol::ManageEventView;

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
    /// [`SessionApi::submit`] / [`SessionApi::submit_from`].
    Submit {
        /// Target session.
        session: SessionId,
        /// The §17 command.
        command: AgentCommand,
        /// Optional per-event attribution. `None` (old encodings) drops to the host-local default;
        /// `Some` routes through [`SessionApi::submit_from`] so the origin is recorded on the log.
        #[serde(default)]
        origin: Option<Origin>,
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
    /// [`ProfileApi::config_get`].
    ConfigGet {
        /// The profile to resolve (`None` = the active default).
        profile: Option<String>,
    },
    /// [`ProfileApi::config_set`].
    ConfigSet {
        /// The profile to patch (`None` = the active default).
        profile: Option<String>,
        /// The partial config update.
        patch: ConfigPatch,
    },
    /// [`ProfileApi::config_schema`].
    ConfigSchema,
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
    /// One profile's full spec, or `None` if unknown / no active default (profile_get/config_get).
    Profile(Option<ProfileSpec>),
    /// The settable-config schema.
    ConfigSchema(ConfigSchema),
    /// A redacted credential listing.
    Credentials(Vec<CredentialInfo>),
    /// A discoverable model catalog (cloud + local).
    Models(Vec<ModelDescriptor>),
    /// The model a profile currently resolves to (`None` = none resolvable).
    ModelCurrent(Option<ModelDescriptor>),
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
        } => match origin {
            Some(origin) => unit_or_err(api.submit_from(session, origin, command).await),
            None => unit_or_err(api.submit(session, command).await),
        },
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
        ApiRequest::ConfigGet { profile } => match api.config_get(profile).await {
            Ok(spec) => ApiResponse::Profile(spec),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::ConfigSet { profile, patch } => {
            unit_or_err(api.config_set(profile, patch).await)
        }
        ApiRequest::ConfigSchema => ApiResponse::ConfigSchema(api.config_schema().await),
        ApiRequest::CredentialSet { profile, secret } => {
            unit_or_err(api.credential_set(profile, secret).await)
        }
        ApiRequest::CredentialList => ApiResponse::Credentials(api.credential_list().await),
        ApiRequest::CredentialRemove { profile } => {
            unit_or_err(api.credential_remove(profile).await)
        }
        ApiRequest::Models => ApiResponse::Models(api.models().await),
        ApiRequest::ModelCurrent { profile } => match api.model_current(profile).await {
            Ok(m) => ApiResponse::ModelCurrent(m),
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
        | ApiRequest::Handover { .. }
        | ApiRequest::RecordMeta { .. }
        | ApiRequest::SetSessionModel { .. }
        | ApiRequest::SetSessionMode { .. } => {
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
