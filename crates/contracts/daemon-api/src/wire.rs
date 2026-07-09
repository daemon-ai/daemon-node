// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The serializable wire mirror: `ApiRequest` / `ApiResponse` (CBOR; governed by `daemon-api.cddl`).

use crate::*;
use serde::{Deserialize, Serialize};

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
    /// [`SessionApi::session_create`]: node-authoritative creation of a blank, profile-bound, UN-RUN
    /// session. The node mints (`session = None`) or accepts the id, binds `profile` (or the active
    /// default), persists it (so it appears in the roster + ByProfile query), emits `RosterChanged`,
    /// and replies [`ApiResponse::SessionCreated`] with the id.
    SessionCreate {
        /// The id to accept, or `None` to let the node mint one.
        #[serde(default)]
        session: Option<SessionId>,
        /// The profile to bind on creation, or `None` to bind the node's active default.
        #[serde(default)]
        profile: Option<ProfileRef>,
    },
    /// [`ControlApi::cancel`].
    Cancel {
        /// Session to cancel.
        session: SessionId,
    },
    /// [`ControlApi::fleet`].
    Fleet,
    /// [`ControlApi::tree`].
    Tree {
        /// Resume cursor: the `next` of the previous [`TreeReport`] (the last node's unit id).
        /// `None` starts at the first node.
        #[serde(default)]
        after: Option<String>,
    },
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
        /// Resume cursor: the previous page's `next` (the last served session id).
        #[serde(default)]
        after: Option<String>,
    },
    /// [`SessionApi::handover`].
    Handover {
        /// The session whose `Primary` reply sink to re-point.
        session: SessionId,
        /// The new `Primary` target.
        target: DeliveryTarget,
    },
    /// [`SessionApi::record_meta`] — record an observability-only transport/meta event.
    RecordMeta(RecordMetaArgs),
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
        /// Resume cursor: the previous page's `next` (the last served file `path`).
        #[serde(default)]
        after: Option<String>,
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
    ModelRecommend(ModelRecommendArgs),
    /// [`ModelApi::model_quantize`].
    ModelQuantize(ModelQuantizeArgs),
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
        /// Resume cursor: the previous page's `next` (the last served revision's stringified `seq`).
        #[serde(default)]
        after: Option<String>,
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
    /// [`ProfileApi::soul_get`] — read a profile's persona (SOUL.md) text (wire v36).
    SoulGet {
        /// The profile id whose persona to read.
        id: String,
    },
    /// [`ProfileApi::soul_set`] — replace a profile's persona (SOUL.md) text (wire v36). The node
    /// validates/scans/caps + revision-logs; rejected for a Foreign-engine profile (its agent owns
    /// its own prompt — there is no persona to set).
    SoulSet {
        /// The profile id whose persona to set.
        id: String,
        /// The new persona text.
        text: String,
    },
    /// [`ProfileApi::skill_history`].
    SkillHistory {
        /// The skill (bundle) name whose history to list.
        name: String,
        /// Resume cursor: the previous page's `next` (the last served revision's stringified `seq`).
        #[serde(default)]
        after: Option<String>,
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
    /// [`CredentialApi::credential_set_label`] — set/clear a credential/account's human label
    /// (wire v35).
    CredentialSetLabel {
        /// The profile / credential-ref to label.
        profile: String,
        /// The new label (`None` clears).
        #[serde(default)]
        label: Option<String>,
    },
    /// [`AuthApi::auth_begin`].
    AuthBegin(AuthBeginRequest),
    /// [`AuthApi::auth_step`].
    AuthStep(AuthStepRequest),
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
    Models {
        /// Resume cursor: the previous page's `next` (the last served descriptor `id`).
        #[serde(default)]
        after: Option<String>,
    },
    /// [`ModelApi::model_current`].
    ModelCurrent {
        /// The profile to resolve (`None` = the active default).
        profile: Option<String>,
    },
    /// [`ModelApi::provider_catalog`] — enumerate providers (local engines + every genai cloud
    /// vendor + Daemon Cloud) for the setup picker. Gated on `Capability::ModelsRead`.
    ProviderCatalog,
    /// [`ModelApi::provider_models`] — list one provider's discoverable models, credential-aware for
    /// genai vendors. Gated on `Capability::ModelsRead`.
    ProviderModels {
        /// The provider to enumerate — a [`ProviderDescriptor::id`] from `ProviderCatalog` (e.g.
        /// `"anthropic"`, `"daemon_cloud"`, `"llama_cpp"`). This carries the vendor dimension that
        /// `ProviderSelector` alone cannot (every genai cloud vendor shares `ProviderSelector::GenAi`).
        provider: String,
        /// A stored credential ref to authenticate the LIST call with (genai vendors).
        #[serde(default)]
        credential_ref: Option<String>,
        /// A first-run transient key to authenticate the LIST call with, before a credential is
        /// stored. Never persisted; used only for this listing call (turns use the stored profile
        /// credential).
        #[serde(default)]
        transient_key: Option<String>,
        /// Resume cursor: the previous page's `next` (the last served descriptor `id`).
        #[serde(default)]
        after: Option<String>,
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
        /// Resume cursor: the previous page's `next` (the last served `request_id`).
        #[serde(default)]
        after: Option<String>,
    },
    /// [`ControlApi::approval_decide`].
    ApprovalDecide {
        /// The session that parked the request.
        session: SessionId,
        /// The opaque parked-request id (from [`ApprovalInfo`]).
        request_id: String,
        /// The operator's decision (allow / deny).
        allow: bool,
        /// "Allow permanently" (Cluster B): when allowing, also remember the approved command's
        /// fingerprint on the session allow-list so an identical in-session re-request auto-approves.
        /// Honored only where the parked approval carries a fingerprint (the durable inbox offers it
        /// when `ApprovalInfo.fingerprint` is set); otherwise it degrades to a single allow. Additive.
        #[serde(default)]
        allow_permanent: bool,
        /// An optional operator justification (wire v29). On a deny it is injected into the
        /// agent's conversation as the gated tool's error content, so the model can adapt its
        /// next attempt instead of guessing why the action was refused. Ignored on allow.
        #[serde(default)]
        reason: Option<String>,
    },
    /// [`ControlApi::fingerprint_list`] — a session's remembered exec-approval fingerprints.
    FingerprintList {
        /// The session whose `allow_permanent` allow-list to read.
        session: SessionId,
    },
    /// [`ControlApi::fingerprint_revoke`] — drop one remembered fingerprint (re-prompts next time).
    FingerprintRevoke {
        /// The session whose allow-list to edit.
        session: SessionId,
        /// The fingerprint hex (from [`ControlApi::fingerprint_list`] / `ApprovalInfo.fingerprint`).
        fingerprint: String,
    },
    /// [`ControlApi::checkpoints`].
    CheckpointList {
        /// Filter to one session, or `None` for the node-wide checkpoint list.
        #[serde(default)]
        session: Option<SessionId>,
        /// Resume cursor: the previous page's `next` (the last served checkpoint `id`).
        #[serde(default)]
        after: Option<String>,
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
    /// [`ControlApi::session_search`] — full-text session search.
    SessionSearch {
        /// The search query.
        query: String,
        /// Max hits (`0` = a server default).
        limit: u32,
    },
    /// [`ControlApi::session_recap`] — a pure-local recap of one session's recent activity.
    SessionRecap {
        /// The session to recap.
        session: SessionId,
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
    /// [`ControlApi::agent_discover`] — trigger a foreign-agent discovery scan.
    AgentDiscover,
    /// [`ControlApi::agent_catalog`] — the persisted foreign-agent catalog.
    AgentCatalog,
    /// [`ControlApi::agent_register`] — register a foreign-agent launch recipe.
    AgentRegister {
        /// The recipe to persist.
        entry: AgentEntry,
    },
    /// [`ControlApi::agent_remove`] — remove a cataloged/registered foreign agent.
    AgentRemove {
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
    /// [`ControlApi::tool_set_enabled`] — persist a node-wide enable/disable override (wire v30).
    ToolSetEnabled {
        /// The tool name (as in `tool-info.name` / `ProfileSpec.tool_allowlist`).
        tool: String,
        /// The override: `true` force-enable, `false` force-disable.
        enabled: bool,
    },
    /// [`ControlApi::command_list`] — the daemon-authoritative command catalog.
    CommandList,
    /// [`ControlApi::command_invoke`] — run a command by name.
    CommandInvoke {
        /// The command + args + session/origin context.
        invocation: CommandInvocation,
    },
    /// [`ControlApi::caps`] — the read-only delegation guardrail caps (wire v29).
    Caps,
    /// [`ControlApi::config_get`].
    ConfigGet,
    /// [`ControlApi::config_set`].
    ConfigSet {
        /// The replacement config.
        config: NodeConfigView,
    },
    /// [`ControlApi::gateway_get`] — read the node-owned gateway's runtime status.
    GatewayGet,
    /// [`ControlApi::gateway_set`] — enable/disable + optionally rebind the gateway listener.
    GatewaySet {
        /// Whether the gateway should be serving.
        enabled: bool,
        /// An optional new bind address (loopback recommended); `None` keeps the current/boot addr.
        #[serde(default)]
        addr: Option<String>,
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
    RoutingListChats {
        /// Resume cursor: the previous page's `next` (the last served route's origin pin key).
        #[serde(default)]
        after: Option<String>,
    },
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
        /// Resume cursor: the previous page's `next` (the last served `room`).
        #[serde(default)]
        after: Option<String>,
    },
    /// [`ControlApi::transport_adapters`] — the available adapter families + capabilities + schema.
    TransportAdapters,
    /// [`ControlApi::transport_instances`] — the configured instances + live connection/presence.
    TransportInstances,
    /// [`ControlApi::transport_disconnect`] — stop an instance's serve loop, go
    /// `ConnectionState::Offline`, KEEP credential/config/bound_profile (reversible; wire v30).
    TransportDisconnect {
        /// The instance-qualified transport id.
        transport: TransportId,
    },
    /// [`ControlApi::transport_remove`] — remove implies disconnect, then one node-side teardown:
    /// close conversations, unbind routing, drop credential + config (wire v30).
    TransportRemove {
        /// The instance-qualified transport id.
        transport: TransportId,
    },
    /// [`ControlApi::transport_connect`] — resume a disconnected instance's family serve loop
    /// (wire v35; the reversible counterpart of `TransportDisconnect`). Idempotent.
    TransportConnect {
        /// The instance-qualified transport id.
        transport: TransportId,
    },
    /// [`ControlApi::transport_set_enabled`] — persist the desired enabled/disabled state (wire
    /// v35): `false` disconnects now + skips at spawn; `true` persists + attempts to reconnect.
    TransportSetEnabled {
        /// The instance-qualified transport id.
        transport: TransportId,
        /// The desired enabled state.
        enabled: bool,
    },
    /// [`ControlApi::transport_set_label`] — set/clear the instance's human label (wire v35).
    TransportSetLabel {
        /// The instance-qualified transport id.
        transport: TransportId,
        /// The new label (`None` clears).
        #[serde(default)]
        label: Option<String>,
    },
    /// [`ControlApi::conv_list`] — a transport's conversations.
    ConvList {
        /// The owning transport.
        transport: TransportId,
        /// Resume cursor: the previous page's `next` (the last served conversation `id`).
        #[serde(default)]
        after: Option<String>,
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
    ConvSend(ConvSendArgs),
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
    ConvHistory(ConvHistoryArgs),
    /// [`ControlApi::member_invite`] — invite/add a participant.
    MemberInvite(MemberInviteArgs),
    /// [`ControlApi::member_remove`] — remove/kick a participant.
    MemberRemove(MemberRemoveArgs),
    /// [`ControlApi::member_ban`] — ban a participant.
    MemberBan(MemberBanArgs),
    /// [`ControlApi::member_set_role`] — set a participant's role.
    MemberSetRole(MemberSetRoleArgs),
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
    /// [`ControlApi::roster_list`] — a transport's server-side contact roster (wire v34).
    RosterList {
        /// The owning transport.
        transport: TransportId,
        /// Resume cursor: the previous page's `next` (the last served contact `id`).
        #[serde(default)]
        after: Option<String>,
    },
    /// [`ControlApi::roster_add`] — add a contact to the server-side roster (wire v34).
    RosterAdd {
        /// The owning transport.
        transport: TransportId,
        /// The contact to add.
        contact: ContactInfo,
    },
    /// [`ControlApi::roster_update`] — update a contact on the server-side roster (wire v34).
    RosterUpdate {
        /// The owning transport.
        transport: TransportId,
        /// The contact to update.
        contact: ContactInfo,
    },
    /// [`ControlApi::roster_remove`] — remove a contact from the server-side roster (wire v34).
    RosterRemove {
        /// The owning transport.
        transport: TransportId,
        /// The contact to remove.
        contact: ContactInfo,
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
        /// Resume cursor: the `next` of the previous [`FsListPage`] (the last entry's `path`).
        /// `None` starts at the first entry.
        #[serde(default)]
        after: Option<String>,
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
    FsWrite(FsWriteArgs),
    /// [`ControlApi::fs_search`].
    FsSearch {
        /// The root to search within.
        root: FsRootId,
        /// The search query.
        query: FsSearchQuery,
    },
    /// [`ControlApi::fs_watch_after`] — the cursor / long-poll form of the change stream.
    FsWatchPoll(FsWatchAfterArgs),
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
    FsWriteFromBlob(FsWriteFromBlobArgs),

    // -- access control (Auth 5): admin user/role/session management -----------------------------
    /// [`AccessControlApi::user_create`].
    UserCreate {
        /// The new account's username.
        username: String,
        /// The initial password (request-only; never echoed, logged, or audited).
        password: String,
        /// The initial role set (snake_case names, e.g. `"operator"`).
        roles: Vec<String>,
    },
    /// [`AccessControlApi::user_list`].
    UserList,
    /// [`AccessControlApi::user_disable`].
    UserDisable {
        /// The target user's stable id.
        user_id: String,
        /// `true` disables the account (and revokes its sessions); `false` re-enables it.
        disabled: bool,
    },
    /// [`AccessControlApi::user_set_roles`].
    UserSetRoles {
        /// The target user's stable id.
        user_id: String,
        /// The replacement role set (snake_case names).
        roles: Vec<String>,
    },
    /// [`AccessControlApi::user_set_password`].
    UserSetPassword {
        /// The target user's stable id.
        user_id: String,
        /// The new password (request-only; never echoed, logged, or audited).
        password: String,
    },
    /// [`AccessControlApi::role_list`].
    RoleList,
    /// [`AccessControlApi::who_am_i`] — the caller's own [`PrincipalView`]. Allowed for any
    /// authenticated principal (no `AccessAdmin` required).
    WhoAmI,
    /// [`AccessControlApi::session_revoke`] — revoke **all** session tokens for a user.
    SessionRevoke {
        /// The user whose sessions to revoke.
        user_id: String,
    },
    /// [`AccessControlApi::resource_grant_create`] — reserved (option B per the access-control spec
    /// §5); the handler returns [`ApiError::Unsupported`] until per-resource grants are enforced.
    ResourceGrantCreate {
        /// The grantee user id.
        user_id: String,
        /// The resource kind (e.g. `"session"`).
        resource_kind: String,
        /// The resource id.
        resource_id: String,
        /// The capability granted over the resource (snake_case name).
        capability: String,
    },
    /// [`AccessControlApi::resource_grant_list`] — reserved; returns [`ApiError::Unsupported`].
    ResourceGrantList {
        /// Filter to one grantee, or `None` for all grants.
        #[serde(default)]
        user_id: Option<String>,
    },
    /// [`AccessControlApi::resource_grant_revoke`] — reserved; returns [`ApiError::Unsupported`].
    ResourceGrantRevoke {
        /// The grant id to revoke.
        id: String,
    },

    // -- user feedback over OpenTelemetry (N1; wire v32) -----------------------------------------
    /// [`ControlApi::feedback_submit`] — submit thumbs up/down + optional comment on an agent
    /// response, or general app feedback. Explicit feedback is per-event consent: it is
    /// accepted+queued even when the global telemetry toggle is off (passive telemetry stays
    /// gated). Answered by [`ApiResponse::FeedbackAck`].
    FeedbackSubmit {
        /// The feedback flavor (response vs. app).
        kind: FeedbackKind,
        /// The rated response, for [`FeedbackKind::Response`] (`None` for app feedback).
        #[serde(default)]
        target: Option<FeedbackTarget>,
        /// The thumbs up/down rating, when given.
        #[serde(default)]
        rating: Option<FeedbackRating>,
        /// A free-form comment, when given (server-capped at [`crate::FEEDBACK_COMMENT_MAX`] bytes).
        #[serde(default)]
        comment: Option<String>,
        /// Whether the client consents to including the rated response content in the exported event.
        include_content: bool,
        /// Optional client diagnostics (app version / OS).
        #[serde(default)]
        diagnostics: Option<FeedbackDiagnostics>,
        /// The UI surface the feedback was given from (free-form label, e.g. `"transcript"`).
        surface: String,
    },
    /// [`ControlApi::telemetry_consent_get`] — read the node-owned global telemetry consent toggle
    /// (default OFF / opt-in). Answered by [`ApiResponse::TelemetryConsent`].
    TelemetryConsentGet,
    /// [`ControlApi::telemetry_consent_set`] — set the node-owned global telemetry consent toggle;
    /// the reply ([`ApiResponse::TelemetryConsent`]) echoes the new state.
    TelemetryConsentSet {
        /// The new consent state (`true` opts passive telemetry in).
        enabled: bool,
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
    /// The session a [`ApiRequest::SessionCreate`] minted/accepted (subscribe/poll it, or open it in
    /// the GUI). The node-authoritative counterpart to a client-minted id.
    SessionCreated {
        /// The created (blank, profile-bound, UN-RUN) session id.
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
    /// A page of parked §12 edit-approval requests awaiting an operator decision (request_id order).
    Approvals(WirePage<ApprovalInfo>),
    /// A session's remembered exec-approval fingerprints (fingerprint_list; wire v29).
    Fingerprints(Vec<RememberedFingerprint>),
    /// A page of recorded §12 tool checkpoints (rewind points), checkpoint-id order.
    Checkpoints(WirePage<CheckpointInfo>),
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
    /// A page of the live sessions a transport instance owns for delivery (`delivery_sessions`),
    /// session-id order.
    DeliverySessions(WirePage<SessionId>),
    /// The node's journal verifying key (hex dCBOR), or `None` if it exposes no signer.
    VerifyingKey(Option<String>),
    /// A page of model search results.
    ModelSearch(SearchPage),
    /// A page of a repo's loadable files, path order.
    ModelFiles(WirePage<ModelFile>),
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
    /// A profile's persona (SOUL.md) text (soul_get; wire v36).
    SoulText(String),
    /// A redacted credential listing.
    Credentials(Vec<CredentialInfo>),
    /// A begun interactive-auth flow handle (`auth_begin`).
    AuthBegun(AuthBeginResponse),
    /// A step result: the next challenge or the completed outcome (`auth_step`).
    AuthStepped(AuthStepResult),
    /// A completed interactive-auth flow outcome (`auth_complete`).
    AuthCompleted(AuthCompleteResponse),
    /// The registered interactive-auth providers (`auth_providers`).
    AuthProviders(Vec<AuthProviderInfo>),
    /// A page of the discoverable model catalog (cloud + local), descriptor-id order.
    Models(WirePage<ModelDescriptor>),
    /// The model a profile currently resolves to (`None` = none resolvable).
    ModelCurrent(Option<ModelDescriptor>),
    /// The discoverable provider catalog (`provider_catalog`): local engines + genai vendors +
    /// Daemon Cloud.
    ProviderCatalog(Vec<ProviderDescriptor>),
    /// A page of a provider's discoverable models (`provider_models`), descriptor-id order.
    ProviderModels(WirePage<ModelDescriptor>),
    /// A profile distribution (profile_export).
    Distribution(Distribution),
    /// A created profile id (profile_import).
    ProfileId(String),
    /// A page of a revision history (profile_history / skill_history), oldest first (`seq` order;
    /// the cursor is the stringified `seq`).
    Revisions(WirePage<Revision>),
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
    /// Full-text session-search hits (session_search).
    SessionSearch(Vec<SessionSearchHit>),
    /// A session's pure-local activity recap, or `None` if unknown/unrecoverable (session_recap).
    SessionRecap(Option<SessionRecap>),
    /// The foreign-agent catalog (agent_discover / agent_catalog).
    AgentCatalog(Vec<AgentEntry>),
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
    /// The node-owned gateway's runtime status (gateway_get / gateway_set).
    GatewayStatus(GatewayStatus),
    /// The read-only delegation guardrail caps (caps; wire v29).
    Caps(CapsReport),
    /// The scheduled cron jobs (cron_list).
    CronJobs(Vec<CronJob>),
    /// A created cron job id (cron_create).
    CronId(String),
    /// Recent runs of a scheduled job (cron_runs).
    CronRuns(Vec<CronRun>),
    /// Pending cron-job suggestions (cron_suggestions).
    CronSuggestions(Vec<CronSuggestion>),
    /// A page of the chat→session routing pins (routing_list_chats), origin-pin-key order.
    ChatRoutes(WirePage<ChatRoute>),
    /// One origin's routing pin, if set (routing_get).
    ChatRoute(Option<ChatRoute>),
    /// A page of a transport instance's rooms (transport_rooms), room order.
    Rooms(WirePage<RoomInfo>),
    /// A page of a transport's conversations (conv_list), conversation-id order.
    Conversations(WirePage<ConversationInfo>),
    /// One conversation, if present (conv_get / conv_create / conv_join).
    Conversation(Option<ConversationInfo>),
    /// A remote contact's profile text (contact_get_profile).
    ContactProfile(String),
    /// A list of contacts (directory_search).
    Contacts(Vec<ContactInfo>),
    /// A page of a transport's server-side contact roster (roster_list), contact-id order (wire v34).
    ContactPage(WirePage<ContactInfo>),
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
    /// A directory listing page (fs_list).
    FsList(FsListPage),
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

    // -- access control (Auth 5) -----------------------------------------------------------------
    /// A single user record (user_create).
    AccessUser(AccessUser),
    /// The user listing (user_list).
    AccessUsers(Vec<AccessUser>),
    /// The built-in roles and their effective capabilities (role_list).
    AccessRoles(Vec<RoleInfo>),
    /// The caller's own principal view (who_am_i).
    WhoAmI(PrincipalView),

    // -- user feedback over OpenTelemetry (N1; wire v32) -----------------------------------------
    /// The acknowledgement for a `FeedbackSubmit` — accepted+queued to the durable feedback outbox
    /// (NOT delivered; export is a separate best-effort drain).
    FeedbackAck(FeedbackAck),
    /// The node-owned global telemetry consent toggle (the reply to both
    /// `TelemetryConsentGet` and `TelemetryConsentSet`; the latter echoes the new state).
    TelemetryConsent {
        /// The current consent state.
        enabled: bool,
    },
}

// ---------------------------------------------------------------------------
// Multiplexed / server-streaming socket envelope (wire L0; daemon-sync-protocol-spec.md §2)
// ---------------------------------------------------------------------------

/// The wire protocol version a `Hello` negotiates. Bumped when the envelope shape changes.
/// v2 adds the SASL-style authentication exchange (`AuthStart`/`AuthStep`/`AuthResume` ->
/// `AuthChallenge`/`AuthOk`/`AuthError`) and the `auth_mechanisms` list on the server `Hello`.
pub const WIRE_VERSION: u32 = 2;
/// Feature flag: the connection speaks the multiplexed `Call`/`Reply` envelope.
pub const WIRE_FEATURE_MUX: &str = "mux";
/// Feature flag: the server can push `Item`/`End` frames for streaming requests.
pub const WIRE_FEATURE_STREAM: &str = "stream";
/// Feature flag: the node hosts profile/skill versioning (a bound revision log), so the
/// `Profile{History,At,Revert}` (+ skill) ops are available rather than `Unsupported`.
pub const WIRE_FEATURE_VERSIONING: &str = "versioning";
/// Feature flag: the node requires/offers SASL-style authentication before it accepts `Call`/`Open`.
/// When advertised, the server `Hello` carries the offered `auth_mechanisms` and the client must
/// complete an `AuthStart`/`AuthStep` (or `AuthResume`) exchange ending in `AuthOk` first.
pub const WIRE_FEATURE_AUTH: &str = "auth";
/// Feature-string prefix carrying the server's daemon-api CONTRACT version in its `Hello`
/// (`"api/<N>"`, N = [`crate::API_WIRE_VERSION`]). Distinct from [`WIRE_VERSION`] (the envelope
/// version): the contract version governs whether a peer can decode the wrapped
/// [`ApiRequest`]/[`ApiResponse`] payloads at all. A server `Hello` without an `api/` feature is a
/// pre-v23 daemon; clients treat that as incompatible (the stale-managed-daemon connect gate).
pub const WIRE_FEATURE_API_PREFIX: &str = "api/";

/// The `api/<N>` feature string this build advertises (see [`WIRE_FEATURE_API_PREFIX`]).
pub fn wire_feature_api() -> String {
    format!("{WIRE_FEATURE_API_PREFIX}{}", crate::API_WIRE_VERSION.0)
}

// The wire page bound + clamp live in `daemon-common` (so producers below this crate — e.g.
// `daemon-core`'s `ConvView` projection — can honor them); re-exported here so every existing
// `daemon_api::{WIRE_PAGE_MAX, clamp_page_max}` call site keeps working.
pub use daemon_common::{clamp_page_max, WIRE_PAGE_MAX};

/// One page of a paginated list op, bounded at [`WIRE_PAGE_MAX`] items. The uniform envelope for
/// every list that can exceed the wire bound: the CDDL side is a per-payload rule of the shape
/// `x-page = { "items": [0*64 x], ? "next": (tstr / null) }`, and the request carries a matching
/// optional `after` resume cursor (`? "after": (tstr / null)`).
///
/// `next` is the cursor key of the last served item when more remain (`None` => last page); pass
/// it back as the request's `after` to resume. Produced by [`paginate`] (or by a bespoke slicer
/// with equivalent semantics, e.g. `WorkspaceFs::list`, whose composite sort order needs a
/// comparator-aware resume).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WirePage<T> {
    /// The items in this page (at most [`WIRE_PAGE_MAX`]).
    pub items: Vec<T>,
    /// The resume cursor when more items remain (`None` => last page).
    #[serde(default)]
    pub next: Option<String>,
}

// Manual impl: `derive(Default)` would demand `T: Default`, which the empty page does not need.
impl<T> Default for WirePage<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            next: None,
        }
    }
}

/// Slice one page out of a full, **key-ascending-sorted** listing — the shared pagination the
/// converted list ops run on. `key` must be the cursor key AND the sort key (handlers sort by it
/// before calling); `limit` is clamped to [`WIRE_PAGE_MAX`] (`0` => `WIRE_PAGE_MAX`).
///
/// Resume semantics (the ones proven in `WorkspaceFs::list` / `paginate_roster`): start PAST the
/// exact-match `after` item (the normal case); a cursor whose item vanished between pages falls
/// back to the first item whose key sorts strictly greater under the same ascending order, so
/// nothing already served is re-served. `next` is the last served item's key iff items remain.
pub fn paginate<T>(
    mut items: Vec<T>,
    after: Option<&str>,
    limit: usize,
    key: impl Fn(&T) -> String,
) -> WirePage<T> {
    let limit = if limit == 0 {
        WIRE_PAGE_MAX
    } else {
        limit.min(WIRE_PAGE_MAX)
    };
    let start = match after {
        None => 0,
        Some(after) => match items.iter().position(|item| key(item) == after) {
            Some(idx) => idx + 1,
            None => items.partition_point(|item| key(item).as_str() <= after),
        },
    };
    let mut page: Vec<T> = items.split_off(start.min(items.len()));
    let next = if page.len() > limit {
        page.truncate(limit);
        page.last().map(&key)
    } else {
        None
    };
    WirePage { items: page, next }
}

/// The authenticated principal as surfaced to a client on [`WireS2C::AuthOk`] (and the future
/// `WhoAmI` admin op): the user identity plus its effective role/capability names. This is the wire
/// mirror of `daemon_auth::Principal` (`roles`/`capabilities` are the model's stable snake_case
/// strings). Advisory, for client-side UI gating only — the node independently enforces every
/// capability server-side; a client must never trust this in lieu of the server's own checks.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrincipalView {
    /// Stable opaque user id.
    pub user_id: String,
    /// Human-facing username.
    pub username: String,
    /// Assigned role names (snake_case, e.g. `"operator"`).
    pub roles: Vec<String>,
    /// Effective capability names (snake_case, e.g. `"session_write"`).
    pub capabilities: Vec<String>,
}

/// A persisted user record as surfaced over the admin access-control surface (`AccessControlApi`).
/// The wire mirror of `daemon_auth::UserRecord` enriched with the user's resolved role names. Carries
/// **no** credential material (no password, PHC, or token).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessUser {
    /// Stable opaque user id.
    pub user_id: String,
    /// Unique username.
    pub username: String,
    /// Whether an admin has disabled the account.
    pub disabled: bool,
    /// Unix seconds at creation.
    pub created_at: i64,
    /// The user's assigned role names (snake_case).
    pub roles: Vec<String>,
}

/// A built-in role and the effective capabilities it grants, surfaced by `role_list` so an admin UI
/// can render the role→capability matrix. The wire mirror of `daemon_auth::Role` + its
/// `capabilities()` (snake_case capability names).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleInfo {
    /// The role name (snake_case, e.g. `"operator"`).
    pub role: String,
    /// The capability names the role grants (snake_case).
    pub capabilities: Vec<String>,
}

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
    /// Begin a SASL authentication exchange with the named mechanism (e.g. `"SCRAM-SHA-256"`,
    /// `"PLAIN"`, `"EXTERNAL"`), carrying that mechanism's optional initial response. The server
    /// answers with [`WireS2C::AuthChallenge`] (more steps), [`WireS2C::AuthOk`], or
    /// [`WireS2C::AuthError`]. `initial` is opaque mechanism bytes.
    AuthStart {
        /// The chosen mechanism name (must be one the server advertised in its `Hello`).
        mechanism: String,
        /// The mechanism's initial client response (may be empty).
        initial: Vec<u8>,
    },
    /// A subsequent client response in a multi-step mechanism, answering a prior `AuthChallenge`.
    AuthStep {
        /// Opaque mechanism bytes.
        data: Vec<u8>,
    },
    /// Re-authenticate a reconnecting client by presenting a previously issued opaque session
    /// token (the fast path that skips the full mechanism exchange). Answered by `AuthOk`/`AuthError`.
    AuthResume {
        /// The opaque server-issued session token from a prior `AuthOk`.
        token: String,
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
        /// SASL mechanisms the server offers, in preference order (e.g. `["SCRAM-SHA-256",
        /// "EXTERNAL", "PLAIN"]`). Empty when the server does not advertise [`WIRE_FEATURE_AUTH`]
        /// (an unauthenticated/local-trust node), so an older client still negotiates cleanly.
        auth_mechanisms: Vec<String>,
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
    /// A server challenge in a multi-step authentication mechanism; the client replies with
    /// [`WireC2S::AuthStep`]. Opaque mechanism bytes.
    AuthChallenge {
        /// Opaque mechanism bytes.
        data: Vec<u8>,
    },
    /// Authentication succeeded: the connection is now bound to `principal`, and `token` is an
    /// opaque session token the client may present via [`WireC2S::AuthResume`] on reconnect.
    AuthOk {
        /// The opaque server-issued session token (store it, never the password).
        token: String,
        /// The authenticated principal and its effective capabilities (for client-side UI gating;
        /// the server independently enforces).
        principal: PrincipalView,
    },
    /// Authentication did not succeed (wrong password, no such or disabled account, or an
    /// unsupported mechanism). The `reason` is deliberately coarse to avoid an account-probing oracle.
    AuthError {
        /// A short, non-revealing failure reason.
        reason: String,
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

impl ApiRequest {
    /// The size, in bytes, of any inline byte payload this decoded request carries — used by the
    /// Cluster-F ingress governor's post-decode ("max decoded size") check. O(1): it reads the
    /// length of a `Vec<u8>` already decoded in place, never re-encoding or re-allocating. Only
    /// [`ApiRequest::BlobPut`] carries inline bytes today (a `serde_bytes` byte string); every other
    /// variant carries no bulk inline payload and returns `0`.
    pub fn ingress_payload_len(&self) -> usize {
        match self {
            ApiRequest::BlobPut { bytes } => bytes.len(),
            _ => 0,
        }
    }
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

/// One page of a directory listing (fs_list) — the uniform [`WirePage`] envelope over
/// [`FsEntry`]. The listing's stable order (dirs first, then case-insensitive name) defines the
/// cursor: `next` is the last entry's `path`; pass it back as [`ApiRequest::FsList::after`] to
/// resume. (The slicing stays bespoke in `WorkspaceFs::list`: the composite sort key is NOT the
/// `path` cursor key, so the generic [`paginate`] key-resume would mis-place a deleted cursor.)
pub type FsListPage = WirePage<FsEntry>;

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
    /// The caller has not authenticated (no principal bound to the connection), so the node refuses
    /// the operation. Fail-closed: the absence of identity never implies access.
    #[error("unauthenticated: {0}")]
    Unauthenticated(String),
    /// The caller is authenticated but lacks the capability (or resource ownership) the operation
    /// requires.
    #[error("forbidden: {0}")]
    Forbidden(String),
    /// Any other failure.
    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod pagination_tests {
    use super::*;

    /// A key-sorted item list `["k000", "k001", ...)`.
    fn items(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("k{i:03}")).collect()
    }

    fn page(items: Vec<String>, after: Option<&str>, limit: usize) -> WirePage<String> {
        paginate(items, after, limit, |s| s.clone())
    }

    #[test]
    fn empty_listing_yields_empty_last_page() {
        let p = page(Vec::new(), None, 10);
        assert!(p.items.is_empty());
        assert_eq!(p.next, None);
    }

    #[test]
    fn exactly_limit_is_one_final_page() {
        let p = page(items(10), None, 10);
        assert_eq!(p.items.len(), 10);
        assert_eq!(p.next, None, "a full-but-final page carries no cursor");
    }

    #[test]
    fn limit_plus_one_truncates_and_sets_cursor() {
        let p = page(items(11), None, 10);
        assert_eq!(p.items.len(), 10);
        assert_eq!(p.next.as_deref(), Some("k009"));
    }

    #[test]
    fn consecutive_pages_never_overlap_and_cover_everything() {
        let all = items(25);
        let mut served = Vec::new();
        let mut after: Option<String> = None;
        loop {
            let p = page(all.clone(), after.as_deref(), 10);
            served.extend(p.items);
            match p.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
        assert_eq!(served, all, "pages chain without dedup or gaps");
    }

    #[test]
    fn deleted_cursor_resumes_at_first_strictly_greater_key() {
        // The cursor "k004x" no longer exists; resume must serve k005.. (never re-serve <= cursor).
        let p = page(items(10), Some("k004x"), 4);
        assert_eq!(p.items, vec!["k005", "k006", "k007", "k008"]);
        assert_eq!(p.next.as_deref(), Some("k008"));
    }

    #[test]
    fn cursor_past_the_end_yields_empty_last_page() {
        let p = page(items(3), Some("zzz"), 10);
        assert!(p.items.is_empty());
        assert_eq!(p.next, None);
    }

    /// The client page loops re-issue while `next` is non-empty, so a `next` that echoes the
    /// request's own `after` would spin a well-behaved client forever (the same page served over
    /// and over). With unique keys — every converted handler's cursor is a unique id/path — the
    /// resume always advances past the cursor, whether it still exists or has vanished.
    #[test]
    fn next_never_echoes_the_requests_after() {
        let all = items(2 * WIRE_PAGE_MAX + 7);
        for limit in [1, 2, WIRE_PAGE_MAX] {
            // Every live key as the cursor, plus vanished cursors (before/between/past the keys).
            let cursors =
                all.iter()
                    .cloned()
                    .chain(["".into(), "k".into(), "k010x".into(), "zzz".into()]);
            for after in cursors {
                let p = page(all.clone(), Some(&after), limit);
                assert_ne!(
                    p.next.as_deref(),
                    Some(after.as_str()),
                    "next must advance past after={after:?} (limit {limit})"
                );
            }
        }
    }

    #[test]
    fn limit_is_clamped_to_wire_page_max() {
        let p = page(items(WIRE_PAGE_MAX + 10), None, WIRE_PAGE_MAX + 10);
        assert_eq!(p.items.len(), WIRE_PAGE_MAX);
        assert!(p.next.is_some());
    }

    #[test]
    fn zero_limit_means_wire_page_max() {
        let p = page(items(WIRE_PAGE_MAX + 1), None, 0);
        assert_eq!(p.items.len(), WIRE_PAGE_MAX);
        assert_eq!(p.next.as_deref(), Some("k063"));
    }

    #[test]
    fn wire_page_round_trips_and_defaults() {
        let p = WirePage {
            items: vec!["a".to_string()],
            next: Some("a".into()),
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&p, &mut bytes).unwrap();
        let back: WirePage<String> = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(back, p);
        assert_eq!(WirePage::<String>::default().items, Vec::<String>::new());
        assert_eq!(WirePage::<String>::default().next, None);
    }
}

#[cfg(test)]
mod auth_contract_tests {
    use super::*;

    fn rt_c2s(frame: &WireC2S) {
        let mut bytes = Vec::new();
        ciborium::into_writer(frame, &mut bytes).expect("encode WireC2S");
        let decoded: WireC2S = ciborium::from_reader(&bytes[..]).expect("decode WireC2S");
        assert_eq!(&decoded, frame);
    }

    fn rt_s2c(frame: &WireS2C) {
        let mut bytes = Vec::new();
        ciborium::into_writer(frame, &mut bytes).expect("encode WireS2C");
        let decoded: WireS2C = ciborium::from_reader(&bytes[..]).expect("decode WireS2C");
        assert_eq!(&decoded, frame);
    }

    #[test]
    fn wire_version_is_two() {
        assert_eq!(WIRE_VERSION, 2);
    }

    /// The `api/<N>` feature string is formatted from the API mirror version (never hardcoded)
    /// and parses back to it through the public prefix.
    #[test]
    fn api_feature_carries_current_contract_version() {
        let feature = wire_feature_api();
        assert_eq!(
            feature,
            format!("api/{}", daemon_common::WireVersion::CURRENT.0)
        );
        let n: u16 = feature
            .strip_prefix(WIRE_FEATURE_API_PREFIX)
            .expect("api feature carries the prefix")
            .parse()
            .expect("api feature suffix is the numeric contract version");
        assert_eq!(daemon_common::WireVersion(n), crate::API_WIRE_VERSION);
    }

    #[test]
    fn client_auth_frames_round_trip() {
        rt_c2s(&WireC2S::AuthStart {
            mechanism: "SCRAM-SHA-256".into(),
            initial: vec![1, 2, 3],
        });
        rt_c2s(&WireC2S::AuthStep { data: Vec::new() });
        rt_c2s(&WireC2S::AuthResume {
            token: "session-token".into(),
        });
    }

    #[test]
    fn server_auth_frames_round_trip() {
        rt_s2c(&WireS2C::AuthChallenge {
            data: vec![9, 8, 7],
        });
        rt_s2c(&WireS2C::AuthOk {
            token: "tok".into(),
            principal: PrincipalView {
                user_id: "u1".into(),
                username: "alice".into(),
                roles: vec!["user".into()],
                capabilities: vec!["session_write".into()],
            },
        });
        rt_s2c(&WireS2C::AuthError {
            reason: "invalid credentials".into(),
        });
    }

    #[test]
    fn server_hello_carries_auth_mechanisms() {
        let hello = WireS2C::Hello {
            wire_version: WIRE_VERSION,
            features: vec![WIRE_FEATURE_MUX.into(), WIRE_FEATURE_AUTH.into()],
            auth_mechanisms: vec!["SCRAM-SHA-256".into(), "PLAIN".into()],
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&hello, &mut bytes).unwrap();
        let decoded: WireS2C = ciborium::from_reader(&bytes[..]).unwrap();
        match decoded {
            WireS2C::Hello {
                auth_mechanisms, ..
            } => assert_eq!(auth_mechanisms, vec!["SCRAM-SHA-256", "PLAIN"]),
            other => panic!("expected Hello, got {other:?}"),
        }
    }

    #[test]
    fn access_control_frames_round_trip() {
        fn rt_req(req: &ApiRequest) {
            let mut bytes = Vec::new();
            ciborium::into_writer(req, &mut bytes).expect("encode ApiRequest");
            let decoded: ApiRequest = ciborium::from_reader(&bytes[..]).expect("decode ApiRequest");
            assert_eq!(&decoded, req);
        }
        fn rt_res(res: &ApiResponse) {
            let mut bytes = Vec::new();
            ciborium::into_writer(res, &mut bytes).expect("encode ApiResponse");
            let decoded: ApiResponse =
                ciborium::from_reader(&bytes[..]).expect("decode ApiResponse");
            assert_eq!(&decoded, res);
        }
        rt_req(&ApiRequest::UserCreate {
            username: "alice".into(),
            password: "s3cret".into(),
            roles: vec!["user".into()],
        });
        rt_req(&ApiRequest::UserList);
        rt_req(&ApiRequest::UserDisable {
            user_id: "u1".into(),
            disabled: true,
        });
        rt_req(&ApiRequest::UserSetRoles {
            user_id: "u1".into(),
            roles: vec!["operator".into()],
        });
        rt_req(&ApiRequest::UserSetPassword {
            user_id: "u1".into(),
            password: "next".into(),
        });
        rt_req(&ApiRequest::RoleList);
        rt_req(&ApiRequest::WhoAmI);
        rt_req(&ApiRequest::SessionRevoke {
            user_id: "u1".into(),
        });
        rt_req(&ApiRequest::ResourceGrantCreate {
            user_id: "u1".into(),
            resource_kind: "session".into(),
            resource_id: "s1".into(),
            capability: "session_read".into(),
        });
        rt_req(&ApiRequest::ResourceGrantList { user_id: None });
        rt_req(&ApiRequest::ResourceGrantRevoke { id: "g1".into() });

        rt_res(&ApiResponse::AccessUser(AccessUser {
            user_id: "u1".into(),
            username: "alice".into(),
            disabled: false,
            created_at: 42,
            roles: vec!["user".into()],
        }));
        rt_res(&ApiResponse::AccessUsers(Vec::new()));
        rt_res(&ApiResponse::AccessRoles(vec![RoleInfo {
            role: "admin".into(),
            capabilities: vec!["access_admin".into()],
        }]));
        rt_res(&ApiResponse::WhoAmI(PrincipalView {
            user_id: "u1".into(),
            username: "alice".into(),
            roles: vec!["admin".into()],
            capabilities: vec!["access_admin".into()],
        }));
    }

    /// The multi-step interactive-auth wire surface (wire v31): the `AuthStep` request + `AuthStepped`
    /// response, every `AuthChallenge` / `AuthStepInput` / `AuthStepResult` arm, the reshaped
    /// `AuthBeginResponse`, and the extended `AuthFlowKind`, all round-trip through ciborium.
    #[test]
    fn interactive_auth_multistep_round_trips() {
        fn rt_req(req: &ApiRequest) {
            let mut bytes = Vec::new();
            ciborium::into_writer(req, &mut bytes).expect("encode ApiRequest");
            let decoded: ApiRequest = ciborium::from_reader(&bytes[..]).expect("decode ApiRequest");
            assert_eq!(&decoded, req);
        }
        fn rt_res(res: &ApiResponse) {
            let mut bytes = Vec::new();
            ciborium::into_writer(res, &mut bytes).expect("encode ApiResponse");
            let decoded: ApiResponse =
                ciborium::from_reader(&bytes[..]).expect("decode ApiResponse");
            assert_eq!(&decoded, res);
        }

        // Every AuthFlowKind variant (the 4 new ones + the 2 originals).
        for kind in [
            AuthFlowKind::MatrixSso,
            AuthFlowKind::OAuth2Pkce,
            AuthFlowKind::BotToken,
            AuthFlowKind::UserToken,
            AuthFlowKind::PhoneOtp,
            AuthFlowKind::QrPairing,
        ] {
            let mut bytes = Vec::new();
            ciborium::into_writer(&kind, &mut bytes).expect("encode AuthFlowKind");
            let back: AuthFlowKind = ciborium::from_reader(&bytes[..]).expect("decode");
            assert_eq!(back, kind);
        }

        // Every AuthChallenge arm inside a reshaped AuthBeginResponse.
        let challenges = vec![
            AuthChallenge::Redirect {
                authorization_url: "https://idp.example/authorize?x=1".into(),
            },
            AuthChallenge::Form {
                title: "Enter the code".into(),
                fields: vec![AuthParamField {
                    key: "otp".into(),
                    label: "One-time code".into(),
                    required: true,
                }],
            },
            AuthChallenge::Qr {
                payload: "wa://link?tok=abc".into(),
                image: Some(vec![0x89, 0x50, 0x4e, 0x47]),
                poll_interval_ms: 2000,
            },
            AuthChallenge::Qr {
                payload: "sig://link".into(),
                image: None,
                poll_interval_ms: 1500,
            },
            AuthChallenge::Message {
                text: "Approve on your other device".into(),
            },
        ];
        for challenge in &challenges {
            rt_res(&ApiResponse::AuthBegun(AuthBeginResponse {
                flow_id: "flow-1".into(),
                challenge: challenge.clone(),
                expires_at: 1_700_000_000,
            }));
        }

        // Every AuthStepInput arm on the AuthStep request.
        let inputs = vec![
            AuthStepInput::Fields(BTreeMap::from([(
                "phone".to_string(),
                "+15551234".to_string(),
            )])),
            AuthStepInput::Callback("https://cb.example/?code=xyz&state=s".into()),
            AuthStepInput::Poll,
        ];
        for input in inputs {
            rt_req(&ApiRequest::AuthStep(AuthStepRequest {
                flow_id: "flow-1".into(),
                input,
            }));
        }

        // Both AuthStepResult arms on the AuthStepped response.
        rt_res(&ApiResponse::AuthStepped(AuthStepResult::Challenge(
            challenges[1].clone(),
        )));
        rt_res(&ApiResponse::AuthStepped(AuthStepResult::Completed(
            AuthCompleteResponse {
                credential_ref: "matrix/@bot:hs.org".into(),
                account_label: "@bot:hs.org".into(),
                transport_instance: TransportId::new("matrix/@bot:hs.org"),
                bound_profile: Some(ProfileRef::new("default")),
            },
        )));
    }

    #[test]
    fn feedback_frames_round_trip() {
        fn rt_req(req: &ApiRequest) {
            let mut bytes = Vec::new();
            ciborium::into_writer(req, &mut bytes).expect("encode ApiRequest");
            let decoded: ApiRequest = ciborium::from_reader(&bytes[..]).expect("decode ApiRequest");
            assert_eq!(&decoded, req);
        }
        fn rt_res(res: &ApiResponse) {
            let mut bytes = Vec::new();
            ciborium::into_writer(res, &mut bytes).expect("encode ApiResponse");
            let decoded: ApiResponse =
                ciborium::from_reader(&bytes[..]).expect("decode ApiResponse");
            assert_eq!(&decoded, res);
        }
        rt_req(&ApiRequest::FeedbackSubmit {
            kind: FeedbackKind::Response,
            target: Some(FeedbackTarget {
                session: "s1".into(),
                cursor: 7,
                trace: Some(TraceId(0xabc)),
            }),
            rating: Some(FeedbackRating::Down),
            comment: Some("could be better".into()),
            include_content: true,
            diagnostics: Some(FeedbackDiagnostics {
                app_version: Some("2.0".into()),
                os: None,
            }),
            surface: "transcript".into(),
        });
        // App feedback: no target, comment-only.
        rt_req(&ApiRequest::FeedbackSubmit {
            kind: FeedbackKind::App,
            target: None,
            rating: None,
            comment: Some("great app".into()),
            include_content: false,
            diagnostics: None,
            surface: "settings".into(),
        });
        rt_req(&ApiRequest::TelemetryConsentGet);
        rt_req(&ApiRequest::TelemetryConsentSet { enabled: true });
        rt_res(&ApiResponse::FeedbackAck(FeedbackAck {
            accepted: true,
            queued: true,
        }));
        rt_res(&ApiResponse::TelemetryConsent { enabled: false });
    }

    #[test]
    fn new_api_error_variants_round_trip() {
        for err in [
            ApiError::Unauthenticated("no principal".into()),
            ApiError::Forbidden("missing capability".into()),
        ] {
            let mut bytes = Vec::new();
            ciborium::into_writer(&err, &mut bytes).unwrap();
            let decoded: ApiError = ciborium::from_reader(&bytes[..]).unwrap();
            assert_eq!(decoded, err);
        }
    }
}
