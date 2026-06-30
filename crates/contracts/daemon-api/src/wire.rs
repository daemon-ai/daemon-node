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

    // -- access control (Auth 5) -----------------------------------------------------------------
    /// A single user record (user_create).
    AccessUser(AccessUser),
    /// The user listing (user_list).
    AccessUsers(Vec<AccessUser>),
    /// The built-in roles and their effective capabilities (role_list).
    AccessRoles(Vec<RoleInfo>),
    /// The caller's own principal view (who_am_i).
    WhoAmI(PrincipalView),
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
