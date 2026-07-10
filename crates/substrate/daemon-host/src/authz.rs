// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The per-request capability gate (authz core, Auth 2).
//!
//! [`required_capability`] maps every [`ApiRequest`] variant to the single coarse [`Capability`]
//! that gates it, as ONE exhaustive `match` with **no `_` arm** — adding a request variant without
//! a mapping is a compile error (the build-time exhaustiveness guard). The arms are organized to
//! mirror the `serve_*` fan-out in `daemon-api`'s `dispatch`.
//!
//! [`authorize`] reads the task-local [`Principal`](daemon_auth::Principal) bound by
//! [`with_request_context`](crate::request_context::with_request_context) and enforces the gate:
//! no principal → [`ApiError::Unauthenticated`] (fail-closed), authenticated-but-missing-the-cap →
//! [`ApiError::Forbidden`].
//!
//! This is the *coarse* half of the two-step model: it answers "may this caller perform this *kind*
//! of operation at all?". The *per-resource* half — "may they touch *this* session?" — is the
//! ownership check enforced later by the session layer (Track C), with
//! [`Capability::SessionSeeAll`](daemon_auth::Capability::SessionSeeAll) /
//! [`SessionControlAny`](daemon_auth::Capability::SessionControlAny) as the operator overrides. No
//! variant maps to those override caps here; they gate cross-owner access downstream, not entry.

use crate::request_context::current_principal;
use daemon_api::{ApiError, ApiRequest};
use daemon_auth::Capability;

/// What a request requires to be authorized. Most ops require a specific [`Capability`]; a few
/// (e.g. [`ApiRequest::WhoAmI`]) require only that *some* principal is authenticated, with no
/// particular capability. Modeling the latter explicitly (rather than mapping it to a nominal
/// capability) keeps the gate honest: "any authenticated principal" is a distinct requirement from
/// "holds capability X".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequiredAccess {
    /// Allowed for any authenticated principal (no specific capability). Still fail-closed: an
    /// unauthenticated caller (no bound principal) is rejected.
    Authenticated,
    /// Requires the named [`Capability`].
    Cap(Capability),
}

/// The [`RequiredAccess`] gating `req`. ONE exhaustive match, NO `_` arm: a new [`ApiRequest`]
/// variant that lands without a mapping breaks the build (and thus the test suite). Arms are grouped
/// to parallel the `serve_*` dispatch surfaces.
pub fn required_capability(req: &ApiRequest) -> RequiredAccess {
    // `Capability` is aliased (`C`) because two of its variant names collide with `ApiRequest`'s
    // (`FsRead`/`FsWrite`): bare names in patterns resolve to `ApiRequest`; capabilities are `C::*`.
    use ApiRequest::*;
    use Capability as C;
    let cap = match req {
        // -- serve_session: own-session interaction --------------------------------------------
        Submit { .. }
        | SubmitRouted { .. }
        | SessionCreate { .. }
        | Respond { .. }
        | Handover { .. }
        | RecordMeta(_)
        | SetSessionModel { .. }
        | SetSessionMode { .. }
        | SetSessionOverlay { .. } => C::SessionWrite,
        Poll { .. }
        | SessionHistory { .. }
        | Subscribe { .. }
        | DeliveryTargets { .. }
        | DeliverySessions { .. } => C::SessionRead,

        // -- serve_control: node diagnostics + durable lifecycle + roster -----------------------
        Health
        | Stats
        | Telemetry
        | VerifyingKey
        | ApprovalsPending { .. }
        | CheckpointList { .. }
        | EventsSince { .. }
        // rung 3 (api/39): the Bootstrap probe is a node-wide control-plane read (revs + cursor
        // + epoch), the same tier as EventsSince — a cold client anchors its initial sync with it.
        | Bootstrap
        // The notification list is a node-wide control-plane read (wire v37).
        | NotificationList
        // The person/metacontact registry is a node-wide control-plane read (wire v37).
        | PersonList { .. } => C::ControlRead,
        // Durable control-plane lifecycle is operator-level (Assign wakes a durable session).
        Assign { .. } => C::ControlWrite,
        // The roster reads are "one's own sessions"; SeeAll (Track C) widens them cross-owner.
        Sessions
        | SessionsQuery { .. }
        | SessionGet { .. }
        | SessionSearch { .. }
        | SessionRecap { .. }
        | FingerprintList { .. } => C::SessionRead,
        // A user may drive their OWN session's lifecycle (cancel/rewind/checkpoint-rewind/approve/
        // fingerprint-revoke + roster metadata); Track C scopes it to ownership, operators cross
        // via SessionControlAny.
        Cancel { .. }
        | Rewind { .. }
        | CheckpointRewind { .. }
        | ApprovalDecide { .. }
        | FingerprintRevoke { .. }
        | SessionUpdateMeta { .. } => C::SessionWrite,
        // User feedback (N1) is a user-owned write, like an approval decision on one's own session
        // (analogous to ApprovalDecide / SetSessionMode -> SessionWrite): any User may submit it.
        FeedbackSubmit { .. } => C::SessionWrite,
        // Reading the node-owned telemetry consent toggle is a control-plane read; flipping it is a
        // node-wide control-plane write (operator tier), mirroring Telemetry -> ControlRead.
        TelemetryConsentGet => C::ControlRead,
        TelemetryConsentSet { .. } => C::ControlWrite,
        // The node-owned gateway is a resident service: reading its status is a control-plane read;
        // enabling/rebinding it is a node-wide control-plane write (operator tier), mirroring the
        // telemetry-consent toggle above.
        GatewayGet => C::ControlRead,
        GatewaySet { .. } => C::ControlWrite,
        // Saved presences (W2-F) are node-wide shared config (like the gateway/telemetry-consent
        // toggles): listing is a control-plane read (viewer-readable); mutating them is a node-wide
        // control-plane write (operator tier). Not per-owner session state.
        PresenceList => C::ControlRead,
        PresenceSave { .. } | PresenceDelete { .. } | PresenceSetActive { .. } => C::ControlWrite,

        // -- serve_fleet: orchestration tree ----------------------------------------------------
        Fleet
        | Tree { .. }
        | Unit { .. }
        | UnitEvents { .. }
        | UnitOutbound { .. }
        | UnitHistory { .. } => C::FleetRead,
        Pause { .. } | Resume { .. } | Scale { .. } => C::FleetWrite,

        // -- serve_models: model management -----------------------------------------------------
        ModelSearch { .. }
        | ModelFiles { .. }
        | ModelDownloads
        | ModelCatalog
        | ModelRecommend(_)
        | ModelQuantizes
        | ModelInspect { .. }
        | Models { .. }
        | ModelCurrent { .. }
        | ProviderCatalog
        | ProviderModels { .. }
        | CustomProviderList => C::ModelsRead,
        ModelDownload { .. }
        | ModelCancel { .. }
        | ModelPause { .. }
        | ModelResume { .. }
        | ModelDelete { .. }
        | ModelActivate { .. }
        | ModelQuantize(_)
        | CustomProviderSet { .. }
        | CustomProviderRemove { .. } => C::ModelsWrite,

        // -- serve_profile: profiles + skills (versioned) + personas (wire v36) -----------------
        ProfileList
        | ProfileGet { .. }
        | ProfileExport { .. }
        | ProfileHistory { .. }
        | ProfileAt { .. }
        | SoulGet { .. }
        | SkillHistory { .. }
        | SkillAt { .. }
        | SkillGet { .. } => C::ProfileRead,
        ProfileCreate { .. }
        | ProfileUpdate { .. }
        | ProfileDelete { .. }
        | ProfileSelect { .. }
        | ProfileClone { .. }
        | ProfileImport { .. }
        | ProfileRevert { .. }
        | SoulSet { .. }
        | SkillRevert { .. }
        | SkillPut { .. } => C::ProfileWrite,

        // -- serve_curator: per-profile skill library -------------------------------------------
        CuratorList { .. } => C::ProfileRead,
        CuratorPin { .. }
        | CuratorUnpin { .. }
        | CuratorArchive { .. }
        | CuratorRestore { .. }
        | CuratorRun { .. } => C::ProfileWrite,

        // -- serve_auth: interactive (OAuth) flows + credential store ---------------------------
        AuthProviders | CredentialList => C::CredentialRead,
        AuthBegin(_)
        | AuthStep(_)
        | AuthComplete(_)
        | AuthCancel { .. }
        | CredentialSet { .. }
        | CredentialRemove { .. }
        | CredentialSetLabel { .. } => C::CredentialWrite,

        // -- serve_cron: scheduled jobs ---------------------------------------------------------
        CronList | CronRuns { .. } | CronSuggestions => C::CronRead,
        CronCreate { .. }
        | CronUpdate { .. }
        | CronDelete { .. }
        | CronTrigger { .. }
        | CronPause { .. }
        | CronAcceptSuggestion { .. }
        | CronDismissSuggestion { .. } => C::CronWrite,

        // -- serve_routing: chat routing + transport registry -----------------------------------
        RoutingListChats { .. }
        | RoutingGet { .. }
        | TransportRooms { .. }
        | TransportAdapters
        | TransportInstances
        // Reading an instance's persisted non-secret settings (wire v38) is a registry read.
        | TransportSettings { .. } => C::RoutingRead,
        RoutingSet { .. }
        | RoutingBindChat { .. }
        | RoutingUnbindChat { .. }
        | TransportDisconnect { .. }
        | TransportRemove { .. }
        | TransportConnect { .. }
        | TransportSetEnabled { .. }
        | TransportSetLabel { .. }
        // Editing an instance's settings (wire v38) is an account-management write, like the
        // enabled/label ops above.
        | TransportConfigure { .. } => C::RoutingWrite,

        // -- serve_messaging: conversations, membership, contacts -------------------------------
        ConvList { .. }
        | ConvGet { .. }
        | ConvCreateDetails { .. }
        | ConvJoinDetails { .. }
        | ConvHistory(_)
        | ContactGetProfile { .. }
        | ContactActionMenu { .. }
        | DirectorySearch { .. }
        | RosterList { .. } => C::MessagingRead,
        ConvCreate { .. }
        | ConvJoin { .. }
        | ConvLeave { .. }
        | ConvSend(_)
        | ConvSetTopic { .. }
        | ConvSetTitle { .. }
        | ConvSetDescription { .. }
        | ConvDelete { .. }
        | MemberInvite(_)
        | MemberRemove(_)
        | MemberBan(_)
        | MemberSetRole(_)
        | ContactSetAlias { .. }
        | RosterAdd { .. }
        | RosterUpdate { .. }
        | RosterRemove { .. }
        | FtSend { .. }
        | FtReceive { .. } => C::MessagingWrite,

        // -- serve_registry: extension/agent/provider registry + node config --------------------
        // AgentDiscover only probes recipes and caches in memory (no persistence) -> a read.
        AgentDiscover | AgentCatalog | ProviderList | ToolList | CommandList | ConfigGet | Caps => {
            C::RegistryRead
        }
        // CommandInvoke is a coarse-floor read; the command catalog's own `min_access` (now
        // principal-driven, see `commands::caller_access`) does the per-command Admin-tier gating.
        CommandInvoke { .. } => C::RegistryRead,
        AgentRegister { .. }
        | AgentRemove { .. }
        | ProviderRegister { .. }
        | ToolRegister { .. }
        | ToolSetEnabled { .. }
        | ConfigSet { .. } => C::RegistryWrite,

        // -- serve_fs: filesystem surface + blob store ------------------------------------------
        FsRoots
        | FsList { .. }
        | FsStat { .. }
        | FsRead { .. }
        | FsSearch { .. }
        | FsWatchPoll(_)
        | BlobGet { .. }
        | BlobStat { .. } => C::FsRead,
        FsWrite(_) | BlobPut { .. } | FsWriteFromBlob(_) => C::FsWrite,

        // -- serve_access: admin user/role/session management + reserved per-resource grants ----
        // WhoAmI is allowed to ANY authenticated principal (no specific capability); the `return`
        // diverges, so this arm unifies with the surrounding `Capability`-typed match.
        WhoAmI => return RequiredAccess::Authenticated,
        UserCreate { .. }
        | UserList
        | UserDisable { .. }
        | UserSetRoles { .. }
        | UserSetPassword { .. }
        | RoleList
        | SessionRevoke { .. }
        | ResourceGrantCreate { .. }
        | ResourceGrantList { .. }
        | ResourceGrantRevoke { .. } => C::AccessAdmin,
    };
    RequiredAccess::Cap(cap)
}

/// Authorize `req` against the task-local request principal.
///
/// Fail-closed: with no active [`RequestContext`](crate::request_context::RequestContext) the
/// principal is absent and the request is [`ApiError::Unauthenticated`]. An authenticated caller
/// missing the [`required_capability`] gets [`ApiError::Forbidden`]. Per-resource ownership is a
/// separate, downstream check (Track C).
pub fn authorize(req: &ApiRequest) -> Result<(), ApiError> {
    // Fail-closed: every requirement first needs *a* bound principal.
    let principal = current_principal().ok_or_else(|| {
        ApiError::Unauthenticated("no authenticated principal bound to this request".into())
    })?;
    match required_capability(req) {
        // Any authenticated principal satisfies an `Authenticated` requirement (e.g. `WhoAmI`).
        RequiredAccess::Authenticated => Ok(()),
        RequiredAccess::Cap(cap) if principal.has(cap) => Ok(()),
        RequiredAccess::Cap(cap) => Err(ApiError::Forbidden(format!(
            "operation requires capability {cap:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_context::{with_request_context, RequestContext};
    use daemon_auth::{Principal, Role};
    use daemon_protocol::TransportId;

    fn principal(role: Role) -> Principal {
        Principal::from_roles("u", "u", vec![role])
    }

    /// Run the gate for `req` under a principal holding exactly `role`.
    async fn gate(role: Role, req: ApiRequest) -> Result<(), ApiError> {
        let p = principal(role);
        with_request_context(RequestContext::authenticated(p, None), async move {
            authorize(&req)
        })
        .await
    }

    // ---- a representative request per `serve_*` surface, with its expected capability ----------
    fn read_samples() -> Vec<(ApiRequest, Capability)> {
        vec![
            (ApiRequest::Health, Capability::ControlRead),
            (ApiRequest::Sessions, Capability::SessionRead),
            (ApiRequest::Fleet, Capability::FleetRead),
            (ApiRequest::Models { after: None }, Capability::ModelsRead),
            (ApiRequest::ProviderCatalog, Capability::ModelsRead),
            (
                ApiRequest::ProviderModels {
                    provider: "daemon_cloud".into(),
                    credential_ref: None,
                    transient_key: None,
                    after: None,
                },
                Capability::ModelsRead,
            ),
            (ApiRequest::ProfileList, Capability::ProfileRead),
            // Persona reads (wire v36) are gated exactly like profile reads.
            (
                ApiRequest::SoulGet { id: "p".into() },
                Capability::ProfileRead,
            ),
            (
                ApiRequest::CuratorList { profile: None },
                Capability::ProfileRead,
            ),
            (ApiRequest::CredentialList, Capability::CredentialRead),
            (ApiRequest::CronList, Capability::CronRead),
            (
                ApiRequest::RoutingListChats { after: None },
                Capability::RoutingRead,
            ),
            (
                ApiRequest::ConvList {
                    transport: TransportId::new("t"),
                    after: None,
                    since_rev: None,
                },
                Capability::MessagingRead,
            ),
            (ApiRequest::AgentCatalog, Capability::RegistryRead),
            (ApiRequest::FsRoots, Capability::FsRead),
        ]
    }

    /// User-tier writes: `User` (and up) may; `Viewer` may not.
    fn user_writes() -> Vec<(ApiRequest, Capability)> {
        vec![
            (
                ApiRequest::Cancel {
                    session: "s".into(),
                },
                Capability::SessionWrite,
            ),
            (
                ApiRequest::CheckpointRewind {
                    session: "s".into(),
                    checkpoint_id: "c".into(),
                },
                Capability::SessionWrite,
            ),
            (
                ApiRequest::ProfileDelete { id: "p".into() },
                Capability::ProfileWrite,
            ),
            // Persona writes (wire v36) are gated exactly like profile writes.
            (
                ApiRequest::SoulSet {
                    id: "p".into(),
                    text: "persona".into(),
                },
                Capability::ProfileWrite,
            ),
            (
                ApiRequest::CredentialRemove {
                    profile: "p".into(),
                },
                Capability::CredentialWrite,
            ),
            (
                ApiRequest::CronDelete { id: "c".into() },
                Capability::CronWrite,
            ),
            (
                ApiRequest::ConvLeave {
                    transport: TransportId::new("t"),
                    conv: "c".into(),
                },
                Capability::MessagingWrite,
            ),
            (
                ApiRequest::BlobPut { bytes: Vec::new() },
                Capability::FsWrite,
            ),
        ]
    }

    /// Operator-tier writes: `Operator` (and `Admin`) may; `User` may not.
    fn operator_writes() -> Vec<(ApiRequest, Capability)> {
        vec![
            (
                ApiRequest::Assign {
                    session: "s".into(),
                },
                Capability::ControlWrite,
            ),
            (
                ApiRequest::Pause { unit: "u".into() },
                Capability::FleetWrite,
            ),
            (
                ApiRequest::AgentRemove { name: "a".into() },
                Capability::RegistryWrite,
            ),
            (
                ApiRequest::ModelDelete { id: "m".into() },
                Capability::ModelsWrite,
            ),
        ]
    }

    #[test]
    fn representative_mapping_per_group_is_stable() {
        for (req, cap) in read_samples()
            .into_iter()
            .chain(user_writes())
            .chain(operator_writes())
        {
            assert_eq!(
                required_capability(&req),
                RequiredAccess::Cap(cap),
                "mapping drifted for {req:?}"
            );
        }
    }

    #[tokio::test]
    async fn unauthenticated_request_is_rejected() {
        // No active scope: every surface fails closed with `Unauthenticated`.
        for (req, _) in read_samples() {
            assert!(
                matches!(authorize(&req), Err(ApiError::Unauthenticated(_))),
                "expected Unauthenticated for {req:?}"
            );
        }
    }

    #[tokio::test]
    async fn authenticated_missing_capability_is_forbidden() {
        let err = gate(
            Role::Viewer,
            ApiRequest::Cancel {
                session: "s".into(),
            },
        )
        .await;
        assert!(matches!(err, Err(ApiError::Forbidden(_))));
    }

    #[tokio::test]
    async fn role_matrix_viewer_reads_only() {
        for (req, _) in read_samples() {
            assert!(
                gate(Role::Viewer, req.clone()).await.is_ok(),
                "viewer read {req:?}"
            );
        }
        for (req, _) in user_writes().into_iter().chain(operator_writes()) {
            assert!(
                matches!(
                    gate(Role::Viewer, req.clone()).await,
                    Err(ApiError::Forbidden(_))
                ),
                "viewer must be denied write {req:?}"
            );
        }
    }

    #[tokio::test]
    async fn role_matrix_user_owns_writes_but_no_operator_caps() {
        for (req, _) in read_samples().into_iter().chain(user_writes()) {
            assert!(
                gate(Role::User, req.clone()).await.is_ok(),
                "user allowed {req:?}"
            );
        }
        for (req, _) in operator_writes() {
            assert!(
                matches!(
                    gate(Role::User, req.clone()).await,
                    Err(ApiError::Forbidden(_))
                ),
                "user must be denied operator op {req:?}"
            );
        }
    }

    /// Admin-tier ops (Auth 5 access-control surface): every one maps to `AccessAdmin`.
    fn admin_ops() -> Vec<ApiRequest> {
        vec![
            ApiRequest::UserCreate {
                username: "x".into(),
                password: "x".into(),
                roles: vec!["user".into()],
            },
            ApiRequest::UserList,
            ApiRequest::UserDisable {
                user_id: "u".into(),
                disabled: true,
            },
            ApiRequest::UserSetRoles {
                user_id: "u".into(),
                roles: vec!["user".into()],
            },
            ApiRequest::UserSetPassword {
                user_id: "u".into(),
                password: "x".into(),
            },
            ApiRequest::RoleList,
            ApiRequest::SessionRevoke {
                user_id: "u".into(),
            },
            ApiRequest::ResourceGrantCreate {
                user_id: "u".into(),
                resource_kind: "session".into(),
                resource_id: "s".into(),
                capability: "session_read".into(),
            },
            ApiRequest::ResourceGrantList { user_id: None },
            ApiRequest::ResourceGrantRevoke { id: "g".into() },
        ]
    }

    #[test]
    fn admin_ops_require_access_admin() {
        for req in admin_ops() {
            assert_eq!(
                required_capability(&req),
                RequiredAccess::Cap(Capability::AccessAdmin),
                "admin op {req:?} must require AccessAdmin"
            );
        }
        // WhoAmI is the lone "any authenticated principal" requirement.
        assert_eq!(
            required_capability(&ApiRequest::WhoAmI),
            RequiredAccess::Authenticated
        );
    }

    #[tokio::test]
    async fn role_matrix_operator_drives_node_but_lacks_access_admin() {
        for (req, _) in read_samples()
            .into_iter()
            .chain(user_writes())
            .chain(operator_writes())
        {
            assert!(
                gate(Role::Operator, req.clone()).await.is_ok(),
                "operator allowed {req:?}"
            );
        }
        // The Auth 5 admin ops DO map to AccessAdmin: an Operator (no AccessAdmin) is forbidden.
        assert!(!principal(Role::Operator).has(Capability::AccessAdmin));
        for req in admin_ops() {
            assert!(
                matches!(
                    gate(Role::Operator, req.clone()).await,
                    Err(ApiError::Forbidden(_))
                ),
                "operator must be denied admin op {req:?}"
            );
        }
    }

    #[tokio::test]
    async fn admin_ops_deny_every_non_admin_role() {
        for role in [Role::Viewer, Role::User, Role::Operator] {
            for req in admin_ops() {
                assert!(
                    matches!(gate(role, req.clone()).await, Err(ApiError::Forbidden(_))),
                    "{role:?} must be denied admin op {req:?}"
                );
            }
        }
        // Admin is allowed every admin op.
        for req in admin_ops() {
            assert!(
                gate(Role::Admin, req.clone()).await.is_ok(),
                "admin allowed {req:?}"
            );
        }
    }

    #[tokio::test]
    async fn who_am_i_allowed_for_any_authenticated_principal() {
        // Even the lowest role (Viewer) may call WhoAmI; no capability is required.
        for role in [Role::Viewer, Role::User, Role::Operator, Role::Admin] {
            assert!(
                gate(role, ApiRequest::WhoAmI).await.is_ok(),
                "{role:?} may call WhoAmI"
            );
        }
        // But an unauthenticated caller (no bound principal) is still rejected (fail-closed).
        assert!(matches!(
            authorize(&ApiRequest::WhoAmI),
            Err(ApiError::Unauthenticated(_))
        ));
    }

    #[tokio::test]
    async fn role_matrix_admin_allows_everything_tested() {
        for (req, _) in read_samples()
            .into_iter()
            .chain(user_writes())
            .chain(operator_writes())
        {
            assert!(
                gate(Role::Admin, req.clone()).await.is_ok(),
                "admin allowed {req:?}"
            );
        }
        assert!(principal(Role::Admin).has(Capability::AccessAdmin));
    }

    #[tokio::test]
    async fn system_principal_passes_every_surface() {
        for (req, _) in read_samples()
            .into_iter()
            .chain(user_writes())
            .chain(operator_writes())
        {
            let r = with_request_context(RequestContext::system(), async { authorize(&req) }).await;
            assert!(r.is_ok(), "system principal allowed {req:?}");
        }
    }
}
