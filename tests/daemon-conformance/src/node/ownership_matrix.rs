// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Auth 4 ownership **breadth** conformance (Phase 4 `conformance-cddl`): every session-touching
//! `ApiRequest` variant is proven DENIED for a non-owner principal that lacks `SessionSeeAll`,
//! driven through the REAL production request→handler fan-out (`daemon_api::dispatch`).
//!
//! The backbone is a single exhaustive [`classify`] match over `&ApiRequest` with **NO `_` arm**:
//! a future variant that lands without an explicit ownership classification is a COMPILE ERROR, so
//! "the Nth surface ships without the guard" becomes a red build — the anti-"hand-picked few" net.
//! Each variant is classified as:
//!
//! * [`Coverage::OwnerGated`] — session-touching; a non-owner must be denied (the table drives it and
//!   asserts the per-variant deny shape: [`Deny::Forbidden`] for fallible ops, [`Deny::EmptyOrAbsent`]
//!   for the infallible reads that deny by returning nothing — no existence oracle);
//! * [`Coverage::NotSessionTouching`] — its own domain/capability gate; not deny-asserted here.
//!
//! The former F3 (fleet/unit surface) and F4 (node-wide `EventsSince` feed, transport-keyed
//! `DeliverySessions`) `KnownGap` residuals are now fully `OwnerGated`: the fleet/unit reads resolve
//! a unit to its owner via `UnitId -> UnitNode.session -> session owner` (children inherit the
//! delegating parent's owner), the event feed filters its session-bearing variants to the request
//! principal, and `delivery_sessions` filters per-row — so this table now asserts EVERY
//! session-touching variant denies a non-owner, with no `KnownGap` arm left. The dedicated
//! RED→GREEN repro pair lives in `f3f4_ownership.rs`.
//!
//! It also lands the two live cross-owner leaks this net made visible (F1): `approvals_pending` and
//! `checkpoints` had no ownership scope, so any `User` (which holds `ControlRead`) could read another
//! owner's parked-approval prompts/paths or checkpoint metadata by passing that owner's session id —
//! the same class as the Phase 1 auth4 transcript leak. The `f1_*` tests are the bug-repro pair
//! (they fail before the control.rs owner-scope fix); the F2 FS-session-root leak is covered inside
//! the deny table (the `Fs*` samples target a `FsRootId::Session`).

use super::harness::*;
use daemon_api::{
    ApiError, FsRootId, FsSearchQuery, FsWriteArgs, FsWriteFromBlobArgs, RecordMetaArgs,
    RewindPoint, SessionApi, SessionMetaPatch, SessionOverlay, SessionQuery, SessionScope,
};
use daemon_auth::{Principal, Role};
use daemon_common::{ContentHash, Epoch, JobId, ReqId, UnitId};
use daemon_core::{CheckpointStore, LocalCheckpointStore, LocalEnvironment, Snapshot};
use daemon_host::{with_request_context, RequestContext};
use daemon_protocol::{
    AgentCommand, DeliveryTarget, HostResponse, HostResponseBody, Origin, OriginScope,
    RewindAnchor, SinkKind, UserMsg,
};
use daemon_store::{Checkpoint, ParkedApproval};

/// A node retaining its shared durable store (so a test can seed owner-stamped state directly).
fn assemble_with_store() -> (
    Arc<NodeApiImpl>,
    daemon_host::SupervisorHandle,
    Arc<dyn SessionStore>,
) {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 1, [0x4c; 32], fast_host_config());
    (node, handle, store)
}

/// A request context bound to `name` (its own `user_id`) holding exactly `role`.
fn ctx(name: &str, role: Role) -> RequestContext {
    RequestContext::authenticated(Principal::from_roles(name, name, vec![role]), None)
}

/// Alice (a plain `User`) creates + owns a blank session `s` (create stamps her as owner).
async fn alice_owns(node: &Arc<NodeApiImpl>, s: &SessionId) {
    with_request_context(ctx("alice", Role::User), async {
        node.session_create(Some(s.clone()), None).await
    })
    .await
    .expect("alice creates her own session");
}

fn start_turn(text: &str) -> AgentCommand {
    AgentCommand::StartTurn {
        input: UserMsg::new(text),
        request_id: ReqId(1),
    }
}

// ---------------------------------------------------------------------------
// F1 — the two live cross-owner leaks (RED before the control.rs fix, GREEN after)
// ---------------------------------------------------------------------------

/// `approvals_pending` must be owner-scoped: a non-owner peer sees an empty page for another
/// owner's session, while the owner and an operator (`SessionSeeAll`) see the parked approval.
/// BEFORE the fix `approvals_pending` had NO ownership check, so `bob` read `alice`'s approval
/// prompt/path (a cross-owner HITL leak).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn f1_approvals_pending_is_owner_scoped() {
    let (node, handle, store) = assemble_with_store();
    let s = SessionId::new("s-appr");
    alice_owns(&node, &s).await;
    seed_approval(&store, &s).await;

    // A non-owner peer must see NONE of it.
    let bob = with_request_context(ctx("bob", Role::User), async {
        node.approvals_pending(Some(s.clone()), None).await
    })
    .await;
    assert!(
        bob.items.is_empty(),
        "SECURITY: a non-owner read {} of alice's pending approvals",
        bob.items.len()
    );

    // The owner and an operator (SessionSeeAll) DO see it.
    let alice = with_request_context(ctx("alice", Role::User), async {
        node.approvals_pending(Some(s.clone()), None).await
    })
    .await;
    assert_eq!(
        alice.items.len(),
        1,
        "the owner sees her own pending approval"
    );
    let op = with_request_context(ctx("op", Role::Operator), async {
        node.approvals_pending(Some(s.clone()), None).await
    })
    .await;
    assert_eq!(op.items.len(), 1, "an operator (SeeAll) sees any approval");

    handle.shutdown().await;
}

/// `checkpoints` (the `CheckpointList` path) must be owner-scoped: a non-owner peer sees an empty
/// page for another owner's session, while the owner and an operator see the recorded checkpoint.
/// BEFORE the fix `checkpoints` had NO ownership check, so `bob` read `alice`'s checkpoint metadata.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn f1_checkpoint_list_is_owner_scoped() {
    let (node0, handle, _store) = assemble_with_store();
    let cp_dir = tempfile::tempdir().expect("checkpoint dir");
    let cp = Arc::new(LocalCheckpointStore::new(cp_dir.path()));
    // Wire the checkpoint store onto a clone (the harness assembles without one).
    let node = Arc::new((*node0).clone().with_checkpoints(cp.clone()));

    let s = SessionId::new("s-cp");
    alice_owns(&node, &s).await;
    seed_checkpoint(&cp, &s).await;

    // A non-owner peer must see NONE of it.
    let bob = with_request_context(ctx("bob", Role::User), async {
        node.checkpoints(Some(s.clone()), None).await
    })
    .await;
    assert!(
        bob.items.is_empty(),
        "SECURITY: a non-owner read {} of alice's checkpoints",
        bob.items.len()
    );

    // The owner and an operator DO see it.
    let alice = with_request_context(ctx("alice", Role::User), async {
        node.checkpoints(Some(s.clone()), None).await
    })
    .await;
    assert_eq!(alice.items.len(), 1, "the owner sees her own checkpoint");
    let op = with_request_context(ctx("op", Role::Operator), async {
        node.checkpoints(Some(s.clone()), None).await
    })
    .await;
    assert_eq!(
        op.items.len(),
        1,
        "an operator (SeeAll) sees any checkpoint"
    );

    handle.shutdown().await;
}

/// Seed a parked approval on `s` straight through the store (no turn needed).
async fn seed_approval(store: &Arc<dyn SessionStore>, s: &SessionId) {
    let fence = store
        .acquire_activation_lease(s)
        .await
        .expect("activation lease");
    let blob = Snapshot::fresh(s.clone())
        .encode()
        .expect("encode snapshot");
    store
        .park_approval(
            Checkpoint::new(s.clone(), Epoch(1), blob),
            vec![ParkedApproval {
                session_id: s.clone(),
                job_id: JobId::new(format!("{}:1:approval:0", s.as_str())),
                epoch: Epoch(1),
                prompt: "approve write to secret.txt".into(),
                path: Some("secret.txt".into()),
                fingerprint: None,
                decision: None,
            }],
            fence,
        )
        .await
        .expect("park an approval");
}

/// Record a checkpoint on `s` (capture over an empty workspace dir).
async fn seed_checkpoint(cp: &Arc<LocalCheckpointStore>, s: &SessionId) {
    let ws = tempfile::tempdir().expect("workspace dir");
    let env = LocalEnvironment::new(ws.path().to_path_buf());
    cp.capture(s.as_str(), "call-1", "fs_write", &env)
        .await
        .expect("capture a checkpoint");
}

// ---------------------------------------------------------------------------
// The exhaustive classifier (NO `_` arm) + the deny table
// ---------------------------------------------------------------------------

/// The no-leak shape a denied non-owner must observe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Deny {
    /// A fallible op returns `ApiResponse::Error(ApiError::Forbidden(_))`.
    Forbidden,
    /// An infallible read denies by returning nothing of the owner's (empty page / `None` /
    /// roster/tree/roots excludes the session) — no existence oracle.
    EmptyOrAbsent,
}

/// How a variant is covered by this ownership net.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Coverage {
    /// Session-touching: a non-owner MUST be denied with the given shape (driven + asserted).
    OwnerGated(Deny),
    /// Not session-touching: its own domain/capability gate applies; not deny-asserted here.
    NotSessionTouching,
}

/// Classify EVERY `ApiRequest` variant for the Auth 4 ownership net. ONE match, **NO `_` arm**: a
/// new variant that lands without an explicit classification breaks the build (and this suite), so a
/// session-touching surface can never ship without a conscious ownership decision. Arms are grouped
/// to mirror `daemon-api`'s `serve_*` dispatch fan-out.
fn classify(req: &ApiRequest) -> Coverage {
    use ApiRequest::*;
    use Coverage::*;
    use Deny::*;
    match req {
        // -- serve_session: own-session interaction (fallible → Forbidden) ----------------------
        Submit { .. }
        | SubmitRouted { .. }
        | SessionCreate { .. }
        | Poll { .. }
        | Respond { .. }
        | Subscribe { .. }
        | Handover { .. }
        | RecordMeta(_)
        | SetSessionModel { .. }
        | SetSessionMode { .. }
        | SetSessionOverlay { .. } => OwnerGated(Forbidden),
        // Infallible session reads: deny by returning nothing of the owner's. `DeliverySessions`
        // (F4, now owner-scoped): a transport's session list is filtered per-row by ownership, so a
        // non-owner never enumerates another owner's session on a shared transport.
        SessionHistory { .. } | DeliveryTargets { .. } | DeliverySessions { .. } => {
            OwnerGated(EmptyOrAbsent)
        }

        // -- serve_control: roster / read-of-one / durable lifecycle ----------------------------
        Sessions
        | SessionsQuery { .. }
        | SessionGet { .. }
        | SessionSearch { .. }
        | SessionRecap { .. } => OwnerGated(EmptyOrAbsent),
        ApprovalsPending { .. } | CheckpointList { .. } => OwnerGated(EmptyOrAbsent), // F1 (fixed)
        Assign { .. }
        | Cancel { .. }
        | ApprovalDecide { .. }
        | FingerprintList { .. }
        | FingerprintRevoke { .. }
        | CheckpointRewind { .. }
        | SessionUpdateMeta { .. }
        | Rewind { .. } => OwnerGated(Forbidden),
        Health | Stats | Telemetry | VerifyingKey => NotSessionTouching,
        // Node-wide L3 event feed (F4, now owner-scoped): the session-bearing events
        // (SessionAdvanced/SessionMetaChanged/ApprovalPending) are filtered to the request
        // principal; the payload-free node-wide pointers pass (the refetch they nudge is scoped).
        EventsSince { .. } => OwnerGated(EmptyOrAbsent),

        // -- serve_fleet: the orchestration tree AND the flat unit surface are owner-scoped (F3, now
        // gated): a unit maps to its owner via UnitId -> UnitNode.session -> session owner (children
        // inherit the delegating parent's owner at the seam), so an owned subtree resolves whole and
        // a foreign one is denied whole. A sessionless/unknown unit is operator-only (fail-closed).
        Tree { .. }
        | Fleet
        | Unit { .. }
        | UnitEvents { .. }
        | UnitOutbound { .. }
        | UnitHistory { .. } => OwnerGated(EmptyOrAbsent),
        Pause { .. } | Resume { .. } | Scale { .. } => NotSessionTouching,

        // -- serve_fs: a `FsRootId::Session` root addresses a session sandbox (owner-gated) -------
        FsRoots => OwnerGated(EmptyOrAbsent),
        FsList { .. }
        | FsStat { .. }
        | FsRead { .. }
        | FsSearch { .. }
        | FsWatchPoll(_)
        | FsWrite(_)
        | FsWriteFromBlob(_) => OwnerGated(Forbidden),
        // Content-addressed blob store: not session-scoped.
        BlobPut { .. } | BlobGet { .. } | BlobStat { .. } => NotSessionTouching,

        // -- everything else: its own domain + capability gate (not per-owner session ownership) -
        ModelSearch { .. }
        | ModelFiles { .. }
        | ModelDownload { .. }
        | ModelDownloads
        | ModelCancel { .. }
        | ModelPause { .. }
        | ModelResume { .. }
        | ModelCatalog
        | ModelDelete { .. }
        | ModelActivate { .. }
        | ModelRecommend(_)
        | ModelQuantize(_)
        | ModelQuantizes
        | ModelInspect { .. }
        | Models { .. }
        | ModelCurrent { .. }
        | ProviderCatalog
        | ProviderModels { .. }
        | CustomProviderList
        | CustomProviderSet { .. }
        | CustomProviderRemove { .. } => NotSessionTouching,
        ProfileList
        | ProfileGet { .. }
        | ProfileCreate { .. }
        | ProfileUpdate { .. }
        | ProfileDelete { .. }
        | ProfileSelect { .. }
        | ProfileClone { .. }
        | ProfileExport { .. }
        | ProfileImport { .. }
        | ProfileHistory { .. }
        | ProfileAt { .. }
        | ProfileRevert { .. }
        // Persona ops (wire v36) are profile-scoped like every other profile op: gated by
        // ProfileRead/ProfileWrite capability, not per-owner session ownership.
        | SoulGet { .. }
        | SoulSet { .. }
        | SkillHistory { .. }
        | SkillAt { .. }
        | SkillRevert { .. }
        | SkillGet { .. }
        | SkillPut { .. } => NotSessionTouching,
        CuratorList { .. }
        | CuratorPin { .. }
        | CuratorUnpin { .. }
        | CuratorArchive { .. }
        | CuratorRestore { .. }
        | CuratorRun { .. } => NotSessionTouching,
        AuthProviders
        | AuthBegin(_)
        | AuthStep(_)
        | AuthComplete(_)
        | AuthCancel { .. }
        | CredentialSet { .. }
        | CredentialList
        | CredentialRemove { .. }
        | CredentialSetLabel { .. } => NotSessionTouching,
        CronList
        | CronCreate { .. }
        | CronUpdate { .. }
        | CronDelete { .. }
        | CronTrigger { .. }
        | CronRuns { .. }
        | CronPause { .. }
        | CronSuggestions
        | CronAcceptSuggestion { .. }
        | CronDismissSuggestion { .. } => NotSessionTouching,
        // `RoutingBindChat` carries a session id but is `RoutingWrite` (operator-tier): the wire
        // capability gate blocks a non-operator before dispatch; it is not per-owner session-gated.
        RoutingListChats { .. }
        | RoutingGet { .. }
        | RoutingSet { .. }
        | RoutingBindChat { .. }
        | RoutingUnbindChat { .. }
        | TransportRooms { .. }
        | TransportAdapters
        | TransportInstances
        | TransportDisconnect { .. }
        | TransportRemove { .. }
        | TransportConnect { .. }
        | TransportSetEnabled { .. }
        | TransportSetLabel { .. }
        // The account-settings read + merge-edit (wire vNEXT) are RoutingRead/RoutingWrite
        // capability-gated like the other transport account-management ops; not session-scoped.
        | TransportSettings { .. }
        | TransportConfigure { .. } => NotSessionTouching,
        ConvList { .. }
        | ConvGet { .. }
        | ConvCreateDetails { .. }
        | ConvCreate { .. }
        | ConvJoinDetails { .. }
        | ConvJoin { .. }
        | ConvLeave { .. }
        | ConvSend(_)
        | ConvSetTopic { .. }
        | ConvSetTitle { .. }
        | ConvSetDescription { .. }
        | ConvDelete { .. }
        | ConvHistory(_)
        | MemberInvite(_)
        | MemberRemove(_)
        | MemberBan(_)
        | MemberSetRole(_)
        | ContactGetProfile { .. }
        | ContactSetAlias { .. }
        | ContactActionMenu { .. }
        | DirectorySearch { .. }
        | RosterList { .. }
        | RosterAdd { .. }
        | RosterUpdate { .. }
        | RosterRemove { .. }
        | FtSend { .. }
        | FtReceive { .. } => NotSessionTouching,
        AgentDiscover
        | AgentCatalog
        | AgentRegister { .. }
        | AgentRemove { .. }
        | ProviderList
        | ProviderRegister { .. }
        | ToolList
        | ToolRegister { .. }
        | ToolSetEnabled { .. }
        | CommandList
        | CommandInvoke { .. }
        | ConfigGet
        | ConfigSet { .. }
        | Caps => NotSessionTouching,
        UserCreate { .. }
        | UserList
        | UserDisable { .. }
        | UserSetRoles { .. }
        | UserSetPassword { .. }
        | RoleList
        | WhoAmI
        | SessionRevoke { .. }
        | ResourceGrantCreate { .. }
        | ResourceGrantList { .. }
        | ResourceGrantRevoke { .. } => NotSessionTouching,
        // -- user feedback + node-owned telemetry consent (N1): the coarse capability gate governs
        // (FeedbackSubmit -> SessionWrite, consent -> ControlRead/ControlWrite). FeedbackSubmit reads
        // a session's existence for response feedback but does not touch per-owner session state, so
        // it is not per-owner ownership-gated; consent is node-wide.
        FeedbackSubmit { .. } | TelemetryConsentGet | TelemetryConsentSet { .. } => {
            NotSessionTouching
        }
        // The notification list is node-wide (not per-owner session state): the coarse ControlRead
        // capability gate governs it (wire v37).
        NotificationList => NotSessionTouching,
        // The person/metacontact registry is node-wide (not per-owner session state): the coarse
        // ControlRead capability gate governs it (wire v37).
        PersonList => NotSessionTouching,
        // The node-owned gateway is a node-wide resident service (not per-owner session state): the
        // coarse capability gate governs (GatewayGet -> ControlRead, GatewaySet -> ControlWrite).
        GatewayGet | GatewaySet { .. } => NotSessionTouching,
        // Saved presences (W2-F) are node-wide shared config, not per-owner session state: the
        // coarse capability gate governs (PresenceList -> ControlRead, the mutations -> ControlWrite).
        PresenceList
        | PresenceSave { .. }
        | PresenceDelete { .. }
        | PresenceSetActive { .. } => NotSessionTouching,
    }
}

/// The pinned origin used for the `SubmitRouted` sample (routed to alice's session by a pin).
fn pinned_origin() -> Origin {
    Origin::new(
        "matrix/acct",
        OriginScope::Dm {
            user: "peer".into(),
        },
    )
}

/// One concrete instance of every `OwnerGated` variant, each targeting alice's session `s` (the
/// `Fs*` samples via a `FsRootId::Session(s)` root; `SubmitRouted` via a pinned origin). Paired with
/// the deny shape a non-owner must observe.
fn owner_gated_samples(s: &SessionId) -> Vec<(&'static str, ApiRequest, Deny)> {
    let sroot = FsRootId::Session(s.clone());
    vec![
        (
            "Submit",
            ApiRequest::Submit {
                session: s.clone(),
                command: start_turn("x"),
                origin: None,
                profile: None,
            },
            Deny::Forbidden,
        ),
        (
            "SubmitRouted",
            ApiRequest::SubmitRouted {
                origin: pinned_origin(),
                command: start_turn("x"),
            },
            Deny::Forbidden,
        ),
        (
            "SessionCreate",
            ApiRequest::SessionCreate {
                session: Some(s.clone()),
                profile: None,
            },
            Deny::Forbidden,
        ),
        (
            "Poll",
            ApiRequest::Poll {
                session: s.clone(),
                max: 8,
            },
            Deny::Forbidden,
        ),
        (
            "Respond",
            ApiRequest::Respond {
                session: s.clone(),
                response: HostResponse {
                    request_id: ReqId(1),
                    body: HostResponseBody::Approved {
                        approved: true,
                        allow_permanent: false,
                        reason: None,
                    },
                },
            },
            Deny::Forbidden,
        ),
        (
            "Subscribe",
            ApiRequest::Subscribe {
                session: s.clone(),
                after_seq: 0,
                max: 64,
            },
            Deny::Forbidden,
        ),
        (
            "Handover",
            ApiRequest::Handover {
                session: s.clone(),
                target: DeliveryTarget::new("matrix/acct", "!room:server", SinkKind::Primary),
            },
            Deny::Forbidden,
        ),
        (
            "RecordMeta",
            ApiRequest::RecordMeta(RecordMetaArgs {
                session: s.clone(),
                origin: pinned_origin(),
                kind: "presence".into(),
                body: Vec::new(),
            }),
            Deny::Forbidden,
        ),
        (
            "SetSessionModel",
            ApiRequest::SetSessionModel {
                session: s.clone(),
                model: "m".into(),
                provider: None,
            },
            Deny::Forbidden,
        ),
        (
            "SetSessionMode",
            ApiRequest::SetSessionMode {
                session: s.clone(),
                mode: daemon_api::ApprovalMode::Ask,
            },
            Deny::Forbidden,
        ),
        (
            "SetSessionOverlay",
            ApiRequest::SetSessionOverlay {
                session: s.clone(),
                overlay: SessionOverlay::default(),
            },
            Deny::Forbidden,
        ),
        (
            "Assign",
            ApiRequest::Assign { session: s.clone() },
            Deny::Forbidden,
        ),
        (
            "Cancel",
            ApiRequest::Cancel { session: s.clone() },
            Deny::Forbidden,
        ),
        (
            "ApprovalDecide",
            ApiRequest::ApprovalDecide {
                session: s.clone(),
                request_id: "r".into(),
                allow: true,
                allow_permanent: false,
                reason: None,
            },
            Deny::Forbidden,
        ),
        (
            "FingerprintList",
            ApiRequest::FingerprintList { session: s.clone() },
            Deny::Forbidden,
        ),
        (
            "FingerprintRevoke",
            ApiRequest::FingerprintRevoke {
                session: s.clone(),
                fingerprint: "fp".into(),
            },
            Deny::Forbidden,
        ),
        (
            "CheckpointRewind",
            ApiRequest::CheckpointRewind {
                session: s.clone(),
                checkpoint_id: "c".into(),
            },
            Deny::Forbidden,
        ),
        (
            "SessionUpdateMeta",
            ApiRequest::SessionUpdateMeta {
                session: s.clone(),
                patch: SessionMetaPatch {
                    title: None,
                    pinned: Some(true),
                    archived: None,
                },
            },
            Deny::Forbidden,
        ),
        (
            "Rewind",
            ApiRequest::Rewind {
                session: s.clone(),
                point: RewindPoint {
                    anchor: RewindAnchor::UserTurn { ordinal: 0 },
                    restore_workspace: false,
                },
            },
            Deny::Forbidden,
        ),
        (
            "FsList",
            ApiRequest::FsList {
                root: sroot.clone(),
                dir: String::new(),
                show_ignored: false,
                after: None,
            },
            Deny::Forbidden,
        ),
        (
            "FsStat",
            ApiRequest::FsStat {
                root: sroot.clone(),
                path: "a".into(),
            },
            Deny::Forbidden,
        ),
        (
            "FsRead",
            ApiRequest::FsRead {
                root: sroot.clone(),
                path: "a".into(),
                max_bytes: 0,
            },
            Deny::Forbidden,
        ),
        (
            "FsSearch",
            ApiRequest::FsSearch {
                root: sroot.clone(),
                query: FsSearchQuery {
                    query: "x".into(),
                    regex: false,
                    case_sensitive: false,
                    max_results: 0,
                    page: 0,
                },
            },
            Deny::Forbidden,
        ),
        (
            "FsWatchPoll",
            ApiRequest::FsWatchPoll(daemon_api::FsWatchAfterArgs {
                root: sroot.clone(),
                dir: String::new(),
                after_seq: 0,
                max: 8,
            }),
            Deny::Forbidden,
        ),
        (
            "FsWrite",
            ApiRequest::FsWrite(FsWriteArgs {
                root: sroot.clone(),
                path: "a".into(),
                bytes: Vec::new(),
                base_revision: None,
                force: false,
            }),
            Deny::Forbidden,
        ),
        (
            "FsWriteFromBlob",
            ApiRequest::FsWriteFromBlob(FsWriteFromBlobArgs {
                root: sroot.clone(),
                path: "a".into(),
                hash: ContentHash::new([0u8; 32]),
                base_revision: None,
                force: false,
            }),
            Deny::Forbidden,
        ),
        // Infallible reads (deny → nothing of the owner's).
        (
            "SessionHistory",
            ApiRequest::SessionHistory {
                session: s.clone(),
                after_cursor: 0,
                max: 64,
            },
            Deny::EmptyOrAbsent,
        ),
        (
            "DeliveryTargets",
            ApiRequest::DeliveryTargets { session: s.clone() },
            Deny::EmptyOrAbsent,
        ),
        ("Sessions", ApiRequest::Sessions, Deny::EmptyOrAbsent),
        (
            "SessionsQuery",
            ApiRequest::SessionsQuery {
                query: SessionQuery {
                    scope: SessionScope::All,
                    ..Default::default()
                },
            },
            Deny::EmptyOrAbsent,
        ),
        (
            "SessionGet",
            ApiRequest::SessionGet { session: s.clone() },
            Deny::EmptyOrAbsent,
        ),
        (
            "SessionSearch",
            ApiRequest::SessionSearch {
                query: "x".into(),
                limit: 10,
            },
            Deny::EmptyOrAbsent,
        ),
        (
            "SessionRecap",
            ApiRequest::SessionRecap { session: s.clone() },
            Deny::EmptyOrAbsent,
        ),
        (
            "Tree",
            ApiRequest::Tree { after: None },
            Deny::EmptyOrAbsent,
        ),
        (
            "ApprovalsPending",
            ApiRequest::ApprovalsPending {
                session: Some(s.clone()),
                after: None,
            },
            Deny::EmptyOrAbsent,
        ),
        (
            "CheckpointList",
            ApiRequest::CheckpointList {
                session: Some(s.clone()),
                after: None,
            },
            Deny::EmptyOrAbsent,
        ),
        ("FsRoots", ApiRequest::FsRoots, Deny::EmptyOrAbsent),
        // F3 (now gated): the fleet/unit surface — deny by returning nothing of the owner's.
        ("Fleet", ApiRequest::Fleet, Deny::EmptyOrAbsent),
        (
            "Unit",
            ApiRequest::Unit {
                unit: UnitId::new(s.as_str()),
            },
            Deny::EmptyOrAbsent,
        ),
        (
            "UnitEvents",
            ApiRequest::UnitEvents {
                unit: UnitId::new(s.as_str()),
                max: 8,
            },
            Deny::EmptyOrAbsent,
        ),
        (
            "UnitOutbound",
            ApiRequest::UnitOutbound {
                unit: UnitId::new(s.as_str()),
                max: 8,
            },
            Deny::EmptyOrAbsent,
        ),
        (
            "UnitHistory",
            ApiRequest::UnitHistory {
                unit: UnitId::new(s.as_str()),
                after_cursor: 0,
                max: 8,
            },
            Deny::EmptyOrAbsent,
        ),
        // F4 (now gated): the node-wide feeds — deny by returning nothing of the owner's.
        (
            "EventsSince",
            ApiRequest::EventsSince {
                cursor: 0,
                wait_ms: None,
            },
            Deny::EmptyOrAbsent,
        ),
        (
            "DeliverySessions",
            ApiRequest::DeliverySessions {
                transport: daemon_protocol::TransportId::new("matrix/acct"),
                after: None,
            },
            Deny::EmptyOrAbsent,
        ),
    ]
}

/// Assert a non-owner's response leaks none of alice's session `s`.
fn assert_denied(label: &str, resp: &ApiResponse, deny: Deny, s: &SessionId) {
    match deny {
        Deny::Forbidden => assert!(
            matches!(resp, ApiResponse::Error(ApiError::Forbidden(_))),
            "{label}: a non-owner must be Forbidden, got {resp:?}"
        ),
        Deny::EmptyOrAbsent => match resp {
            ApiResponse::Journal(p) => assert!(
                p.entries.is_empty(),
                "{label}: leaked {} entries",
                p.entries.len()
            ),
            ApiResponse::DeliveryTargets(v) => {
                assert!(v.is_empty(), "{label}: leaked {} targets", v.len())
            }
            ApiResponse::Sessions(v) => assert!(
                !v.iter().any(|i| &i.session == s),
                "{label}: leaked session"
            ),
            ApiResponse::SessionPage(p) => assert!(
                !p.sessions.iter().any(|i| &i.session == s),
                "{label}: leaked session"
            ),
            ApiResponse::SessionDetail(d) => assert!(d.is_none(), "{label}: leaked detail"),
            ApiResponse::SessionSearch(v) => {
                assert!(!v.iter().any(|h| &h.session == s), "{label}: leaked hit")
            }
            ApiResponse::SessionRecap(r) => assert!(r.is_none(), "{label}: leaked recap"),
            ApiResponse::Tree(t) => assert!(
                !t.nodes.iter().any(|n| n.session.as_ref() == Some(s)),
                "{label}: leaked tree node"
            ),
            ApiResponse::Approvals(p) => assert!(
                p.items.is_empty(),
                "{label}: leaked {} approvals",
                p.items.len()
            ),
            ApiResponse::Checkpoints(p) => assert!(
                p.items.is_empty(),
                "{label}: leaked {} checkpoints",
                p.items.len()
            ),
            ApiResponse::FsRoots(v) => assert!(
                !v.iter()
                    .any(|r| matches!(&r.id, FsRootId::Session(sid) if sid == s)),
                "{label}: leaked a session fs root"
            ),
            // F3 fleet/unit surface (now gated).
            ApiResponse::Fleet(r) => assert!(
                !r.children.contains(&UnitId::new(s.as_str())),
                "{label}: leaked a fleet child"
            ),
            ApiResponse::Unit(u) => assert!(u.is_none(), "{label}: leaked a unit node"),
            ApiResponse::UnitEvents(v) => {
                assert!(v.is_empty(), "{label}: leaked {} unit events", v.len())
            }
            ApiResponse::Drained(v) => {
                assert!(
                    v.is_empty(),
                    "{label}: leaked {} unit outbound items",
                    v.len()
                )
            }
            // F4 node-wide feeds (now gated): no session-bearing event / no session-scoped list entry.
            ApiResponse::EventsPage(p) => assert!(
                !p.events.iter().any(|e| matches!(e,
                    daemon_api::NodeEvent::SessionAdvanced { session, .. }
                    | daemon_api::NodeEvent::SessionMetaChanged { session, .. }
                    | daemon_api::NodeEvent::ApprovalPending { session, .. }
                        if session == s)),
                "{label}: leaked a session-bearing node event"
            ),
            ApiResponse::DeliverySessions(p) => {
                assert!(!p.items.contains(s), "{label}: leaked a delivery session")
            }
            other => panic!("{label}: unexpected empty-deny response shape {other:?}"),
        },
    }
}

/// Assemble a node with a wired checkpoint store, alice owning `s`, and `s` seeded with a parked
/// approval + a checkpoint + a routing pin (origin → `s`) — so the deny/positive assertions run
/// against real owner-stamped state.
async fn fixture(
    s: &SessionId,
) -> (
    Arc<NodeApiImpl>,
    daemon_host::SupervisorHandle,
    tempfile::TempDir,
) {
    let (node0, handle, store) = assemble_with_store();
    let cp_dir = tempfile::tempdir().expect("checkpoint dir");
    let cp = Arc::new(LocalCheckpointStore::new(cp_dir.path()));
    let node = Arc::new((*node0).clone().with_checkpoints(cp.clone()));
    alice_owns(&node, s).await;
    seed_approval(&store, s).await;
    seed_checkpoint(&cp, s).await;
    // Pin the routed origin to alice's session (operator-tier op), so `SubmitRouted` resolves to it.
    // `routing_bind_chat` avoids constructing the `#[non_exhaustive]` `ChatRoute`/`IsolationPolicy`.
    with_request_context(ctx("op", Role::Operator), async {
        node.routing_bind_chat(pinned_origin(), s.clone(), None)
            .await
    })
    .await
    .expect("pin origin → alice's session");
    (node, handle, cp_dir)
}

/// THE DENY TABLE: every `OwnerGated` `ApiRequest` variant, driven through the real
/// `daemon_api::dispatch` fan-out under a non-owner `User` (`bob`), is denied per its shape — proving
/// no session-touching surface leaks to a peer who lacks ownership and `SessionSeeAll`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_owner_gated_variant_denies_a_non_owner() {
    let s = SessionId::new("s-matrix");
    let (node, handle, _cp) = fixture(&s).await;

    for (label, req, deny) in owner_gated_samples(&s) {
        // The sample's static classification must match its declared deny shape (no drift).
        assert_eq!(
            classify(&req),
            Coverage::OwnerGated(deny),
            "{label}: classify() disagrees with the sample's declared deny shape"
        );
        let resp = with_request_context(ctx("bob", Role::User), async {
            daemon_api::dispatch(&*node, req.clone()).await
        })
        .await;
        assert_denied(label, &resp, deny, &s);
    }

    handle.shutdown().await;
}

/// The owner (alice) and an operator (SeeAll/ControlAny) are NEVER `Forbidden` on the same
/// owner-gated surface — proving the gate scopes by ownership rather than always-denying.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn owner_and_operator_are_not_denied() {
    let s = SessionId::new("s-matrix-pos");
    let (node, handle, _cp) = fixture(&s).await;

    for (name, role) in [("alice", Role::User), ("op", Role::Operator)] {
        for (label, req, _deny) in owner_gated_samples(&s) {
            let resp = with_request_context(ctx(name, role), async {
                daemon_api::dispatch(&*node, req.clone()).await
            })
            .await;
            assert!(
                !matches!(resp, ApiResponse::Error(ApiError::Forbidden(_))),
                "{name} ({role:?}) must NOT be Forbidden on {label}, got {resp:?}"
            );
        }
    }

    handle.shutdown().await;
}
