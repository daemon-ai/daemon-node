// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The projection-sync **mutation census** (daemon-projection-sync-spec.md §5): every
//! [`ApiRequest`] variant's declared effect class, as one exhaustive match — adding a variant
//! without classifying it is a compile error, which is the census's completeness gate.
//!
//! The classes drive the dispatch-side check ([`dispatch`](crate::dispatch)): a
//! [`MutationClass::MustChange`] handler that succeeds without recording an effect
//! ([`record_effect`](crate::record_effect)) is a silent-mutation defect; a
//! [`MutationClass::NonStateful`] handler that records one is a misclassification. Either logs
//! loud in every build; the conformance suite additionally hard-asserts per domain.
//!
//! Migration note (spec §10): during stages 3–4 a variant is promoted to `MustChange` only once
//! its emission is verified to happen *synchronously within the handler* (the dispatch scope
//! cannot see a spawned task's later emission). Variants whose invalidation is still a known gap,
//! rides an async adapter/worker path, or genuinely depends on the outcome stay
//! [`MutationClass::Conditional`] — each `GAP` marker below is a stage-4 work item.

use crate::{ApiRequest, ProjectionId};

/// A variant's declared effect on client-visible projections (spec §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationClass {
    /// Never changes any projection: reads, probes, and the explicitly non-projection surfaces
    /// (workspace fs / blobs / feedback). Recording an effect is a misclassification.
    NonStateful,
    /// A success MUST have recorded at least one effect; the listed projections document the
    /// expected domains (the dispatch check counts effects, the conformance census asserts
    /// domains).
    MustChange(&'static [ProjectionId]),
    /// May change the listed projections depending on outcome/timing (async adapter paths,
    /// content-dependent writes, still-unwired gaps). No dispatch-side assertion.
    Conditional(&'static [ProjectionId]),
}

/// The declared [`MutationClass`] of `req` (spec §5's census, in code).
pub fn census(req: &ApiRequest) -> MutationClass {
    use ApiRequest as R;
    use MutationClass::{Conditional, MustChange, NonStateful};
    use ProjectionId as P;
    match req {
        // -- pure reads / probes ------------------------------------------------------------
        R::Health
        | R::Stats
        | R::Telemetry
        | R::Sessions
        | R::Fleet
        | R::Tree { .. }
        | R::Unit { .. }
        | R::UnitEvents { .. }
        | R::UnitOutbound { .. }
        | R::SessionHistory { .. }
        | R::Subscribe { .. }
        | R::EventsSince { .. }
        | R::DeliveryTargets { .. }
        | R::DeliverySessions { .. }
        | R::UnitHistory { .. }
        | R::VerifyingKey
        | R::ModelSearch { .. }
        | R::ModelFiles { .. }
        | R::ModelDownloads
        | R::ModelCatalog
        | R::ModelRecommend { .. }
        | R::ModelQuantizes
        | R::ModelInspect { .. }
        | R::VhcRunList
        | R::VhcRunDetail { .. }
        | R::VhcHardwareReport
        | R::VhcDiskUsage
        | R::ProfileList
        | R::ProfileGet { .. }
        | R::ProfileExport { .. }
        | R::ProfileHistory { .. }
        | R::ProfileAt { .. }
        | R::SoulGet { .. }
        | R::SkillHistory { .. }
        | R::SkillAt { .. }
        | R::CuratorList { .. }
        | R::CredentialList
        | R::AuthProviders
        | R::Models { .. }
        | R::ModelCurrent { .. }
        | R::ProviderCatalog
        | R::ProviderModels { .. }
        | R::CustomProviderList
        | R::ApprovalsPending { .. }
        | R::FingerprintList { .. }
        | R::CheckpointList { .. }
        | R::SessionsQuery { .. }
        | R::SessionGet { .. }
        | R::SessionSearch { .. }
        | R::SessionRecap { .. }
        | R::AgentCatalog
        | R::SkillGet { .. }
        | R::ProviderList
        | R::ToolList
        | R::CommandList
        | R::Caps
        | R::ConfigGet
        | R::GatewayGet
        | R::CronList
        | R::CronRuns { .. }
        | R::CronSuggestions
        | R::RoutingListChats { .. }
        | R::RoutingGet { .. }
        | R::TransportRooms { .. }
        | R::TransportAdapters
        | R::TransportInstances
        | R::TransportSettings { .. }
        | R::ConvList { .. }
        | R::ConvGet { .. }
        | R::ConvCreateDetails { .. }
        | R::ConvJoinDetails { .. }
        | R::ConvHistory { .. }
        | R::ContactGetProfile { .. }
        | R::ContactActionMenu { .. }
        | R::DirectorySearch { .. }
        | R::RosterList { .. }
        | R::FsRoots
        | R::FsList { .. }
        | R::FsStat { .. }
        | R::FsRead { .. }
        | R::FsSearch { .. }
        | R::FsWatchPoll { .. }
        | R::BlobGet { .. }
        | R::BlobStat { .. }
        | R::UserList
        | R::RoleList
        | R::WhoAmI
        | R::ResourceGrantList { .. }
        | R::TelemetryConsentGet
        | R::CrashConsentGet
        | R::PresenceList
        | R::NotificationList
        | R::PersonList { .. }
        | R::Bootstrap => NonStateful,

        // -- non-projection writes (spec §5: explicit) --------------------------------------
        // `Poll` drains one's own session queue (a read of a live stream, not a projection);
        // workspace fs / blob writes and feedback land outside every client projection.
        R::Poll { .. }
        | R::FsWrite { .. }
        | R::FsWriteFromBlob { .. }
        | R::BlobPut { .. }
        | R::FeedbackSubmit { .. } => NonStateful,

        // -- sessions: verified synchronous emission ----------------------------------------
        // Submit/SubmitRouted stamp activity (`note_activity` -> SessionMetaChanged) before the
        // turn runs; SessionCreate emits RosterChanged after the durable create;
        // SessionUpdateMeta and the three override setters emit from their persistence paths
        // (stage 1).
        R::Submit { .. }
        | R::SubmitRouted { .. }
        | R::SessionCreate { .. }
        | R::SessionUpdateMeta { .. }
        | R::SetSessionModel { .. }
        | R::SetSessionMode { .. }
        | R::SetSessionOverlay { .. } => MustChange(&[P::Sessions]),

        // -- profiles / soul: every author path emits `ProfilesChanged` synchronously (stage 1
        // added `ProfileSelect`) -------------------------------------------------------------
        R::ProfileCreate { .. }
        | R::ProfileUpdate { .. }
        | R::ProfileDelete { .. }
        | R::ProfileSelect { .. }
        | R::ProfileClone { .. }
        | R::ProfileImport { .. }
        | R::ProfileRevert { .. }
        | R::SoulSet { .. } => MustChange(&[P::Profiles]),

        // -- contact roster: `emit_contacts_changed` fires synchronously after the awaited
        // adapter mutation ------------------------------------------------------------------
        R::RosterAdd { .. } | R::RosterUpdate { .. } | R::RosterRemove { .. } => {
            MustChange(&[P::Contacts])
        }

        // -- session control: stage 4 wired each success path to emit synchronously ----------
        // Assign emits RosterChanged after the create/re-arm; Cancel / Handover / both rewinds
        // emit SessionMetaChanged from their success paths.
        R::Assign { .. }
        | R::Cancel { .. }
        | R::Handover { .. }
        | R::Rewind { .. }
        | R::CheckpointRewind { .. } => MustChange(&[P::Sessions]),
        // A respond's Approvals invalidation lands on every accepted answer; RecordMeta is a log
        // append (only sometimes projection-visible).
        R::Respond { .. } => MustChange(&[P::Approvals]),
        R::RecordMeta(..) => Conditional(&[P::Sessions]),

        // -- approvals: stage 4 emits at the durable answer, before the wake -----------------
        R::ApprovalDecide { .. } => MustChange(&[P::Approvals]),

        // -- fleet drive (effects surface via the async fleet bridge) -----------------------
        R::Pause { .. } | R::Resume { .. } | R::Scale { .. } => Conditional(&[P::Fleet]),

        // -- model catalog / downloads (worker-driven; the catalog sink emits from the download
        // worker when the artifact lands) ----------------------------------------------------
        R::ModelDownload { .. }
        | R::ModelInstallFromUrl { .. }
        | R::ModelCancel { .. }
        | R::ModelPause { .. }
        | R::ModelResume { .. }
        | R::ModelQuantize { .. } => Conditional(&[P::Catalog]),
        // Delete invokes the catalog-changed callback inline; Activate rewrites the profile's
        // model binding and emits `ProfilesChanged` from the handler (stage 4).
        R::ModelDelete { .. } => MustChange(&[P::Catalog]),
        R::ModelActivate { .. } => MustChange(&[P::Profiles]),

        // -- vhc (service-driven emission rides the worker event pump — not verifiably
        // synchronous within the handler, so no dispatch assertion) --------------------------
        R::VhcJoin { .. }
        | R::VhcLeave { .. }
        | R::VhcPause { .. }
        | R::VhcResume { .. }
        | R::VhcSwitchModule { .. }
        | R::VhcSetPolicy { .. }
        | R::VhcDiskWipe { .. } => Conditional(&[P::Vhc]),

        // -- skills: stage 4 emits after the import/revert lands ----------------------------
        R::SkillPut { .. } | R::SkillRevert { .. } => MustChange(&[P::Skills]),

        // -- curator: stage 4 emits per verb (archive/restore also move the discovery set) ---
        R::CuratorPin { .. } | R::CuratorUnpin { .. } => MustChange(&[P::Curator]),
        R::CuratorArchive { .. } | R::CuratorRestore { .. } => MustChange(&[P::Curator, P::Skills]),
        R::CuratorRun { .. } => Conditional(&[P::Curator, P::Skills]),

        // -- credentials: stage 4 emits from every store landing (set/remove/label and the
        // interactive-auth completion paths — never carrying material) -----------------------
        R::CredentialSet { .. } | R::CredentialRemove { .. } | R::CredentialSetLabel { .. } => {
            MustChange(&[P::Credentials])
        }
        // The single-callback wrapper completes the flow or errors — a success landed a row.
        R::AuthComplete { .. } => MustChange(&[P::Credentials]),
        // A step may or may not complete the flow (a completion lands Credentials, and can
        // rebind Profiles / flip Agents); begin/cancel are flow-local.
        R::AuthStep { .. } => Conditional(&[P::Credentials, P::Profiles, P::Agents]),
        R::AuthBegin { .. } | R::AuthCancel { .. } => NonStateful,

        // -- custom providers: stage 4 emits after the durable write ------------------------
        R::CustomProviderSet { .. } | R::CustomProviderRemove { .. } => {
            MustChange(&[P::CustomProviders])
        }

        // -- foreign agents (register/remove emit AgentsChanged synchronously; a discover
        // sweep only emits when the catalog actually moved) ----------------------------------
        R::AgentRegister { .. } | R::AgentRemove { .. } => MustChange(&[P::Agents]),
        R::AgentDiscover => Conditional(&[P::Agents]),

        // -- registry / tools / config (Provider/ToolRegister + ConfigSet are unsupported
        // stubs on this node — reclassified when implemented; the census test pins this) -----
        R::ToolSetEnabled { .. } => MustChange(&[P::Tools]),
        R::ProviderRegister { .. } | R::ToolRegister { .. } | R::ConfigSet { .. } => NonStateful,
        R::GatewaySet { .. } => MustChange(&[P::Gateway]),

        // -- consent toggles: stage 4 emits after the durable write -------------------------
        R::TelemetryConsentSet { .. } => MustChange(&[P::TelemetryConsent]),
        R::CrashConsentSet { .. } => MustChange(&[P::CrashConsent]),

        // -- cron: stage 4 emits after every successful verb --------------------------------
        R::CronCreate { .. }
        | R::CronUpdate { .. }
        | R::CronDelete { .. }
        | R::CronTrigger { .. }
        | R::CronPause { .. }
        | R::CronAcceptSuggestion { .. }
        | R::CronDismissSuggestion { .. } => MustChange(&[P::Cron]),

        // -- routing: stage 4 emits after persist + hot-reload -------------------------------
        R::RoutingSet { .. } | R::RoutingBindChat { .. } | R::RoutingUnbindChat { .. } => {
            MustChange(&[P::Routing])
        }

        // -- presence profiles: stage 4 emits after the manager write -----------------------
        R::PresenceSave { .. } | R::PresenceDelete { .. } | R::PresenceSetActive { .. } => {
            MustChange(&[P::Presence])
        }

        // -- transports: the four durable-config verbs emit the Transports pointer (stage 4);
        // connect/disconnect surface via the async adapter's live TransportChanged -----------
        R::TransportRemove { .. }
        | R::TransportSetEnabled { .. }
        | R::TransportSetLabel { .. }
        | R::TransportConfigure { .. } => MustChange(&[P::Transports]),
        R::TransportDisconnect { .. } | R::TransportConnect { .. } => Conditional(&[P::Transports]),

        // -- conversations / membership / messages (adapter round-trips; echoes land async) --
        R::ConvCreate { .. }
        | R::ConvJoin { .. }
        | R::ConvLeave { .. }
        | R::ConvDelete { .. }
        | R::ConvSetTopic { .. }
        | R::ConvSetTitle { .. }
        | R::ConvSetDescription { .. } => Conditional(&[P::Conversations]),
        R::ConvSend { .. } | R::FtSend { .. } | R::FtReceive { .. } => Conditional(&[P::Messages]),
        R::MemberInvite { .. }
        | R::MemberRemove { .. }
        | R::MemberBan { .. }
        | R::MemberSetRole { .. } => Conditional(&[P::Conversations]),
        // Stage 4: the alias write emits via the shared contacts seam after the awaited adapter op.
        R::ContactSetAlias { .. } => MustChange(&[P::Contacts]),

        // -- command invocation (content-dependent: may drive any surface) -------------------
        R::CommandInvoke { .. } => Conditional(&[]),

        // -- access control: stage 4 emits after every committed admin mutation --------------
        R::UserCreate { .. }
        | R::UserDisable { .. }
        | R::UserSetRoles { .. }
        | R::UserSetPassword { .. }
        | R::SessionRevoke { .. } => MustChange(&[P::AccessControl]),
        // Reserved (`Unsupported` trait defaults, option B) — MustChange so an implementation
        // landing without an emission is caught on its first successful call.
        R::ResourceGrantCreate { .. } | R::ResourceGrantRevoke { .. } => {
            MustChange(&[P::AccessControl])
        }

        // -- fingerprints: stage 4 emits after the dormant-snapshot swap ---------------------
        R::FingerprintRevoke { .. } => MustChange(&[P::Fingerprints]),
    }
}
