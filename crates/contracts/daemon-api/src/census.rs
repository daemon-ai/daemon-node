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

        // -- session control (outcome-dependent: a no-op cancel/assign is a valid success) ---
        R::Respond { .. }
        | R::Assign { .. }
        | R::Cancel { .. }
        | R::Handover { .. }
        | R::RecordMeta(..)
        | R::Rewind { .. }
        | R::CheckpointRewind { .. } => Conditional(&[P::Sessions]),

        // -- approvals (GAP: decide has no invalidation event yet) --------------------------
        R::ApprovalDecide { .. } => Conditional(&[P::Approvals, P::Sessions]),

        // -- fleet drive (effects surface via the async fleet bridge) -----------------------
        R::Pause { .. } | R::Resume { .. } | R::Scale { .. } => Conditional(&[P::Fleet]),

        // -- model catalog / downloads (worker-driven; catalog sink emits from the download
        // worker, GAP: ModelActivate/ModelDelete synchronous paths unverified) ---------------
        R::ModelDownload { .. }
        | R::ModelInstallFromUrl { .. }
        | R::ModelCancel { .. }
        | R::ModelPause { .. }
        | R::ModelResume { .. }
        | R::ModelDelete { .. }
        | R::ModelActivate { .. }
        | R::ModelQuantize { .. } => Conditional(&[P::Catalog]),

        // -- vhc (service-driven emission; verify synchronicity before promoting) -----------
        R::VhcJoin { .. }
        | R::VhcLeave { .. }
        | R::VhcPause { .. }
        | R::VhcResume { .. }
        | R::VhcSwitchModule { .. }
        | R::VhcSetPolicy { .. }
        | R::VhcDiskWipe { .. } => Conditional(&[P::Vhc]),

        // -- skills (GAP: no skills-scoped invalidation; edits surface via profile reads) ----
        R::SkillPut { .. } | R::SkillRevert { .. } => Conditional(&[P::Skills, P::Profiles]),

        // -- curator (GAP: pin/archive flip roster meta silently today) ---------------------
        R::CuratorPin { .. }
        | R::CuratorUnpin { .. }
        | R::CuratorArchive { .. }
        | R::CuratorRestore { .. }
        | R::CuratorRun { .. } => Conditional(&[P::Curator, P::Sessions]),

        // -- credentials (GAP: no CredentialsChanged; auth flows may land one at completion) -
        R::CredentialSet { .. }
        | R::CredentialRemove { .. }
        | R::CredentialSetLabel { .. }
        | R::AuthBegin { .. }
        | R::AuthStep { .. }
        | R::AuthComplete { .. }
        | R::AuthCancel { .. } => Conditional(&[P::Credentials]),

        // -- custom providers (GAP) ----------------------------------------------------------
        R::CustomProviderSet { .. } | R::CustomProviderRemove { .. } => {
            Conditional(&[P::CustomProviders])
        }

        // -- foreign agents (register/remove emit AgentsChanged; discover lands async) ------
        R::AgentDiscover | R::AgentRegister { .. } | R::AgentRemove { .. } => {
            Conditional(&[P::Agents])
        }

        // -- registry / tools / config (GAP: ToolSetEnabled + ConfigSet mutate silently;
        // Provider/ToolRegister are unsupported stubs on this node) --------------------------
        R::ProviderRegister { .. } | R::ToolRegister { .. } | R::ToolSetEnabled { .. } => {
            Conditional(&[P::Tools])
        }
        R::ConfigSet { .. } => Conditional(&[P::Gateway]),
        R::GatewaySet { .. } => Conditional(&[P::Gateway]),

        // -- consent toggles (GAP) -----------------------------------------------------------
        R::TelemetryConsentSet { .. } => Conditional(&[P::TelemetryConsent]),
        R::CrashConsentSet { .. } => Conditional(&[P::CrashConsent]),

        // -- cron (GAP) ----------------------------------------------------------------------
        R::CronCreate { .. }
        | R::CronUpdate { .. }
        | R::CronDelete { .. }
        | R::CronTrigger { .. }
        | R::CronPause { .. }
        | R::CronAcceptSuggestion { .. }
        | R::CronDismissSuggestion { .. } => Conditional(&[P::Cron]),

        // -- routing (GAP) -------------------------------------------------------------------
        R::RoutingSet { .. } | R::RoutingBindChat { .. } | R::RoutingUnbindChat { .. } => {
            Conditional(&[P::Routing])
        }

        // -- presence profiles (GAP) ---------------------------------------------------------
        R::PresenceSave { .. } | R::PresenceDelete { .. } | R::PresenceSetActive { .. } => {
            Conditional(&[P::Presence])
        }

        // -- transports (lifecycle lands via the async adapter -> TransportChanged) ----------
        R::TransportDisconnect { .. }
        | R::TransportRemove { .. }
        | R::TransportConnect { .. }
        | R::TransportSetEnabled { .. }
        | R::TransportSetLabel { .. }
        | R::TransportConfigure { .. } => Conditional(&[P::Transports]),

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
        R::ContactSetAlias { .. } => Conditional(&[P::Contacts]),

        // -- command invocation (content-dependent: may drive any surface) -------------------
        R::CommandInvoke { .. } => Conditional(&[]),

        // -- access control (GAP: admin mutations are invisible to other admin clients) ------
        R::UserCreate { .. }
        | R::UserDisable { .. }
        | R::UserSetRoles { .. }
        | R::UserSetPassword { .. }
        | R::SessionRevoke { .. }
        | R::ResourceGrantCreate { .. }
        | R::ResourceGrantRevoke { .. } => Conditional(&[P::AccessControl]),

        // -- fingerprints (GAP: revoke is invisible cross-client today) ---------------------
        R::FingerprintRevoke { .. } => Conditional(&[P::Fingerprints]),
    }
}
