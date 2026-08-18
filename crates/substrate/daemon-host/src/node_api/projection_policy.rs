// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Projection-sync **visibility policy** (daemon-projection-sync-spec.md §6): every
//! [`ProjectionId`]'s visibility class, as one exhaustive match — a projection without a class
//! does not compile (default-deny by construction). Enforced at every delivery point: the feed
//! page/subscribe scoping ([`scope_events_page`](super::NodeApiImpl::scope_events_page)) and
//! `Bootstrap` assembly.
//!
//! The table preserves today's exact posture: the session-bearing surfaces stay owner-scoped, the
//! admin/credential surfaces are capability-gated, and everything else passes for any
//! authenticated principal (single-user deployment; the access-control track revisits the Public
//! rows here, the single place to tighten).

use daemon_api::ProjectionId;
use daemon_auth::Capability;

/// A projection's visibility class (spec §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisibilityClass {
    /// Any authenticated principal sees the domain's events + rev.
    Public,
    /// Key-scoped events pass iff the key's owner is visible to the principal
    /// (`owner_visible`); All-scope rev pointers pass (the refetch they nudge is itself
    /// authorization-filtered).
    OwnerScoped,
    /// Events + revs pass iff the principal holds the projection's read capability.
    CapabilityScoped(Capability),
}

/// The visibility class of `projection` — exhaustive, so a new projection MUST take a stance
/// here before it can ship events.
pub(crate) fn visibility_class(projection: ProjectionId) -> VisibilityClass {
    use ProjectionId as P;
    use VisibilityClass as V;
    match projection {
        P::Sessions | P::Approvals => V::OwnerScoped,
        P::Credentials => V::CapabilityScoped(Capability::CredentialRead),
        // Fingerprints authenticate trusted clients and access-control administers users/roles:
        // both are admin material, visible only to an access administrator.
        P::Fingerprints | P::AccessControl => V::CapabilityScoped(Capability::AccessAdmin),
        P::Fleet
        | P::Profiles
        | P::Skills
        | P::Curator
        | P::Catalog
        | P::Agents
        | P::CustomProviders
        | P::Gateway
        | P::Tools
        | P::TelemetryConsent
        | P::CrashConsent
        | P::Cron
        | P::Presence
        | P::Routing
        | P::Transports
        | P::Conversations
        | P::Contacts
        | P::Messages
        | P::Persons
        | P::Notifications
        | P::Vhc => V::Public,
    }
}
