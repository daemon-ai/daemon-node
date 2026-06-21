//! The first-class credential audit event — "who requested which credential, when, under which
//! trace."
//!
//! The authority records one of these for every lifecycle step. The host journals them into the
//! phase-6 verifiable trace (`daemon-host`'s `JournalSink`), so the sealed, signed segment is the
//! tamper-evident answer to "what requested credentials when," reconstructable across a whole
//! brokering chain by `trace_id`. This type is codec-free here; the host renders it to a trace
//! record.

use daemon_common::{CredId, CredScope, ProfileRef, TraceId, UnitId};

/// Which lifecycle step a [`CredentialAuditEvent`] records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredAuditKind {
    /// A holder asked for a capability (pre-decision).
    Request,
    /// The owner minted and granted a capability.
    Grant,
    /// A relay narrowed the scope and re-brokered upward.
    Attenuate,
    /// A capability was used (a `Proxied` call, or a `Native`/`Bearer` resolve).
    Use,
    /// A request was denied (scope exceeded, fenced, unavailable).
    Deny,
    /// A capability was revoked.
    Revoke,
    /// A capability expired.
    Expire,
}

impl CredAuditKind {
    /// A short, stable label (the journal envelope subject).
    pub fn label(self) -> &'static str {
        match self {
            CredAuditKind::Request => "cred.request",
            CredAuditKind::Grant => "cred.grant",
            CredAuditKind::Attenuate => "cred.attenuate",
            CredAuditKind::Use => "cred.use",
            CredAuditKind::Deny => "cred.deny",
            CredAuditKind::Revoke => "cred.revoke",
            CredAuditKind::Expire => "cred.expire",
        }
    }
}

/// One audit record. Carries the requester, profile, scope, capability id, trace context, and a
/// human detail — everything needed to answer "who requested what when."
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredentialAuditEvent {
    /// The lifecycle step.
    pub kind: CredAuditKind,
    /// The profile the credential serves.
    pub profile: ProfileRef,
    /// The capability id, once one exists (a `Request`/`Deny` may precede minting).
    pub cap_id: Option<CredId>,
    /// The (attenuated) scope at this step.
    pub scope: CredScope,
    /// The requesting unit, where known.
    pub requester: Option<UnitId>,
    /// The correlation trace active when the event occurred.
    pub trace: TraceId,
    /// Human/structured detail (e.g. "mode=bearer fresh=true", or a denial reason).
    pub detail: String,
    /// Milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
}

impl CredentialAuditEvent {
    /// A compact one-line rendering for the journal detail field.
    pub fn summary(&self) -> String {
        let cap = self.cap_id.as_ref().map(|c| c.as_str()).unwrap_or("-");
        let who = self.requester.as_ref().map(|u| u.as_str()).unwrap_or("-");
        format!(
            "{} profile={} cap={} requester={} {}",
            self.kind.label(),
            self.profile,
            cap,
            who,
            self.detail
        )
    }
}
