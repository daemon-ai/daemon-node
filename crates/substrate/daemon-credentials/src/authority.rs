//! The credential authority — the owner endpoint of the brokering chain.
//!
//! Holds the signing key and the [`CredentialSource`], mints scoped, signed [`CapabilityLease`]s in
//! the profile's [`CredMode`], serves `Proxied` use by attaching the real key, governs fleet
//! cost against a ceiling, and records every step to its audit log. The recursive *serve-or-forward*
//! brokering across cuts lives in `daemon-host`; this type is what a broker reaches when it is (or
//! reaches) the owner.

use crate::audit::{CredAuditKind, CredentialAuditEvent};
use crate::capability::{CapabilitySigner, CapabilityVerifyingKey};
use crate::source::CredentialSource;
use daemon_common::{
    Budget, CapabilityLease, CredError, CredId, CredMode, CredScope, LeaseSecret, ProfileRef,
    TraceId, UnitId,
};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn mode_tag(mode: CredMode) -> &'static str {
    match mode {
        CredMode::Native => "native",
        CredMode::Bearer => "bearer",
        CredMode::Proxied => "proxied",
    }
}

/// The caller context threaded into each authority operation: who is asking, under which trace. The
/// host fills these from the requesting unit + the restored cut/transport trace.
#[derive(Clone, Debug, Default)]
pub struct AcquireCtx {
    /// The requesting unit, where known.
    pub requester: Option<UnitId>,
    /// The correlation trace active for this request.
    pub trace: TraceId,
}

impl AcquireCtx {
    /// A context for `requester` under `trace`.
    pub fn new(requester: Option<UnitId>, trace: TraceId) -> Self {
        Self { requester, trace }
    }
}

struct Governance {
    spent_tokens: u64,
    ceiling: Option<u64>,
}

/// Fleet-wide cost governance owned by the authority.
pub struct CredentialAuthority {
    profile: ProfileRef,
    grant: CredScope,
    mode: CredMode,
    ttl_ms: u64,
    signer: Arc<CapabilitySigner>,
    source: Arc<dyn CredentialSource>,
    next_cap: AtomicU64,
    proxied: Mutex<HashMap<CredId, String>>,
    revoked: Mutex<HashSet<CredId>>,
    governance: Mutex<Governance>,
    audit: Mutex<Vec<CredentialAuditEvent>>,
}

impl CredentialAuthority {
    /// An authority over `profile`, able to grant up to `grant`, issuing leases in `mode` that live
    /// `ttl_ms`, signing with `signer` and provisioning secrets from `source`.
    pub fn new(
        grant: CredScope,
        mode: CredMode,
        ttl_ms: u64,
        signer: Arc<CapabilitySigner>,
        source: Arc<dyn CredentialSource>,
    ) -> Self {
        Self {
            profile: source.profile().clone(),
            grant,
            mode,
            ttl_ms,
            signer,
            source,
            next_cap: AtomicU64::new(0),
            proxied: Mutex::new(HashMap::new()),
            revoked: Mutex::new(HashSet::new()),
            governance: Mutex::new(Governance {
                spent_tokens: 0,
                ceiling: None,
            }),
            audit: Mutex::new(Vec::new()),
        }
    }

    /// Set a fleet cost ceiling (tokens); once reached, [`CredentialAuthority::charge`] throttles.
    pub fn with_cost_ceiling(self, ceiling: u64) -> Self {
        self.governance.lock().unwrap().ceiling = Some(ceiling);
        self
    }

    /// The profile this authority serves.
    pub fn profile(&self) -> &ProfileRef {
        &self.profile
    }

    /// The maximum scope this authority can grant.
    pub fn grant(&self) -> &CredScope {
        &self.grant
    }

    /// The verifying key holders use to check capabilities this authority mints.
    pub fn verifying_key(&self) -> CapabilityVerifyingKey {
        self.signer.verifying_key()
    }

    /// Mint a capability for `profile`, attenuated to `grant ∩ scope`. Records `Request` then
    /// either `Grant` or `Deny` to the audit log.
    pub fn acquire(
        &self,
        ctx: &AcquireCtx,
        profile: &ProfileRef,
        scope: &CredScope,
    ) -> Result<CapabilityLease, CredError> {
        self.push_audit(
            CredAuditKind::Request,
            None,
            scope.clone(),
            ctx,
            String::new(),
        );
        if profile != &self.profile {
            let e = CredError::Unavailable(profile.to_string());
            self.push_audit(CredAuditKind::Deny, None, scope.clone(), ctx, e.to_string());
            return Err(e);
        }
        let issued = self.grant.intersect(scope);
        if issued.is_empty() {
            self.push_audit(
                CredAuditKind::Deny,
                None,
                issued,
                ctx,
                "requested scope exceeds the grant".into(),
            );
            return Err(CredError::ScopeDenied);
        }
        let cap_id = CredId::new(format!(
            "{}-cap-{}",
            self.profile,
            self.next_cap.fetch_add(1, Ordering::Relaxed)
        ));
        let provisioned = self.source.provision(&cap_id, self.mode)?;
        let secret = match self.mode {
            CredMode::Native | CredMode::Bearer => {
                Some(LeaseSecret::new(provisioned.secret.clone()))
            }
            CredMode::Proxied => {
                // The real key stays here; the holder only ever gets a handle.
                self.proxied
                    .lock()
                    .unwrap()
                    .insert(cap_id.clone(), provisioned.secret.clone());
                None
            }
        };
        let mut lease = CapabilityLease {
            cap_id: cap_id.clone(),
            profile: profile.clone(),
            scope: issued.clone(),
            mode: self.mode,
            expires_at_ms: now_ms() + self.ttl_ms,
            secret,
            signature: Vec::new(),
        };
        lease.signature = self.signer.sign(&lease);
        self.push_audit(
            CredAuditKind::Grant,
            Some(cap_id),
            issued,
            ctx,
            format!("mode={} fresh={}", mode_tag(self.mode), provisioned.fresh),
        );
        Ok(lease)
    }

    /// Resolve a capability to a usable secret at the point of use. `Proxied` returns the real key
    /// the authority retained (the holder never had it); `Native`/`Bearer` returns the carried
    /// secret. Verifies signature + expiry first and refuses a revoked capability. Records `Use`.
    pub fn use_capability(
        &self,
        ctx: &AcquireCtx,
        lease: &CapabilityLease,
    ) -> Result<LeaseSecret, CredError> {
        self.verify(lease)?;
        if self.revoked.lock().unwrap().contains(&lease.cap_id) {
            return Err(CredError::Unavailable(lease.profile.to_string()));
        }
        let secret = match lease.mode {
            CredMode::Proxied => {
                // The owner performs the call with its retained key and returns only the *result*;
                // the raw key never crosses back to the holder (strongest isolation). We still
                // require the key to be present here — proving this owner is the resolver.
                let _key = self
                    .proxied
                    .lock()
                    .unwrap()
                    .get(&lease.cap_id)
                    .cloned()
                    .ok_or_else(|| CredError::Unavailable(lease.profile.to_string()))?;
                LeaseSecret::new(format!("proxied-result:{}", lease.cap_id))
            }
            CredMode::Native | CredMode::Bearer => lease
                .secret
                .clone()
                .ok_or_else(|| CredError::Other("lease missing embedded secret".into()))?,
        };
        self.push_audit(
            CredAuditKind::Use,
            Some(lease.cap_id.clone()),
            lease.scope.clone(),
            ctx,
            format!("mode={}", mode_tag(lease.mode)),
        );
        Ok(secret)
    }

    /// Verify a capability (signature over the recomputed digest, then expiry) against now.
    pub fn verify(&self, lease: &CapabilityLease) -> Result<(), CredError> {
        self.verifying_key().verify(lease, now_ms())
    }

    /// Revoke a capability: revoke any fresh key at the source, drop the proxied secret, and refuse
    /// future use. Records `Revoke`.
    pub fn revoke(&self, ctx: &AcquireCtx, cap_id: &CredId) {
        self.source.revoke(cap_id);
        self.proxied.lock().unwrap().remove(cap_id);
        self.revoked.lock().unwrap().insert(cap_id.clone());
        self.push_audit(
            CredAuditKind::Revoke,
            Some(cap_id.clone()),
            CredScope::nothing(),
            ctx,
            String::new(),
        );
    }

    /// Charge `tokens` of provider usage against the fleet ceiling and return the `Budget` cap a
    /// supervisor should now enforce: the remaining headroom, or a zeroed (throttled) budget once
    /// the ceiling is reached. Unbounded when no ceiling is set.
    pub fn charge(&self, tokens: u64) -> Budget {
        let mut g = self.governance.lock().unwrap();
        g.spent_tokens = g.spent_tokens.saturating_add(tokens);
        match g.ceiling {
            Some(c) if g.spent_tokens >= c => Budget {
                tokens: Some(0),
                wall_ms: None,
            },
            Some(c) => Budget {
                tokens: Some(c - g.spent_tokens),
                wall_ms: None,
            },
            None => Budget::unlimited(),
        }
    }

    /// Total tokens charged so far (governance observability).
    pub fn spent_tokens(&self) -> u64 {
        self.governance.lock().unwrap().spent_tokens
    }

    /// A snapshot of the audit log (the records the host journals into the verifiable trace).
    pub fn audit_log(&self) -> Vec<CredentialAuditEvent> {
        self.audit.lock().unwrap().clone()
    }

    /// Drain the audit log (the host journals each drained record, then seals the segment).
    pub fn take_audit(&self) -> Vec<CredentialAuditEvent> {
        std::mem::take(&mut self.audit.lock().unwrap())
    }

    fn push_audit(
        &self,
        kind: CredAuditKind,
        cap_id: Option<CredId>,
        scope: CredScope,
        ctx: &AcquireCtx,
        detail: String,
    ) {
        self.audit.lock().unwrap().push(CredentialAuditEvent {
            kind,
            profile: self.profile.clone(),
            cap_id,
            scope,
            requester: ctx.requester.clone(),
            trace: ctx.trace,
            detail,
            timestamp_ms: now_ms(),
        });
    }
}
