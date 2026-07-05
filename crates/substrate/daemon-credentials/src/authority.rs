// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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
    /// The current revocation epoch (Cluster F, Part B). Stamped onto every minted lease; bumped by
    /// [`CredentialAuthority::revoke_all`] so a lease minted before the bump is refused at
    /// [`CredentialAuthority::use_capability`].
    epoch: AtomicU64,
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
            epoch: AtomicU64::new(0),
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
            epoch: self.epoch.load(Ordering::Acquire),
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
        // Cluster F (Part B): a lease minted before the authority's last `revoke_all` (credential
        // removed/replaced) is refused, even though its signature + expiry still check out.
        if lease.epoch != self.epoch.load(Ordering::Acquire) {
            return Err(CredError::Unavailable(lease.profile.to_string()));
        }
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

    /// Revoke **every** outstanding lease this authority has minted (Cluster F, Part B): bump the
    /// revocation epoch (so a lease minted before now fails [`use_capability`]'s epoch check) and
    /// drop all retained `Proxied` keys (so a proxied handle can no longer be resolved). Records a
    /// single `Revoke` audit entry with an empty scope. Called when the profile's credential is
    /// removed or replaced, so an already-minted lease against the old material cannot keep resolving.
    ///
    /// Concurrency: bumps the epoch atomic, then clears `proxied` in its own lock scope, then records
    /// audit under the `audit` lock — no lock is held across another (mirrors [`revoke`]).
    pub fn revoke_all(&self, ctx: &AcquireCtx) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.proxied.lock().unwrap().clear();
        self.push_audit(
            CredAuditKind::Revoke,
            None,
            CredScope::nothing(),
            ctx,
            "revoke_all (credential removed/replaced)".into(),
        );
    }

    /// Rotate the credential behind `cap_id`: ask the source to prefer a different key on the next
    /// acquire (a pooled source marks the underlying key exhausted). Single-key sources no-op. This
    /// is the owner-local hop of the engine's `Recovery::Rotate` path; it does not invalidate the
    /// outstanding lease (the engine re-acquires), so no audit `Revoke` is recorded.
    pub fn rotate(&self, cap_id: &CredId) {
        self.source.rotate(cap_id);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilitySigner;
    use crate::source::StubCredentialSource;

    fn authority(mode: CredMode) -> CredentialAuthority {
        let signer = Arc::new(CapabilitySigner::generate());
        let source = Arc::new(StubCredentialSource::new("openai", "sk-configured"));
        CredentialAuthority::new(
            CredScope::new(["openai"], ["chat"], Some(1_000)),
            mode,
            60_000,
            signer,
            source,
        )
    }

    fn scope() -> CredScope {
        CredScope::new(["openai"], ["chat"], Some(1_000))
    }

    /// Cluster F (Part B): a `Bearer` lease minted before `revoke_all` must be refused at
    /// `use_capability` afterwards. This is the epoch-check discriminator — `Bearer` returns the
    /// embedded secret, so ONLY the epoch check (not the proxied-key clear) can stop it.
    #[test]
    fn bearer_lease_use_fails_after_revoke_all() {
        let auth = authority(CredMode::Bearer);
        let ctx = AcquireCtx::default();
        let lease = auth
            .acquire(&ctx, &ProfileRef::new("openai"), &scope())
            .expect("acquire");
        // Usable before revocation.
        auth.use_capability(&ctx, &lease)
            .expect("a fresh Bearer lease resolves");
        // Revoke every outstanding lease (credential removed/replaced).
        auth.revoke_all(&ctx);
        let err = auth
            .use_capability(&ctx, &lease)
            .expect_err("a lease minted before revoke_all must be refused");
        assert!(
            matches!(err, CredError::Unavailable(_)),
            "a stale-epoch lease must be Unavailable, got {err:?}"
        );
        // A freshly-minted lease (under the new epoch) works again.
        let fresh = auth
            .acquire(&ctx, &ProfileRef::new("openai"), &scope())
            .expect("re-acquire under new epoch");
        auth.use_capability(&ctx, &fresh)
            .expect("a lease minted after revoke_all resolves");
    }

    /// `revoke_all` drops retained `Proxied` keys, so a proxied handle can no longer resolve.
    #[test]
    fn proxied_key_dropped_on_revoke_all() {
        let auth = authority(CredMode::Proxied);
        let ctx = AcquireCtx::default();
        let lease = auth
            .acquire(&ctx, &ProfileRef::new("openai"), &scope())
            .expect("acquire");
        assert!(lease.secret.is_none(), "Proxied hands over only a handle");
        auth.use_capability(&ctx, &lease)
            .expect("a fresh proxied handle resolves at the owner");
        auth.revoke_all(&ctx);
        let err = auth
            .use_capability(&ctx, &lease)
            .expect_err("a revoked proxied handle must not resolve");
        assert!(matches!(err, CredError::Unavailable(_)), "got {err:?}");
    }

    /// The epoch is covered by the signature: editing it on a minted lease breaks verification, so a
    /// relay cannot re-stamp a stale epoch to a current one.
    #[test]
    fn epoch_is_signed() {
        let auth = authority(CredMode::Bearer);
        let ctx = AcquireCtx::default();
        let mut lease = auth
            .acquire(&ctx, &ProfileRef::new("openai"), &scope())
            .expect("acquire");
        auth.verify(&lease)
            .expect("a freshly minted lease verifies");
        lease.epoch = lease.epoch.wrapping_add(1);
        assert_eq!(
            auth.verify(&lease).unwrap_err(),
            CredError::BadSignature,
            "tampering the epoch must invalidate the signature"
        );
    }
}
