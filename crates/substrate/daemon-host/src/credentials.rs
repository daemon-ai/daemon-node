//! The host credential broker: serve-or-forward, and the bridge to the engine's §7 port.
//!
//! The authority ([`daemon_credentials`]) is the *owner endpoint* — it holds secrets and mints
//! capabilities. This module is the host glue that makes brokering **recursive** down the
//! supervision tree (host-spec §6; supervision-spec rules #6, #142):
//!
//! - [`CredentialBroker`] is the serve-or-forward seam. [`OwnerBroker`] serves locally against the
//!   authority; [`RelayBroker`] is a host that is itself a placed child — it does **not** own
//!   secrets, so it narrows the requested scope by its own grant and re-brokers *upward* on its own
//!   cut. The wire client ([`crate::cut::RemoteCredentialClient`]) is also a broker, so a relay's
//!   upstream is just another broker and the chain composes to arbitrary depth.
//! - [`BrokeredCredentialProvider`] adapts any broker to the engine's [`CredentialProvider`] port,
//!   so the engine injected at [`CoreEngineFactory`](crate::engine_incarnation) acquires brokered
//!   capabilities through the identical interface whether it runs in-process or across a cut.

use crate::credstore::{CredentialStore, PooledStoreCredentialSource};
use crate::journal::CredentialAuditDrain;
use async_trait::async_trait;
use daemon_common::{
    CapabilityLease, CredError, CredId, CredMode, CredScope, FenceToken, LeaseSecret, ProfileRef,
    UnitId,
};
use daemon_core::CredentialProvider;
use daemon_credentials::{AcquireCtx, CapabilitySigner, CredentialAuditEvent, CredentialAuthority};
use daemon_telemetry::current_trace;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A hop's incarnation guard: the fence this incarnation was granted, against the session's current
/// (live) fence. A stale incarnation — one a newer activation has superseded — must not broker
/// credentials (host-spec §6; the dual-ownership fence rule extended to the credential path). The
/// `live` cell is the session's authoritative current lease, bumped when a newer incarnation
/// acquires; here it is shared so a superseded relay/owner self-rejects.
#[derive(Clone)]
pub struct FenceGuard {
    granted: FenceToken,
    live: Arc<Mutex<FenceToken>>,
}

impl FenceGuard {
    /// A guard for an incarnation `granted` the given fence, sharing the session's `live` fence.
    pub fn new(granted: FenceToken, live: Arc<Mutex<FenceToken>>) -> Self {
        Self { granted, live }
    }

    /// Whether this incarnation has been superseded (the live fence moved past its grant).
    pub fn is_stale(&self) -> bool {
        self.granted.0 < self.live.lock().unwrap().0
    }
}

/// The serve-or-forward seam at one hop of the brokering chain. `acquire` mints (owner) or narrows
/// and forwards (relay); `use_capability` resolves a `Proxied` call at the owner (relays forward).
#[async_trait]
pub trait CredentialBroker: Send + Sync {
    /// Acquire a capability for `profile` attenuated to (at most) `scope`, on behalf of `requester`.
    async fn acquire(
        &self,
        requester: Option<UnitId>,
        profile: &ProfileRef,
        scope: &CredScope,
    ) -> Result<CapabilityLease, CredError>;

    /// Resolve a capability to a usable secret at the point of use (the `Proxied` round-trip to the
    /// owner; `Native`/`Bearer` simply return the carried secret at the owner).
    async fn use_capability(
        &self,
        requester: Option<UnitId>,
        lease: &CapabilityLease,
    ) -> Result<LeaseSecret, CredError>;

    /// Signal a rotatable failure for `cap_id` (quota/auth) so the owning authority's pooled source
    /// prefers a different key on the next acquire. Owner-local by default: relays and the wire
    /// client no-op (cross-cut rotation propagation is a later refinement), and a single-key source
    /// no-ops too. This backs the engine's `Recovery::Rotate` hop.
    async fn rotate(&self, _requester: Option<UnitId>, _profile: &ProfileRef, _cap_id: &CredId) {}
}

/// The owner endpoint: serves brokered calls directly against the authority it holds.
pub struct OwnerBroker {
    authority: Arc<CredentialAuthority>,
    fence: Option<FenceGuard>,
}

impl OwnerBroker {
    /// Wrap the authority as the owner end of a brokering chain.
    pub fn new(authority: Arc<CredentialAuthority>) -> Self {
        Self {
            authority,
            fence: None,
        }
    }

    /// Bind this owner to an incarnation fence; a superseded incarnation rejects brokering.
    pub fn with_fence(mut self, fence: FenceGuard) -> Self {
        self.fence = Some(fence);
        self
    }

    /// The authority behind this owner (for governance/audit access).
    pub fn authority(&self) -> &Arc<CredentialAuthority> {
        &self.authority
    }

    fn fence_ok(&self) -> Result<(), CredError> {
        match &self.fence {
            Some(g) if g.is_stale() => Err(CredError::Fenced),
            _ => Ok(()),
        }
    }
}

#[async_trait]
impl CredentialBroker for OwnerBroker {
    async fn acquire(
        &self,
        requester: Option<UnitId>,
        profile: &ProfileRef,
        scope: &CredScope,
    ) -> Result<CapabilityLease, CredError> {
        if let Err(e) = self.fence_ok() {
            tracing::warn!(
                trace_id = %current_trace(),
                profile = %profile,
                requester = ?requester,
                reason = ?e,
                "cred.deny"
            );
            return Err(e);
        }
        let ctx = AcquireCtx::new(requester, current_trace());
        tracing::debug!(
            trace_id = %current_trace(),
            profile = %profile,
            requester = ?ctx.requester,
            "cred.acquire"
        );
        self.authority.acquire(&ctx, profile, scope)
    }

    async fn use_capability(
        &self,
        requester: Option<UnitId>,
        lease: &CapabilityLease,
    ) -> Result<LeaseSecret, CredError> {
        if let Err(e) = self.fence_ok() {
            tracing::warn!(
                trace_id = %current_trace(),
                requester = ?requester,
                reason = ?e,
                "cred.deny"
            );
            return Err(e);
        }
        let ctx = AcquireCtx::new(requester, current_trace());
        tracing::debug!(
            trace_id = %current_trace(),
            requester = ?ctx.requester,
            "cred.use"
        );
        self.authority.use_capability(&ctx, lease)
    }

    async fn rotate(&self, _requester: Option<UnitId>, _profile: &ProfileRef, cap_id: &CredId) {
        // A superseded incarnation must not mutate pool state; otherwise rotate the owned source.
        if self.fence_ok().is_ok() {
            self.authority.rotate(cap_id);
        }
    }
}

/// The owner endpoint for a node whose provider secrets are keyed **per profile** in a
/// [`CredentialStore`]. Instead of binding the node to one fixed profile (the single-authority
/// `OwnerBroker` shape, which rejects every other profile at `acquire`), this broker lazily builds a
/// per-profile [`CredentialAuthority`] on first use — each over a [`PooledStoreCredentialSource`]
/// for that profile — so a `CredentialSet` on any profile reaches that profile's sessions with no
/// dependence on a launch-configured profile name. All per-profile authorities **share the one node
/// signer**, so every lease verifies under the node's single published verifying key, and
/// `use_capability`/`rotate` route by the lease's / requested profile back to the owning authority.
pub struct MultiProfileStoreBroker {
    store: Arc<dyn CredentialStore>,
    signer: Arc<CapabilitySigner>,
    fallback: String,
    actions: Vec<String>,
    scope_tokens: Option<u64>,
    mode: CredMode,
    ttl_ms: u64,
    authorities: Mutex<HashMap<ProfileRef, Arc<CredentialAuthority>>>,
}

impl MultiProfileStoreBroker {
    /// A broker over `store`: each profile's lazily-built authority issues `mode` leases living
    /// `ttl_ms`, granting that profile + `actions` (with an optional scope token ceiling), and hands
    /// over `fallback` when the profile has no stored key. All authorities sign with `signer` so the
    /// node presents one verifying key.
    pub fn new(
        store: Arc<dyn CredentialStore>,
        signer: Arc<CapabilitySigner>,
        fallback: impl Into<String>,
        actions: impl IntoIterator<Item = impl Into<String>>,
        scope_tokens: Option<u64>,
        mode: CredMode,
        ttl_ms: u64,
    ) -> Self {
        Self {
            store,
            signer,
            fallback: fallback.into(),
            actions: actions.into_iter().map(Into::into).collect(),
            scope_tokens,
            mode,
            ttl_ms,
            authorities: Mutex::new(HashMap::new()),
        }
    }

    /// Get (or lazily create) the authority that serves `profile`.
    fn authority_for(&self, profile: &ProfileRef) -> Arc<CredentialAuthority> {
        let mut map = self.authorities.lock().unwrap();
        if let Some(authority) = map.get(profile) {
            return authority.clone();
        }
        let source = Arc::new(PooledStoreCredentialSource::new(
            self.store.clone(),
            profile.as_str(),
            self.fallback.clone(),
        ));
        let scope = CredScope::new(
            [profile.as_str()],
            self.actions.iter().map(String::as_str),
            self.scope_tokens,
        );
        let authority = Arc::new(CredentialAuthority::new(
            scope,
            self.mode,
            self.ttl_ms,
            self.signer.clone(),
            source,
        ));
        map.insert(profile.clone(), authority.clone());
        authority
    }
}

#[async_trait]
impl CredentialBroker for MultiProfileStoreBroker {
    async fn acquire(
        &self,
        requester: Option<UnitId>,
        profile: &ProfileRef,
        scope: &CredScope,
    ) -> Result<CapabilityLease, CredError> {
        let ctx = AcquireCtx::new(requester, current_trace());
        tracing::debug!(trace_id = %current_trace(), profile = %profile, requester = ?ctx.requester, "cred.acquire");
        self.authority_for(profile).acquire(&ctx, profile, scope)
    }

    async fn use_capability(
        &self,
        requester: Option<UnitId>,
        lease: &CapabilityLease,
    ) -> Result<LeaseSecret, CredError> {
        let ctx = AcquireCtx::new(requester, current_trace());
        tracing::debug!(trace_id = %current_trace(), requester = ?ctx.requester, "cred.use");
        // Route by the lease's own profile so the use reaches the authority that minted it.
        self.authority_for(&lease.profile)
            .use_capability(&ctx, lease)
    }

    async fn rotate(&self, _requester: Option<UnitId>, profile: &ProfileRef, cap_id: &CredId) {
        self.authority_for(profile).rotate(cap_id);
    }
}

impl CredentialAuditDrain for MultiProfileStoreBroker {
    fn take_audit(&self) -> Vec<CredentialAuditEvent> {
        let map = self.authorities.lock().unwrap();
        let mut out = Vec::new();
        for authority in map.values() {
            out.extend(authority.take_audit());
        }
        out
    }
}

/// A relay hop: a host that is itself a placed child and owns no secrets. It narrows the requested
/// scope by its own grant (per-hop attenuation) and re-brokers upward; raw key never reaches here.
pub struct RelayBroker {
    upstream: Arc<dyn CredentialBroker>,
    grant: CredScope,
    fence: Option<FenceGuard>,
}

impl RelayBroker {
    /// A relay that may grant at most `grant`, forwarding the narrowed request to `upstream`.
    pub fn new(upstream: Arc<dyn CredentialBroker>, grant: CredScope) -> Self {
        Self {
            upstream,
            grant,
            fence: None,
        }
    }

    /// Bind this relay to an incarnation fence; a superseded relay rejects brokering (the stale
    /// hop in the chain cannot acquire/forward).
    pub fn with_fence(mut self, fence: FenceGuard) -> Self {
        self.fence = Some(fence);
        self
    }

    fn fence_ok(&self) -> Result<(), CredError> {
        match &self.fence {
            Some(g) if g.is_stale() => Err(CredError::Fenced),
            _ => Ok(()),
        }
    }
}

#[async_trait]
impl CredentialBroker for RelayBroker {
    async fn acquire(
        &self,
        requester: Option<UnitId>,
        profile: &ProfileRef,
        scope: &CredScope,
    ) -> Result<CapabilityLease, CredError> {
        if let Err(e) = self.fence_ok() {
            tracing::warn!(
                trace_id = %current_trace(),
                profile = %profile,
                requester = ?requester,
                reason = ?e,
                "cred.deny"
            );
            return Err(e);
        }
        // Per-hop attenuation: a descendant can never get more than this hop is itself granted.
        let narrowed = self.grant.intersect(scope);
        if narrowed.is_empty() {
            tracing::warn!(
                trace_id = %current_trace(),
                profile = %profile,
                requester = ?requester,
                reason = "scope_denied",
                "cred.deny"
            );
            return Err(CredError::ScopeDenied);
        }
        tracing::debug!(
            trace_id = %current_trace(),
            profile = %profile,
            requester = ?requester,
            "cred.acquire"
        );
        self.upstream.acquire(requester, profile, &narrowed).await
    }

    async fn use_capability(
        &self,
        requester: Option<UnitId>,
        lease: &CapabilityLease,
    ) -> Result<LeaseSecret, CredError> {
        if let Err(e) = self.fence_ok() {
            tracing::warn!(
                trace_id = %current_trace(),
                requester = ?requester,
                reason = ?e,
                "cred.deny"
            );
            return Err(e);
        }
        // A relay holds no secret; the use must reach the owner.
        tracing::debug!(
            trace_id = %current_trace(),
            requester = ?requester,
            "cred.use"
        );
        self.upstream.use_capability(requester, lease).await
    }
}

/// Adapts a [`CredentialBroker`] to the engine's §7 [`CredentialProvider`] port — the host bridge
/// injected into the engine. `acquire`/`release`/`rotate` are the engine-facing surface; the
/// requester is fixed at construction (the unit the engine serves).
pub struct BrokeredCredentialProvider {
    broker: Arc<dyn CredentialBroker>,
    requester: Option<UnitId>,
}

impl BrokeredCredentialProvider {
    /// Bridge `broker` to the engine port on behalf of `requester`.
    pub fn new(broker: Arc<dyn CredentialBroker>, requester: Option<UnitId>) -> Self {
        Self { broker, requester }
    }

    /// Resolve a capability's usable secret (delegates to the broker; `Proxied` round-trips up).
    pub async fn resolve(&self, lease: &CapabilityLease) -> Result<LeaseSecret, CredError> {
        self.broker
            .use_capability(self.requester.clone(), lease)
            .await
    }
}

#[async_trait]
impl CredentialProvider for BrokeredCredentialProvider {
    async fn acquire(
        &self,
        profile: &ProfileRef,
        scope: &CredScope,
    ) -> Result<CapabilityLease, CredError> {
        self.broker
            .acquire(self.requester.clone(), profile, scope)
            .await
    }

    async fn release(&self, _lease: &CapabilityLease) {}

    async fn rotate(&self, profile: &ProfileRef, cap_id: &CredId) {
        self.broker
            .rotate(self.requester.clone(), profile, cap_id)
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credstore::{CredentialStore, MemCredentialStore, PooledStoreCredentialSource};
    use daemon_common::CredMode;
    use daemon_credentials::{CapabilitySigner, CredentialAuthority};

    /// The engine's `Recovery::Rotate` path (`provider.rotate` → broker → authority → source) cycles
    /// the pooled key: after rotating the served key, the next brokered acquire selects another key
    /// from the store's pool (B4 pooled-key rotation, end-to-end through the owner broker).
    #[tokio::test]
    async fn rotate_through_broker_cycles_pooled_keys() {
        let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
        store.set("grok", "key-a").unwrap();
        store.add_key("grok", "key-b").unwrap();

        let signer = Arc::new(CapabilitySigner::generate());
        let source = Arc::new(PooledStoreCredentialSource::new(
            store,
            "grok",
            "sk-fallback",
        ));
        let scope = CredScope::new(["grok"], ["chat"], Some(1_000));
        let authority = Arc::new(CredentialAuthority::new(
            scope.clone(),
            CredMode::Bearer,
            60_000,
            signer,
            source,
        ));
        // Drive through the engine-facing port, exactly as the engine does.
        let provider = BrokeredCredentialProvider::new(
            Arc::new(OwnerBroker::new(authority)) as Arc<dyn CredentialBroker>,
            None,
        );
        let grok = ProfileRef::new("grok");

        let first = provider.acquire(&grok, &scope).await.unwrap();
        assert_eq!(
            first.secret.as_ref().unwrap().expose(),
            "key-a",
            "the first acquire serves the primary pooled key"
        );

        // A rotatable failure rotates the served key; the next acquire prefers the other pooled key.
        provider.rotate(&grok, &first.cap_id).await;
        let second = provider.acquire(&grok, &scope).await.unwrap();
        assert_eq!(
            second.secret.as_ref().unwrap().expose(),
            "key-b",
            "rotation cycles to a different pooled key"
        );
    }

    /// The multi-profile broker serves each profile from its own stored key (no single bound
    /// profile), and `use_capability` routes a lease back to the authority that minted it.
    #[tokio::test]
    async fn multi_profile_broker_serves_each_profile_independently() {
        let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
        store.set("alpha", "key-alpha").unwrap();
        store.set("beta", "key-beta").unwrap();
        let signer = Arc::new(CapabilitySigner::generate());
        let broker = MultiProfileStoreBroker::new(
            store,
            signer,
            "sk-fallback",
            ["chat", "embed"],
            Some(1_000),
            CredMode::Bearer,
            60_000,
        );

        let scope_a = CredScope::new(["alpha"], ["chat"], Some(1_000));
        let lease_a = broker
            .acquire(None, &ProfileRef::new("alpha"), &scope_a)
            .await
            .expect("alpha acquires");
        assert_eq!(lease_a.secret.as_ref().unwrap().expose(), "key-alpha");

        let scope_b = CredScope::new(["beta"], ["chat"], Some(1_000));
        let lease_b = broker
            .acquire(None, &ProfileRef::new("beta"), &scope_b)
            .await
            .expect("beta acquires (a different profile, no rebinding needed)");
        assert_eq!(lease_b.secret.as_ref().unwrap().expose(), "key-beta");

        // The lease resolves at the owner that minted it (routed by lease.profile).
        let used = broker
            .use_capability(None, &lease_a)
            .await
            .expect("alpha's lease resolves");
        assert_eq!(used.expose(), "key-alpha");

        // Audit aggregates across both per-profile authorities.
        assert!(
            !broker.take_audit().is_empty(),
            "acquires recorded audit across profiles"
        );
    }

    /// A profile with no stored key falls back to the configured fallback (zero-config bootstrap).
    #[tokio::test]
    async fn multi_profile_broker_falls_back_for_unset_profile() {
        let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
        let signer = Arc::new(CapabilitySigner::generate());
        let broker = MultiProfileStoreBroker::new(
            store,
            signer,
            "sk-fallback",
            ["chat"],
            Some(1_000),
            CredMode::Bearer,
            60_000,
        );
        let scope = CredScope::new(["ghost"], ["chat"], Some(1_000));
        let lease = broker
            .acquire(None, &ProfileRef::new("ghost"), &scope)
            .await
            .expect("unset profile still acquires via fallback");
        assert_eq!(lease.secret.as_ref().unwrap().expose(), "sk-fallback");
    }
}
