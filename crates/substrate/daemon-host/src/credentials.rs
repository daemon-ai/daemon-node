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

use async_trait::async_trait;
use daemon_common::{
    CapabilityLease, CredError, CredId, CredScope, FenceToken, LeaseSecret, ProfileRef, UnitId,
};
use daemon_core::CredentialProvider;
use daemon_credentials::{AcquireCtx, CredentialAuthority};
use daemon_telemetry::current_trace;
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
        self.fence_ok()?;
        let ctx = AcquireCtx::new(requester, current_trace());
        self.authority.acquire(&ctx, profile, scope)
    }

    async fn use_capability(
        &self,
        requester: Option<UnitId>,
        lease: &CapabilityLease,
    ) -> Result<LeaseSecret, CredError> {
        self.fence_ok()?;
        let ctx = AcquireCtx::new(requester, current_trace());
        self.authority.use_capability(&ctx, lease)
    }

    async fn rotate(&self, _requester: Option<UnitId>, _profile: &ProfileRef, cap_id: &CredId) {
        // A superseded incarnation must not mutate pool state; otherwise rotate the owned source.
        if self.fence_ok().is_ok() {
            self.authority.rotate(cap_id);
        }
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
        self.fence_ok()?;
        // Per-hop attenuation: a descendant can never get more than this hop is itself granted.
        let narrowed = self.grant.intersect(scope);
        if narrowed.is_empty() {
            return Err(CredError::ScopeDenied);
        }
        self.upstream.acquire(requester, profile, &narrowed).await
    }

    async fn use_capability(
        &self,
        requester: Option<UnitId>,
        lease: &CapabilityLease,
    ) -> Result<LeaseSecret, CredError> {
        self.fence_ok()?;
        // A relay holds no secret; the use must reach the owner.
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
