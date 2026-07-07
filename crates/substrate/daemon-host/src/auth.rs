// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Interactive-auth machinery: the host side of the client-driven SSO / OAuth2 login seam
//! (`daemon-interactive-auth-spec`).
//!
//! The daemon is headless — it owns no browser and no redirect listener. A decoupled (possibly
//! remote) client drives a browser-redirect login in two wire calls ([`daemon_api::AuthApi`]):
//! `auth_begin` mints an authorization URL against a *client-owned* `redirect_uri` and parks a
//! pending flow; the client opens a browser, captures the redirect, and relays the callback to
//! `auth_complete`, which finishes the flow and writes the resulting credential through the same
//! [`CredentialStore`](crate::credstore::CredentialStore) the rest of the node uses.
//!
//! This module is **family-agnostic**: a transport/provider family (matrix, an OAuth2 IdP, …)
//! registers an [`AuthFlowFactory`]; the factory mints a [`PendingAuthFlow`] that holds the
//! continuation state (PKCE verifier, a matrix `Client` + SSO handle) between the two calls. The
//! [`PendingAuthFlows`] registry parks flows by a single-use `flow_id`, evicts them on a TTL, and is
//! orchestrated by [`NodeApiImpl`](crate::node_api::NodeApiImpl)'s `AuthApi` impl (which performs the
//! credential write + optional profile bind).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use daemon_api::{
    ApiError, AuthBeginRequest, AuthBeginResponse, AuthBindRequest, AuthChallenge,
    AuthProviderInfo, AuthStepInput,
};
use daemon_protocol::TransportId;

/// The default lifetime of a parked flow: a flow not completed within this window is evicted. Login
/// redirects (matrix SSO, OAuth2) complete in seconds-to-minutes; ten minutes is a generous bound
/// that still bounds the registry's memory and the lifetime of the held continuation state.
pub const DEFAULT_FLOW_TTL_SECS: u64 = 600;

/// Wall-clock seconds since the Unix epoch (the flow-expiry clock).
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A best-effort, process-locally-unique, hard-to-guess flow id. Combines a monotonic counter, a
/// nanosecond time seed, and an ASLR-derived stack address, folded through SHA-256 and hex-encoded.
/// The `flow_id` is a server-side capability (whoever holds it can complete the flow), so it must not
/// be enumerable; a CSPRNG-backed id is a later hardening (kept dependency-free here, layout §3).
fn fresh_flow_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stack = &n as *const _ as usize as u128;
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    sha2::Digest::update(&mut hasher, n.to_le_bytes());
    sha2::Digest::update(&mut hasher, nanos.to_le_bytes());
    sha2::Digest::update(&mut hasher, stack.to_le_bytes());
    let digest = sha2::Digest::finalize(hasher);
    let mut out = String::with_capacity(32);
    for byte in digest.iter().take(16) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// How the node maps a completed flow's credential into the [`CredentialStore`] (non-wire; the node
/// decides the slot, never the client). Selected by the completing flow (an OAuth descriptor sets
/// it) and honored by the node's `auth_complete`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CredentialSlotKind {
    /// The historical transport-account shape: store the opaque blob under the flow's
    /// `credential_ref` (a bind-supplied `credential_ref` may override), and — when a bind was
    /// requested — attach a `BoundAccount { transport_instance, credential_ref }` to the profile.
    /// Used by Matrix SSO and the generic operator-facing `oauth2` family.
    #[default]
    Derived,
    /// A minted **model-provider API key**: store the BARE key under the bound profile's credential
    /// slot (the profile id the credential broker reads), so it rides the exact same downstream
    /// path as a pasted API key, and DO NOT attach a `BoundAccount` (a provider key is not a
    /// transport account). Requires a bind naming the target profile — a provider key with no bind
    /// target would be stranded where no broker reads it, so `auth_complete` rejects the no-bind
    /// case. Used by the provider-bound OAuth families (OpenRouter, Hugging Face).
    ProviderKeyForProfile,
}

/// The product of completing an interactive-auth flow: the opaque credential blob to persist plus the
/// identity needed to label + bind the account. The host writes the blob into the `CredentialStore`
/// and (when a bind was requested) attaches a `BoundAccount` to the target profile.
pub struct AuthOutcome {
    /// The opaque session/credential blob to store (the family's serialized session; never on the wire).
    pub credential_blob: String,
    /// The family-derived default `CredentialStore` key (e.g. from the resolved user id). The host may
    /// override it with a bind-supplied `credential_ref`.
    pub credential_ref: String,
    /// A human label for the account (e.g. the resolved `@user:hs.org`).
    pub account_label: String,
    /// The instance-qualified transport id this account resolves to (e.g. `matrix/@user:hs.org`).
    pub transport_instance: TransportId,
    /// How the node slots this credential (the node decides, never the client). Defaults to
    /// [`CredentialSlotKind::Derived`] (today's transport-account behavior).
    pub slot: CredentialSlotKind,
}

/// The host-side result of advancing a [`PendingAuthFlow`] one step: either the next challenge to
/// relay to the client, or the completed [`AuthOutcome`] the node persists. The flow-facing sibling
/// of the wire [`daemon_api::AuthStepResult`] — it carries the raw [`AuthOutcome`] (credential blob +
/// [`CredentialSlotKind`]) rather than the post-store [`daemon_api::AuthCompleteResponse`], because
/// the credential-store write + profile bind are the node's job, not the flow's.
pub enum AuthStepOutcome {
    /// The flow needs more input: relay this challenge and step again. The flow stays parked.
    Challenge(AuthChallenge),
    /// The flow finished: the node persists the blob + optional bind and evicts the flow.
    Completed(AuthOutcome),
}

/// One in-flight interactive-auth flow: a family-specific object holding the secret continuation
/// state across the challenge/response steps. Parked in the [`PendingAuthFlows`] registry under a
/// single-use `flow_id`. A flow is a small state machine: [`initial_challenge`](Self::initial_challenge)
/// is what `auth_begin` presents; each [`step`](Self::step) advances it (a single-redirect flow maps
/// `step(Callback(cb))` -> `Completed` in one hop). `step` takes `&self` (flows advance in place via
/// interior mutability where they carry state) so the registry can keep a flow parked across steps.
#[async_trait]
pub trait PendingAuthFlow: Send + Sync {
    /// The first challenge to present when the flow is begun (redirect URL, form, QR, message).
    fn initial_challenge(&self) -> AuthChallenge;

    /// Advance the flow with the client's `input`, returning the next challenge or the completed
    /// outcome. Called once per `auth_step`; a completing step produces the credential blob + identity.
    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError>;
}

/// A factory minting flows for one transport/provider family (e.g. `"matrix"`). Stateless beyond the
/// adapter configuration it captures; registered with the node during assembly.
#[async_trait]
pub trait AuthFlowFactory: Send + Sync {
    /// The family this factory serves (matched against [`AuthBeginRequest::family`]).
    fn family(&self) -> &str;

    /// Capability discovery: how a client should render the `auth_begin` form for this family.
    fn provider_info(&self) -> AuthProviderInfo;

    /// Begin a flow against the client-owned `redirect_uri`, returning the parked-flow continuation.
    async fn begin(
        &self,
        params: &BTreeMap<String, String>,
        redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError>;
}

/// A parked flow plus the bind request to honor (if any) and its expiry. The flow is held behind an
/// `Arc` so a `step` can borrow it across the family's `await` without holding the registry lock, and
/// a non-completing (challenge) step can leave it parked for the next step.
struct ParkedFlow {
    flow: Arc<dyn PendingAuthFlow>,
    bind: Option<AuthBindRequest>,
    expires_at: u64,
}

/// The registry-level result of stepping a parked flow: the next challenge (the flow stays parked),
/// or the completed outcome + the parked bind request (the flow has been evicted).
pub enum FlowStep {
    /// The flow issued another challenge; it remains parked under the same `flow_id`.
    Challenge(AuthChallenge),
    /// The flow completed and has been removed. Carries the outcome to persist + the parked bind.
    Completed {
        /// The credential blob + account identity + slot the node persists.
        outcome: AuthOutcome,
        /// The bind request parked at `begin` (honored by the node on completion), if any.
        bind: Option<AuthBindRequest>,
    },
}

/// The registry of interactive-auth factories + parked flows. Family factories are fixed at assembly;
/// flows are parked between `begin` and `complete` and evicted on a TTL. The credential write + profile
/// bind on completion live in the node (which owns the `CredentialStore` + `ProfileStore`); this
/// registry only mints, parks, and hands back flows.
pub struct PendingAuthFlows {
    factories: HashMap<String, std::sync::Arc<dyn AuthFlowFactory>>,
    parked: Mutex<HashMap<String, ParkedFlow>>,
    ttl_secs: u64,
}

impl PendingAuthFlows {
    /// A registry over `factories` (keyed by [`AuthFlowFactory::family`]) with the default flow TTL.
    pub fn new(factories: Vec<std::sync::Arc<dyn AuthFlowFactory>>) -> Self {
        Self::with_ttl(factories, DEFAULT_FLOW_TTL_SECS)
    }

    /// As [`PendingAuthFlows::new`] with an explicit flow `ttl_secs`.
    pub fn with_ttl(factories: Vec<std::sync::Arc<dyn AuthFlowFactory>>, ttl_secs: u64) -> Self {
        let factories = factories
            .into_iter()
            .map(|f| (f.family().to_string(), f))
            .collect();
        Self {
            factories,
            parked: Mutex::new(HashMap::new()),
            ttl_secs,
        }
    }

    /// Whether any factory is registered (the node reports the surface as available only then).
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// The registered providers, for client-side capability discovery (`auth_providers`).
    pub fn providers(&self) -> Vec<AuthProviderInfo> {
        let mut out: Vec<AuthProviderInfo> =
            self.factories.values().map(|f| f.provider_info()).collect();
        out.sort_by(|a, b| a.family.cmp(&b.family));
        out
    }

    /// Begin a flow: dispatch to the family factory, park the continuation under a fresh `flow_id`, and
    /// return the authorization URL + handle. Evicts any expired flows first.
    pub async fn begin(&self, req: AuthBeginRequest) -> Result<AuthBeginResponse, ApiError> {
        let factory = self
            .factories
            .get(&req.family)
            .ok_or_else(|| ApiError::Unsupported(format!("auth family: {}", req.family)))?
            .clone();
        // Box -> Arc so the flow can be parked and borrowed across `step`'s await (multi-step flows
        // stay parked between challenges).
        let flow: Arc<dyn PendingAuthFlow> =
            Arc::from(factory.begin(&req.params, &req.redirect_uri).await?);

        let flow_id = fresh_flow_id();
        let expires_at = now_secs().saturating_add(self.ttl_secs);
        let response = AuthBeginResponse {
            flow_id: flow_id.clone(),
            challenge: flow.initial_challenge(),
            expires_at,
        };

        let mut parked = self.parked.lock().unwrap();
        let now = now_secs();
        parked.retain(|_, p| p.expires_at > now);
        parked.insert(
            flow_id,
            ParkedFlow {
                flow,
                bind: req.bind,
                expires_at,
            },
        );
        Ok(response)
    }

    /// Advance the parked flow `flow_id` one step with `input`. A challenge result leaves the flow
    /// parked for the next step; a completion evicts it and returns the outcome + the parked bind.
    /// Errors if the flow is unknown or expired. The family's `step` is awaited OUTSIDE the registry
    /// lock (the flow is `Arc`-cloned out first), so a network round-trip never blocks the registry.
    pub async fn step(&self, flow_id: &str, input: AuthStepInput) -> Result<FlowStep, ApiError> {
        let flow = {
            let mut guard = self.parked.lock().unwrap();
            let now = now_secs();
            guard.retain(|_, p| p.expires_at > now);
            let parked = guard
                .get(flow_id)
                .ok_or_else(|| ApiError::Other("unknown or expired auth flow".into()))?;
            parked.flow.clone()
        };
        match flow.step(input).await? {
            AuthStepOutcome::Challenge(challenge) => Ok(FlowStep::Challenge(challenge)),
            AuthStepOutcome::Completed(outcome) => {
                // Single-use: evict the completed flow (and recover its parked bind) under the lock.
                let bind = self
                    .parked
                    .lock()
                    .unwrap()
                    .remove(flow_id)
                    .and_then(|p| p.bind);
                Ok(FlowStep::Completed { outcome, bind })
            }
        }
    }

    /// Drop a parked flow (idempotent — unknown ids are a no-op).
    pub fn cancel(&self, flow_id: &str) {
        self.parked.lock().unwrap().remove(flow_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_ids_are_unique_and_hex() {
        let a = fresh_flow_id();
        let b = fresh_flow_id();
        assert_ne!(a, b);
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
