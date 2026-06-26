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
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use daemon_api::{
    ApiError, AuthBeginRequest, AuthBeginResponse, AuthBindRequest, AuthFlowKind, AuthProviderInfo,
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
}

/// One in-flight interactive-auth flow: a family-specific object holding the secret continuation state
/// between `begin` and `complete`. Parked in the [`PendingAuthFlows`] registry under a `flow_id`.
#[async_trait]
pub trait PendingAuthFlow: Send + Sync {
    /// The authorization URL the client opens in a browser.
    fn authorization_url(&self) -> &str;

    /// The flow kind (informs the client how to capture the redirect).
    fn flow_kind(&self) -> AuthFlowKind;

    /// Finish from the captured `callback` (the full redirect URL or just its query string), producing
    /// the credential blob + account identity. Consumes the flow (single-use).
    async fn complete(self: Box<Self>, callback: &str) -> Result<AuthOutcome, ApiError>;
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

/// A parked flow plus the bind request to honor (if any) and its expiry.
struct ParkedFlow {
    flow: Box<dyn PendingAuthFlow>,
    bind: Option<AuthBindRequest>,
    expires_at: u64,
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
        let flow = factory.begin(&req.params, &req.redirect_uri).await?;

        let flow_id = fresh_flow_id();
        let expires_at = now_secs().saturating_add(self.ttl_secs);
        let response = AuthBeginResponse {
            flow_id: flow_id.clone(),
            authorization_url: flow.authorization_url().to_string(),
            redirect_uri: req.redirect_uri,
            expires_at,
            flow_kind: flow.flow_kind(),
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

    /// Remove and return a parked flow (with its bind request) by `flow_id`, erroring if it is unknown
    /// or expired. The caller awaits [`PendingAuthFlow::complete`] outside the registry lock.
    pub fn take(
        &self,
        flow_id: &str,
    ) -> Result<(Box<dyn PendingAuthFlow>, Option<AuthBindRequest>), ApiError> {
        let parked = {
            let mut guard = self.parked.lock().unwrap();
            guard.remove(flow_id)
        };
        let parked =
            parked.ok_or_else(|| ApiError::Other("unknown or expired auth flow".into()))?;
        if parked.expires_at <= now_secs() {
            return Err(ApiError::Other("unknown or expired auth flow".into()));
        }
        Ok((parked.flow, parked.bind))
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
