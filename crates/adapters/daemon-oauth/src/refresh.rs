// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Lease-time OAuth token refresh (credential plan Phase 4).
//!
//! [`TokenSetRefresher`] implements [`daemon_host::CredentialRefresher`]: the broker consults it
//! before minting each lease, so a near-expiry [`CredentialEnvelope::OAuthTokenSet`] is refreshed
//! (RFC 6749 §6 `refresh_token` grant) and atomically rewritten in the store BEFORE the secret is
//! served. Design points, per the plan:
//!
//! - **Refresh context, split by descriptor kind.** A CURATED set (Hugging Face) carries no
//!   context; the refresher recovers token endpoint + client identity from its registration table
//!   via the envelope's `method_id`, so operator config changes take effect without re-auth. A
//!   DYNAMIC set persists the context validated at completion time in the envelope
//!   (`token_endpoint` + `client_id`); absent both, the credential is explicitly non-refreshable
//!   and expires into `reauth_required`.
//! - **Single-flight per credential ref.** Concurrent turns near expiry trigger exactly ONE
//!   vendor exchange; the losers wait on the same lock and re-check the (now fresh) store row.
//! - **Expiry-skew window.** A token is refreshed `skew` BEFORE its stated expiry, so a lease
//!   minted now does not die mid-turn.
//! - **Rotation.** When the endpoint rotates the refresh token (RFC 6749 §6 permits it), the new
//!   one replaces the old in the same atomic rewrite; absent, the old one is kept.
//! - **Classified outcomes** (in the error message, over [`CredError::Other`]):
//!   `refresh_failed (transient): …` — network / 5xx / malformed success body; retrying the turn
//!   may succeed. `reauth_required: …` — the endpoint rejected the grant (4xx), the set expired
//!   with no refresh path, or the envelope cannot be decoded; only a new sign-in fixes these.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;

use daemon_common::{CredError, CredentialEnvelope, OAuthTokenSet};
use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_host::{CredentialRefresher, CredentialStore};

use daemon_api::ApiError;

/// The default freshness window: refresh when the token expires within this many seconds, so a
/// lease minted at the boundary still outlives its turn.
const DEFAULT_SKEW: Duration = Duration::from_secs(120);

/// The per-request deadline on the refresh exchange (same bound as the code exchange).
const REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Seconds since the unix epoch (the envelope's `expires_at` clock).
pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A curated method's refresh identity: where the `refresh_token` grant POSTs and as whom.
/// Registered at boot from the same operator config that registered the sign-in descriptor.
#[derive(Clone, Debug)]
pub struct RefreshEndpoint {
    /// The token endpoint the refresh grant POSTs to.
    pub token_endpoint: String,
    /// The OAuth2 client id presented on the grant.
    pub client_id: String,
    /// The confidential-client secret, when the registration has one (public clients: `None`).
    pub client_secret: Option<String>,
}

/// The lease-time refresher over a credential store: decodes the stored blob, refreshes a
/// near-expiry token set through the one SSRF-safe egress client, and atomically rewrites it.
pub struct TokenSetRefresher {
    store: Arc<dyn CredentialStore>,
    http: EgressClient,
    /// Curated refresh identities, keyed by the envelope's `method_id` (the auth family).
    curated: BTreeMap<String, RefreshEndpoint>,
    skew: Duration,
    /// Per-ref single-flight locks (created on first contention, retained for the process).
    locks: StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl TokenSetRefresher {
    /// A refresher over `store` with the curated `method_id -> refresh identity` table. Fails only
    /// when the TLS backend cannot initialize (a boot-environment defect) — surfaced, not defaulted.
    pub fn new(
        store: Arc<dyn CredentialStore>,
        curated: BTreeMap<String, RefreshEndpoint>,
    ) -> Result<Self, ApiError> {
        let http = EgressClient::new(EgressConfig {
            user_agent: Some("daemon".to_string()),
            timeout: Some(REFRESH_TIMEOUT),
        })
        .map_err(|e| ApiError::Other(format!("token refresh: building egress client: {e}")))?;
        Ok(Self {
            store,
            http,
            curated,
            skew: DEFAULT_SKEW,
            locks: StdMutex::new(HashMap::new()),
        })
    }

    /// Override the expiry-skew window (tests exercise the boundary without waiting hours).
    #[must_use]
    pub fn with_skew(mut self, skew: Duration) -> Self {
        self.skew = skew;
        self
    }

    /// The single-flight lock for `credential_ref`.
    fn lock_for(&self, credential_ref: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock().expect("refresh lock table");
        locks
            .entry(credential_ref.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    /// Decode the stored blob into a token set DUE for refresh. `None` = nothing to do here (no
    /// row, a bare key, a set without expiry, or one still fresh past the skew window). An
    /// undecodable magic-marked envelope is `reauth_required` — the projector could never serve
    /// it, so fail the acquire with the actionable classification instead of downstream noise.
    fn stale_token_set(&self, credential_ref: &str) -> Result<Option<OAuthTokenSet>, CredError> {
        let Some(blob) = self.store.get(credential_ref) else {
            return Ok(None);
        };
        let envelope = CredentialEnvelope::parse(&blob)
            .map_err(|e| reauth_required(credential_ref, &e.to_string()))?;
        let CredentialEnvelope::OAuthTokenSet(ts) = envelope else {
            return Ok(None);
        };
        let Some(expires_at) = ts.expires_at else {
            return Ok(None);
        };
        if unix_now().saturating_add(self.skew.as_secs()) < expires_at {
            return Ok(None);
        }
        Ok(Some(ts))
    }

    /// Resolve the refresh identity for `ts`: the curated table (by `method_id`) wins — it is the
    /// LIVE registration, so config changes apply without re-auth — else the envelope's persisted
    /// dynamic context; absent both the set is explicitly non-refreshable.
    fn refresh_endpoint(&self, ts: &OAuthTokenSet) -> Option<RefreshEndpoint> {
        if let Some(curated) = self.curated.get(&ts.method_id) {
            return Some(curated.clone());
        }
        match (&ts.token_endpoint, &ts.client_id) {
            (Some(token_endpoint), Some(client_id)) => Some(RefreshEndpoint {
                token_endpoint: token_endpoint.clone(),
                client_id: client_id.clone(),
                client_secret: None,
            }),
            _ => None,
        }
    }

    /// Run one RFC 6749 §6 refresh exchange and atomically rewrite the stored envelope.
    async fn refresh(&self, credential_ref: &str, ts: OAuthTokenSet) -> Result<(), CredError> {
        let refresh_token = ts.refresh_token.clone().ok_or_else(|| {
            reauth_required(
                credential_ref,
                "token expired and the grant issued no refresh token",
            )
        })?;
        let endpoint = self.refresh_endpoint(&ts).ok_or_else(|| {
            reauth_required(
                credential_ref,
                "no refresh context: the method is not registered and the envelope persists none",
            )
        })?;

        let mut pairs: Vec<(&str, &str)> = vec![
            ("grant_type", "refresh_token"),
            ("refresh_token", &refresh_token),
            ("client_id", &endpoint.client_id),
        ];
        if let Some(secret) = &endpoint.client_secret {
            pairs.push(("client_secret", secret));
        }
        let request = EgressRequest::post_form(&endpoint.token_endpoint, &pairs)
            .header("accept", "application/json");
        // `Redirects::None` — a token endpoint never legitimately redirects (kills redirect-SSRF).
        let response = self
            .http
            .execute(request, Redirects::None)
            .await
            .map_err(|e| refresh_failed(credential_ref, &format!("refresh exchange: {e}")))?;
        let status = response.status();
        let body = response.text().await.map_err(|e| {
            refresh_failed(credential_ref, &format!("reading refresh response: {e}"))
        })?;
        if !status.is_success() {
            // 5xx: the vendor is unwell — transient, the turn may retry. 4xx (`invalid_grant`,
            // revoked/rotated-away token): only a new sign-in fixes it.
            if status.is_server_error() {
                return Err(refresh_failed(
                    credential_ref,
                    &format!("token endpoint returned {status}"),
                ));
            }
            return Err(reauth_required(
                credential_ref,
                &format!("token endpoint rejected the refresh grant ({status})"),
            ));
        }
        let json: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            refresh_failed(credential_ref, &format!("refresh response not JSON: {e}"))
        })?;
        let access_token = json
            .get("access_token")
            .and_then(|t| t.as_str())
            .filter(|t| !t.is_empty())
            .ok_or_else(|| {
                refresh_failed(credential_ref, "refresh response carries no access_token")
            })?;
        // Rotation (§6): the endpoint MAY issue a new refresh token — store it in the same
        // rewrite (the old one is typically single-use once rotated); absent, keep the old one.
        let rotated = json
            .get("refresh_token")
            .and_then(|t| t.as_str())
            .filter(|t| !t.is_empty())
            .map(str::to_string);
        let expires_at = json
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .map(|secs| unix_now().saturating_add(secs));
        let renewed = OAuthTokenSet {
            access_token: access_token.to_string(),
            refresh_token: rotated.or(Some(refresh_token)),
            expires_at,
            // Identity + any persisted dynamic context carry forward unchanged.
            ..ts
        };
        self.store
            .set(
                credential_ref,
                &CredentialEnvelope::OAuthTokenSet(renewed).encode(),
            )
            .map_err(|e| refresh_failed(credential_ref, &format!("rewriting the store: {e}")))?;
        Ok(())
    }
}

/// Terminal classification: only a new interactive sign-in fixes this credential.
fn reauth_required(credential_ref: &str, detail: &str) -> CredError {
    CredError::Other(format!("reauth_required: {credential_ref}: {detail}"))
}

/// Transient classification: the refresh could not run to completion; retrying the turn may work.
fn refresh_failed(credential_ref: &str, detail: &str) -> CredError {
    CredError::Other(format!(
        "refresh_failed (transient): {credential_ref}: {detail}"
    ))
}

#[async_trait]
impl CredentialRefresher for TokenSetRefresher {
    async fn refresh_if_stale(&self, credential_ref: &str) -> Result<bool, CredError> {
        // Fast path outside the lock: nothing stale → no contention at all.
        if self.stale_token_set(credential_ref)?.is_none() {
            return Ok(false);
        }
        // Single-flight: one exchange per ref; losers wait, then re-check under the lock (the
        // winner's rewrite makes their re-read fresh, so they return without a second exchange).
        let lock = self.lock_for(credential_ref);
        let _guard = lock.lock().await;
        let Some(ts) = self.stale_token_set(credential_ref)? else {
            return Ok(false);
        };
        self.refresh(credential_ref, ts).await?;
        Ok(true)
    }
}
