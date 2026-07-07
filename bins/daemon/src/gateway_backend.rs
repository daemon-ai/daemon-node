// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The binary-side [`daemon_gateway::GatewayBackend`] implementation: the seam between the
//! OpenAI-compatible HTTP surface and the node's existing provider stack.
//!
//! `catalog()` reflects the node model catalog (the same `models` surface the GUI picker reads).
//! `authorize()` resolves a presented bearer to the admin token (external clients) or a per-session
//! token from the [`GatewayTokenRegistry`] (a node-managed foreign agent). `complete()` synthesizes
//! an ephemeral [`ProfileSpec`] from the effective model + resolved provider + credential — for an
//! `Admin` caller the request model + node-default credential; for a `Session` caller the token's
//! bound `{provider, model, credential_ref}` (the request model is ignored). It then resolves the
//! provider through the shared `provider_resolver`, acquires a broker lease for cloud providers
//! (threading the lease secret as the request bearer — local providers need none), and drives the
//! call via `drive_model_call` (non-stream) or `Provider::stream` (SSE). This mirrors
//! `Engine::call_model`'s acquire -> auth -> call -> release, without an engine/session.
//!
//! The per-session token registry ([`GatewayTokenRegistry`]) is the Phase 2 replacement for the
//! global provider->credential map: the interactive session builder mints a token bound to a
//! foreign session's routing (via the injected [`daemon_node::GatewayTokenMinter`] seam, which this
//! registry implements) and revokes it when the session ends, so a real provider key never reaches
//! the agent — the agent holds only an opaque loopback bearer.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use daemon_api::{ModelDescriptor, NodeApi, ProfileSpec};
use daemon_common::{CredScope, ProfileRef};
use daemon_core::{drive_model_call, CredentialProvider, EventSink, Request};
use daemon_gateway::{Completion, GatewayBackend, GatewayError, GatewayPrincipal};
use daemon_node::{GatewayBinding, GatewayTokenMinter, ProviderResolver};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

/// The stale-stream watchdog applied to a non-streaming gateway call (a provider silent longer than
/// this is a recoverable transport failure — see [`drive_model_call`]).
const GATEWAY_WATCHDOG: Duration = Duration::from_secs(300);

/// The per-session gateway-token registry: the node-side token->binding table backing the Phase 2
/// per-profile routing. It implements [`GatewayTokenMinter`] (the seam the session builder mints
/// through) and is shared with the [`NodeGatewayBackend`] (which resolves a presented session token
/// to its binding). Keys stay here, node-side; the agent only ever holds the opaque token.
#[derive(Default)]
pub struct GatewayTokenRegistry {
    table: Mutex<HashMap<String, GatewayBinding>>,
}

impl GatewayTokenRegistry {
    /// A fresh, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a presented per-session token to its binding, if registered.
    fn binding(&self, token: &str) -> Option<GatewayBinding> {
        self.table.lock().unwrap().get(token).cloned()
    }

    /// Whether a token is a currently-registered per-session token.
    fn contains(&self, token: &str) -> bool {
        self.table.lock().unwrap().contains_key(token)
    }
}

impl GatewayTokenMinter for GatewayTokenRegistry {
    fn mint(&self, binding: GatewayBinding) -> String {
        // A cryptographically-random loopback bearer; the (system-wide-catastrophic, effectively
        // unreachable) RNG-failure case falls back to a still-unique random-seeded token rather than
        // panicking a live daemon.
        let token =
            daemon_auth::generate_secret_hex(32).unwrap_or_else(|_| format!("gw-{}", uuid_like()));
        self.table.lock().unwrap().insert(token.clone(), binding);
        token
    }

    fn rebind(&self, token: &str, binding: GatewayBinding) {
        // Update the routed binding in place for a live model change (Phase 3); a no-op for an
        // unknown/revoked token (never resurrect one).
        if let Some(slot) = self.table.lock().unwrap().get_mut(token) {
            *slot = binding;
        }
    }

    fn revoke(&self, token: &str) {
        self.table.lock().unwrap().remove(token);
    }
}

/// A last-resort unique token seed for the unreachable RNG-failure branch (see [`GatewayTokenRegistry::mint`]).
fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{nanos:x}")
}

/// The gateway backend seams captured from the host bootstrap *before* `provider_resolver` is moved
/// into the node assembly. The node surface is bound later (post-assembly) via [`Self::into_backend`].
pub struct GatewaySeams {
    /// The shared provider-resolution seam (a clone of the node's `provider_resolver`).
    pub provider_resolver: ProviderResolver,
    /// A brokered credential client over the node's owner authority.
    pub credentials: Arc<dyn CredentialProvider>,
    /// The admin/global bearer token external OpenAI clients present (minted/pinned at boot).
    pub admin_token: String,
    /// The per-session token registry, shared with the interactive session builder's minter so a
    /// `Session` bearer resolves to its `{provider, model, credential_ref}` binding.
    pub registry: Arc<GatewayTokenRegistry>,
    /// The fallback credential profile (the node default profile) for `Admin` cloud calls.
    pub default_credential_ref: String,
    /// The optional model allowlist from `[gateway].models_allowlist`.
    pub allowlist: Option<Vec<String>>,
}

impl GatewaySeams {
    /// Bind the node surface to complete the backend (post-assembly).
    pub fn into_backend(self, node: Arc<dyn NodeApi>) -> NodeGatewayBackend {
        NodeGatewayBackend {
            node,
            provider_resolver: self.provider_resolver,
            credentials: self.credentials,
            admin_token: self.admin_token,
            registry: self.registry,
            default_credential_ref: self.default_credential_ref,
            allowlist: self.allowlist.map(|list| list.into_iter().collect()),
        }
    }
}

/// The node-side gateway backend. Holds the node surface (for the catalog), the provider-resolution
/// seam, the credential provider (a brokered client over the node's owner authority), the admin
/// bearer + per-session token registry (auth), and the default credential + optional allowlist.
pub struct NodeGatewayBackend {
    node: Arc<dyn NodeApi>,
    provider_resolver: ProviderResolver,
    credentials: Arc<dyn CredentialProvider>,
    /// The admin/global bearer external OpenAI clients present.
    admin_token: String,
    /// The per-session token->binding registry (shared with the session builder's minter).
    registry: Arc<GatewayTokenRegistry>,
    /// The fallback credential profile for an `Admin` cloud call (the node default profile, which
    /// already holds `DAEMON_CLOUD_API_KEY`).
    default_credential_ref: String,
    /// The optional model-id allowlist bounding the catalog (and `Admin` routing).
    allowlist: Option<HashSet<String>>,
}

/// The resolved routing for one completion: which provider + model to call and which stored
/// credential to acquire from.
struct Routing {
    provider: daemon_api::ProviderSelector,
    model: String,
    credential_ref: String,
}

impl NodeGatewayBackend {
    /// Constant-time-ish equality over the admin bearer (a loopback capability, but avoid an
    /// early-exit compare on principle).
    fn admin_token_eq(&self, presented: &str) -> bool {
        let (a, b) = (presented.as_bytes(), self.admin_token.as_bytes());
        if a.len() != b.len() {
            return false;
        }
        let mut diff = 0u8;
        for (x, y) in a.iter().zip(b.iter()) {
            diff |= x ^ y;
        }
        diff == 0
    }

    /// Resolve the effective routing for a completion request from the caller's principal:
    /// an `Admin` caller picks the model per-request (provider resolved from the node catalog, the
    /// node default credential); a `Session` caller's provider+model+credential are pinned to its
    /// token binding (the request model is ignored), so a foreign agent can only invoke its bound
    /// triple.
    async fn routing_for(
        &self,
        principal: &GatewayPrincipal,
        model: &str,
    ) -> Result<Routing, GatewayError> {
        match principal {
            GatewayPrincipal::Admin => {
                let provider = self
                    .catalog()
                    .await
                    .into_iter()
                    .find(|m| m.id == model)
                    .map(|m| m.provider)
                    .ok_or_else(|| GatewayError::UnknownModel(model.to_string()))?;
                Ok(Routing {
                    provider,
                    model: model.to_string(),
                    credential_ref: self.default_credential_ref.clone(),
                })
            }
            GatewayPrincipal::Session(token) => {
                // Re-resolve the binding node-side; a token revoked between auth and completion (the
                // session closed mid-flight) is treated as a bad request rather than an oracle.
                let binding = self.registry.binding(token).ok_or_else(|| {
                    GatewayError::BadRequest("session token is no longer valid".into())
                })?;
                let credential_ref = binding
                    .credential_ref
                    .unwrap_or_else(|| self.default_credential_ref.clone());
                Ok(Routing {
                    provider: binding.provider,
                    model: binding.model,
                    credential_ref,
                })
            }
        }
    }
}

#[async_trait::async_trait]
impl GatewayBackend for NodeGatewayBackend {
    async fn catalog(&self) -> Vec<ModelDescriptor> {
        // Page the node catalog (the same `models` surface the GUI picker reads).
        let mut out = Vec::new();
        let mut after = None;
        loop {
            let page = self.node.models(after.take()).await;
            out.extend(page.items);
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
        if let Some(allow) = &self.allowlist {
            out.retain(|m| allow.contains(&m.id));
        }
        out
    }

    async fn authorize(&self, token: &str) -> Option<GatewayPrincipal> {
        if self.admin_token_eq(token) {
            return Some(GatewayPrincipal::Admin);
        }
        if self.registry.contains(token) {
            return Some(GatewayPrincipal::Session(token.to_string()));
        }
        None
    }

    async fn complete(
        &self,
        principal: &GatewayPrincipal,
        model: &str,
        mut req: Request,
        stream: bool,
    ) -> Result<Completion, GatewayError> {
        let Routing {
            provider,
            model,
            credential_ref,
        } = self.routing_for(principal, model).await?;

        // Synthesize an ephemeral profile: the resolved provider + effective model + the resolved
        // credential ref (cloud only). No persona/tools/memory — this is a bare model call, not a
        // session.
        let spec = ProfileSpec {
            credential_ref: Some(credential_ref),
            ..ProfileSpec::new("gateway", provider, model)
        };

        // Resolve the concrete provider client through the shared seam.
        let client = (self.provider_resolver)(&spec)();

        // Cloud providers acquire a broker lease and thread its secret as the request bearer;
        // local engines skip credentials entirely (mirrors `Engine::call_model`).
        let lease = if provider.is_local() {
            None
        } else {
            let profile = ProfileRef::new(spec.credential_profile().to_string());
            let scope = CredScope::new([spec.credential_profile()], ["chat"], None);
            let lease = self
                .credentials
                .acquire(&profile, &scope)
                .await
                .map_err(|e| GatewayError::Credential(e.to_string()))?;
            req.auth = lease.secret.as_ref().map(|s| s.expose().to_string());
            Some(lease)
        };

        if stream {
            // SSE: drive `Provider::stream` on a task that owns the client + lease, forwarding
            // events over a channel and releasing the lease when the stream ends.
            let (tx, rx) = tokio::sync::mpsc::channel(64);
            let credentials = self.credentials.clone();
            tokio::spawn(async move {
                let mut events = client.stream(req);
                while let Some(ev) = events.next().await {
                    let is_err = ev.is_err();
                    let item = ev.map_err(|f| GatewayError::Provider(f.to_string()));
                    if tx.send(item).await.is_err() || is_err {
                        break;
                    }
                }
                drop(events);
                if let Some(lease) = &lease {
                    credentials.release(lease).await;
                }
            });
            Ok(Completion::Stream(Box::pin(
                tokio_stream::wrappers::ReceiverStream::new(rx),
            )))
        } else {
            // Non-stream: drive the call to completion (discarding the event sink — the gateway has
            // no session log), then release the lease.
            let result = drive_model_call(
                &*client,
                req,
                &CancellationToken::new(),
                GATEWAY_WATCHDOG,
                &EventSink::discarding(),
            )
            .await;
            if let Some(lease) = &lease {
                self.credentials.release(lease).await;
            }
            let out = result.map_err(|f| GatewayError::Provider(f.to_string()))?;
            Ok(Completion::Once(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::ProviderSelector;

    fn binding(model: &str) -> GatewayBinding {
        GatewayBinding {
            provider: ProviderSelector::GenAi,
            model: model.into(),
            credential_ref: Some("openai".into()),
        }
    }

    #[test]
    fn registry_mints_resolves_and_revokes() {
        let reg = GatewayTokenRegistry::new();
        let token = reg.mint(binding("gpt-4o"));
        assert!(reg.contains(&token), "a minted token is registered");
        assert_eq!(
            reg.binding(&token),
            Some(binding("gpt-4o")),
            "the token resolves to its routing binding"
        );
        reg.revoke(&token);
        assert!(
            !reg.contains(&token),
            "a revoked token is gone from the registry"
        );
        assert_eq!(reg.binding(&token), None);
    }

    #[test]
    fn minted_tokens_are_distinct() {
        let reg = GatewayTokenRegistry::new();
        let a = reg.mint(binding("gpt-4o"));
        let b = reg.mint(binding("gpt-4o"));
        assert_ne!(a, b, "each session gets its own token");
    }
}
