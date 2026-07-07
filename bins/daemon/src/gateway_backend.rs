// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The binary-side [`daemon_gateway::GatewayBackend`] implementation: the seam between the
//! OpenAI-compatible HTTP surface and the node's existing provider stack.
//!
//! `catalog()` reflects the node model catalog (the same `models` surface the GUI picker reads).
//! `complete()` synthesizes an ephemeral [`ProfileSpec`] from the requested model + the resolved
//! provider, resolves the provider through the shared `provider_resolver`, acquires a broker lease
//! for cloud providers (threading the lease secret as the request bearer — local providers need
//! none), and drives the call via `drive_model_call` (non-stream) or `Provider::stream` (SSE). This
//! mirrors `Engine::call_model`'s acquire -> auth -> call -> release, without an engine/session.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use daemon_api::{ModelDescriptor, NodeApi, ProfileSpec, ProviderSelector};
use daemon_common::{CredScope, ProfileRef};
use daemon_core::{drive_model_call, CredentialProvider, EventSink, Request};
use daemon_gateway::{Completion, GatewayBackend, GatewayError};
use daemon_node::ProviderResolver;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

/// The stale-stream watchdog applied to a non-streaming gateway call (a provider silent longer than
/// this is a recoverable transport failure — see [`drive_model_call`]).
const GATEWAY_WATCHDOG: Duration = Duration::from_secs(300);

/// The gateway backend seams captured from the host bootstrap *before* `provider_resolver` is moved
/// into the node assembly. The node surface is bound later (post-assembly) via [`Self::into_backend`].
pub struct GatewaySeams {
    /// The shared provider-resolution seam (a clone of the node's `provider_resolver`).
    pub provider_resolver: ProviderResolver,
    /// A brokered credential client over the node's owner authority.
    pub credentials: Arc<dyn CredentialProvider>,
    /// The per-provider credential map from `[gateway].credentials` (a small assoc list; a linear
    /// lookup is cheap and avoids requiring `Hash` on the wire enum).
    pub cred_map: Vec<(ProviderSelector, String)>,
    /// The fallback credential profile (the node default profile).
    pub default_credential_ref: String,
    /// The optional model allowlist from `[gateway].models_allowlist`.
    pub allowlist: Option<Vec<String>>,
}

impl GatewaySeams {
    /// Bind the node surface to complete the backend (post-assembly).
    pub fn into_backend(self, node: Arc<dyn NodeApi>) -> NodeGatewayBackend {
        NodeGatewayBackend::new(
            node,
            self.provider_resolver,
            self.credentials,
            self.cred_map,
            self.default_credential_ref,
            self.allowlist,
        )
    }
}

/// The node-side gateway backend. Holds the node surface (for the catalog), the provider-resolution
/// seam, the credential provider (a brokered client over the node's owner authority), and the
/// per-provider credential map + optional model allowlist from `[gateway]`.
pub struct NodeGatewayBackend {
    node: Arc<dyn NodeApi>,
    provider_resolver: ProviderResolver,
    credentials: Arc<dyn CredentialProvider>,
    /// `provider -> credential_ref` for cloud providers (a small assoc list).
    cred_map: Vec<(ProviderSelector, String)>,
    /// The fallback credential profile for a cloud provider with no explicit `cred_map` entry (the
    /// node default profile, which already holds `DAEMON_CLOUD_API_KEY`).
    default_credential_ref: String,
    /// The optional model-id allowlist bounding the catalog + routing.
    allowlist: Option<HashSet<String>>,
}

impl NodeGatewayBackend {
    /// Build the backend from the node surface + the resolution/credential seams + the `[gateway]`
    /// credential map / allowlist.
    pub fn new(
        node: Arc<dyn NodeApi>,
        provider_resolver: ProviderResolver,
        credentials: Arc<dyn CredentialProvider>,
        cred_map: Vec<(ProviderSelector, String)>,
        default_credential_ref: String,
        allowlist: Option<Vec<String>>,
    ) -> Self {
        Self {
            node,
            provider_resolver,
            credentials,
            cred_map,
            default_credential_ref,
            allowlist: allowlist.map(|list| list.into_iter().collect()),
        }
    }

    /// The credential profile a cloud provider acquires from: its `cred_map` entry, else the node
    /// default profile.
    fn credential_ref_for(&self, provider: ProviderSelector) -> String {
        self.cred_map
            .iter()
            .find(|(p, _)| *p == provider)
            .map(|(_, r)| r.clone())
            .unwrap_or_else(|| self.default_credential_ref.clone())
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

    async fn complete(
        &self,
        model: &str,
        mut req: Request,
        stream: bool,
    ) -> Result<Completion, GatewayError> {
        // Resolve the requested model id to its provider via the node catalog (404 if unknown or
        // outside the allowlist).
        let provider = self
            .catalog()
            .await
            .into_iter()
            .find(|m| m.id == model)
            .map(|m| m.provider)
            .ok_or_else(|| GatewayError::UnknownModel(model.to_string()))?;

        // Synthesize an ephemeral profile: the resolved provider + the requested model + the
        // per-provider credential ref (cloud only). No persona/tools/memory — this is a bare model
        // call, not a session.
        let credential_ref = self.credential_ref_for(provider);
        let spec = ProfileSpec {
            credential_ref: Some(credential_ref),
            ..ProfileSpec::new("gateway", provider, model.to_string())
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
