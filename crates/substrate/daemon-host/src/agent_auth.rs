// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The dynamic `agent/*` interactive-auth namespace (wire v47): ONE resolver serves a login flow
//! for every cataloged foreign agent, minted on demand from the agent's `auth_descriptor` — the
//! catalog is open-ended (families appear when an agent is registered), so these factories cannot
//! be fixed at assembly like the transport/provider ones.
//!
//! Two schemes are served:
//!
//! * `ApiKeyEnv` — a one-step form flow (the [`daemon_api::AuthFlowKind::UserToken`] shape): the
//!   operator pastes the vendor API key, and the completed outcome stores it under the
//!   agent-scoped credential ref `agent/<name>/<VAR>` — where the spawn-time materialization
//!   (A4) reads it back into the agent's environment.
//! * `AcpAuthenticate` — the agent's own advertised ACP method: a message + poll flow whose poll
//!   drives the [`AcpAuthGateway`] seam (spawn the agent, run `authenticate(method_id)`). ACP's
//!   `authenticate` returns no credential — the agent persists its own state through side effects
//!   — so completion stores a node-owned SUCCESS MARKER under `agent/<name>/acp:<method_id>`,
//!   the only derivable fact behind a later `Authenticated` verdict.
//!
//! Deliberately NOT served: `OAuthFamily` / `DeviceCode` (research-gated vendor flows — their
//! dedicated factories register under the exact family id and shadow this namespace) and scheme
//! `None` (nothing to run). Method selection: exactly one advertised method auto-selects;
//! multiple require the client to pass `method_id` in the `auth_begin` params.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{
    AgentAuthScheme, AgentEntry, ApiError, AuthChallenge, AuthFieldKind, AuthFlowKind,
    AuthParamField, AuthProviderInfo, AuthStepInput,
};
use daemon_protocol::TransportId;
use daemon_store::SessionStore;

use crate::auth::{
    AuthFlowFactory, AuthOutcome, AuthStepOutcome, CredentialSlotKind, DynamicAuthFamilies,
    PendingAuthFlow,
};

/// The family-id prefix of the dynamic agent namespace.
pub const AGENT_FAMILY_PREFIX: &str = "agent/";

/// The seam through which the `agent/*` resolver runs an agent's own ACP `authenticate` method.
/// Implemented by the ACP adapter (`daemon-host` cannot link the ACP runtime — `daemon-acp`
/// depends on it); injected at node assembly next to the discoverer.
#[async_trait]
pub trait AcpAuthGateway: Send + Sync {
    /// Spawn the agent from its catalog recipe and run `authenticate(method_id)` to completion.
    /// `Ok(())` means the agent acknowledged the method (it persisted whatever state it needed
    /// through its own side effects); the error string is operator-facing.
    async fn authenticate(&self, entry: &AgentEntry, method_id: &str) -> Result<(), ApiError>;
}

/// The `agent/*` namespace resolver: looks the agent up exactly as the catalog does (a durable
/// manual registration wins over the in-memory discovery row of the same name) and mints the
/// scheme-appropriate factory.
pub struct AgentAuthFamilies {
    store: Arc<dyn SessionStore>,
    last_agents: Arc<std::sync::RwLock<Vec<AgentEntry>>>,
    acp: Option<Arc<dyn AcpAuthGateway>>,
}

impl AgentAuthFamilies {
    /// A resolver over the node's catalog state. `acp` is optional: without the gateway,
    /// `AcpAuthenticate` families do not resolve (ApiKeyEnv ones still do).
    pub fn new(
        store: Arc<dyn SessionStore>,
        last_agents: Arc<std::sync::RwLock<Vec<AgentEntry>>>,
        acp: Option<Arc<dyn AcpAuthGateway>>,
    ) -> Self {
        Self {
            store,
            last_agents,
            acp,
        }
    }

    /// The catalog row for `name`: the durable manual registration wins, else the last discovery
    /// pass. Mirrors `agent_catalog`'s precedence without the full assembly.
    async fn entry(&self, name: &str) -> Option<AgentEntry> {
        for stored in self.store.acp_list().await {
            if stored.name == name {
                if let Ok(entry) = daemon_api::from_cbor::<AgentEntry>(&stored.entry) {
                    return Some(entry);
                }
            }
        }
        self.last_agents
            .read()
            .unwrap()
            .iter()
            .find(|e| e.name == name)
            .cloned()
    }
}

#[async_trait]
impl DynamicAuthFamilies for AgentAuthFamilies {
    fn prefix(&self) -> &str {
        AGENT_FAMILY_PREFIX
    }

    async fn resolve(&self, family: &str) -> Option<Arc<dyn AuthFlowFactory>> {
        let name = family.strip_prefix(AGENT_FAMILY_PREFIX)?;
        // A name that violates the registration slug never resolves: it cannot safely become a
        // credential ref / state path segment (mirrors the catalog's derivation exclusion).
        if !daemon_api::is_valid_agent_slug(name) {
            return None;
        }
        let entry = self.entry(name).await?;
        let descriptor = entry.auth_descriptor.clone()?;
        match descriptor.scheme {
            AgentAuthScheme::ApiKeyEnv { var, label } => Some(Arc::new(ApiKeyFlowFactory {
                family: family.to_string(),
                agent: name.to_string(),
                var,
                label,
            })),
            AgentAuthScheme::AcpAuthenticate => {
                let acp = self.acp.clone()?;
                // The advertised methods captured at probe time (stashed on the entry); a manual
                // descriptor without a captured list gets the synthetic singleton the catalog
                // derivation also offers.
                let methods: Vec<(String, String)> = entry
                    .auth
                    .as_ref()
                    .map(|a| {
                        a.methods
                            .iter()
                            .map(|m| (m.id.clone(), m.label.clone()))
                            .collect()
                    })
                    .filter(|m: &Vec<_>| !m.is_empty())
                    .unwrap_or_else(|| vec![("acp".into(), "Authenticate".into())]);
                Some(Arc::new(AcpFlowFactory {
                    family: family.to_string(),
                    agent: name.to_string(),
                    entry,
                    methods,
                    acp,
                }))
            }
            // Vendor flows are exact-registered dedicated families (shadowing this namespace);
            // scheme None means nothing to run. Neither resolves dynamically.
            AgentAuthScheme::OAuthFamily { .. }
            | AgentAuthScheme::DeviceCode { .. }
            | AgentAuthScheme::None => None,
        }
    }
}

/// The ApiKeyEnv factory for one agent: a single-form flow that stores the pasted key under the
/// agent-scoped credential ref.
struct ApiKeyFlowFactory {
    family: String,
    agent: String,
    var: String,
    label: String,
}

#[async_trait]
impl AuthFlowFactory for ApiKeyFlowFactory {
    fn family(&self) -> &str {
        &self.family
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: self.family.clone(),
            flow_kind: AuthFlowKind::UserToken,
            display_name: format!("{} sign-in", self.agent),
            // The key is collected via the initial Form challenge, not begin params, so the same
            // AuthFlowBody challenge rendering serves every scheme.
            params_schema: Vec::new(),
        }
    }

    async fn begin(
        &self,
        _params: &BTreeMap<String, String>,
        _redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        Ok(Box::new(ApiKeyFlow {
            agent: self.agent.clone(),
            var: self.var.clone(),
            label: self.label.clone(),
        }))
    }
}

/// One pasted-key flow: Form challenge -> Completed outcome.
struct ApiKeyFlow {
    agent: String,
    var: String,
    label: String,
}

#[async_trait]
impl PendingAuthFlow for ApiKeyFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        AuthChallenge::Form {
            title: format!("Enter your {}", self.label),
            fields: vec![AuthParamField {
                key: "key".into(),
                label: self.label.clone(),
                required: true,
                kind: AuthFieldKind::Password,
                default: None,
                placeholder: None,
                choices: Vec::new(),
            }],
        }
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        let AuthStepInput::Fields(fields) = input else {
            return Err(ApiError::Other("this flow expects form fields".into()));
        };
        let key = fields
            .get("key")
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| ApiError::Other(format!("{} is required", self.label)))?;
        Ok(AuthStepOutcome::Completed(AuthOutcome {
            credential_blob: key.clone(),
            // The agent-scoped ref the catalog derivation and the spawn-time materialization
            // (A4) both read: `agent/<name>/<VAR>`.
            credential_ref: format!("agent/{}/{}", self.agent, self.var),
            account_label: self.agent.clone(),
            transport_instance: TransportId::new(format!("agent/{}", self.agent)),
            slot: CredentialSlotKind::Derived,
        }))
    }
}

/// The AcpAuthenticate factory for one agent: method selection at `begin`, the gateway call at
/// the first poll.
struct AcpFlowFactory {
    family: String,
    agent: String,
    entry: AgentEntry,
    /// The runnable `(id, label)` methods (probe-captured, or the synthetic singleton).
    methods: Vec<(String, String)>,
    acp: Arc<dyn AcpAuthGateway>,
}

#[async_trait]
impl AuthFlowFactory for AcpFlowFactory {
    fn family(&self) -> &str {
        &self.family
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: self.family.clone(),
            flow_kind: AuthFlowKind::AcpAuthenticate,
            display_name: format!("{} sign-in", self.agent),
            params_schema: Vec::new(),
        }
    }

    async fn begin(
        &self,
        params: &BTreeMap<String, String>,
        _redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        // Method selection: exactly one advertised method auto-selects; multiple require the
        // client to name one (`method_id` in the begin params) — the node never guesses which
        // vendor ceremony the operator meant.
        let (method_id, method_label) = match params.get("method_id") {
            Some(requested) => self
                .methods
                .iter()
                .find(|(id, _)| id == requested)
                .cloned()
                .ok_or_else(|| {
                    ApiError::Other(format!(
                        "unknown auth method `{requested}` for {} (offered: {})",
                        self.agent,
                        self.methods
                            .iter()
                            .map(|(id, _)| id.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?,
            None if self.methods.len() == 1 => self.methods[0].clone(),
            None => {
                return Err(ApiError::Other(format!(
                    "{} offers multiple auth methods — pass method_id (one of: {})",
                    self.agent,
                    self.methods
                        .iter()
                        .map(|(id, _)| id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )))
            }
        };
        Ok(Box::new(AcpAuthFlow {
            agent: self.agent.clone(),
            entry: self.entry.clone(),
            method_id,
            method_label,
            acp: self.acp.clone(),
        }))
    }
}

/// One ACP authenticate flow: an informational Message challenge, then a poll that runs the
/// agent's `authenticate(method_id)` through the gateway and completes with the marker outcome.
struct AcpAuthFlow {
    agent: String,
    entry: AgentEntry,
    method_id: String,
    method_label: String,
    acp: Arc<dyn AcpAuthGateway>,
}

#[async_trait]
impl PendingAuthFlow for AcpAuthFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        AuthChallenge::Message {
            text: format!(
                "Signing in to {} via {} — the agent may open a browser or prompt on this \
                 machine. Continue when ready.",
                self.agent, self.method_label
            ),
        }
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        let AuthStepInput::Poll = input else {
            return Err(ApiError::Other("this flow expects a poll".into()));
        };
        // The gateway spawns the agent and runs its own `authenticate` ceremony; an error leaves
        // the flow parked for a retry poll (the registry keeps challenge-state flows parked).
        self.acp.authenticate(&self.entry, &self.method_id).await?;
        Ok(AuthStepOutcome::Completed(AuthOutcome {
            // ACP returns no credential — this blob is a node-owned SUCCESS MARKER, never used
            // as secret material; presence under the `acp:` ref is the derivable fact.
            credential_blob: format!("acp-authenticated:{}", self.method_id),
            credential_ref: format!("agent/{}/acp:{}", self.agent, self.method_id),
            account_label: self.agent.clone(),
            transport_instance: TransportId::new(format!("agent/{}", self.agent)),
            slot: CredentialSlotKind::Derived,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::{
        AgentAuth, AgentAuthDescriptor, AgentAuthMethod, AgentAuthState, AgentProtocol,
        AgentRecipe, AgentSource, AgentVerification,
    };
    use daemon_store::InMemoryStore;

    fn entry_with(descriptor: Option<AgentAuthDescriptor>, name: &str) -> AgentEntry {
        AgentEntry {
            name: name.into(),
            recipe: AgentRecipe {
                program: Some("/nonexistent".into()),
                args: Vec::new(),
                env: Vec::new(),
                endpoint: None,
            },
            source: AgentSource::Manual,
            protocol: AgentProtocol::StreamJson,
            installed: false,
            version: None,
            capabilities: Vec::new(),
            verification: AgentVerification::NotInstalled,
            auth_descriptor: descriptor,
            auth: None,
        }
    }

    struct NoopGateway;
    #[async_trait]
    impl AcpAuthGateway for NoopGateway {
        async fn authenticate(&self, _: &AgentEntry, _: &str) -> Result<(), ApiError> {
            Ok(())
        }
    }

    fn resolver_with(entries: Vec<AgentEntry>, acp: bool) -> AgentAuthFamilies {
        let last = Arc::new(std::sync::RwLock::new(entries));
        let gateway: Option<Arc<dyn AcpAuthGateway>> =
            acp.then(|| Arc::new(NoopGateway) as Arc<dyn AcpAuthGateway>);
        AgentAuthFamilies::new(Arc::new(InMemoryStore::new()), last, gateway)
    }

    /// ApiKeyEnv resolves to a form flow whose completion lands the pasted key under the
    /// agent-scoped ref `agent/<name>/<VAR>` (the A4 materialization contract).
    #[tokio::test]
    async fn api_key_family_resolves_and_completes_with_the_scoped_ref() {
        let descriptor = AgentAuthDescriptor {
            scheme: AgentAuthScheme::ApiKeyEnv {
                var: "AMP_API_KEY".into(),
                label: "Amp API key".into(),
            },
            rejection: None,
        };
        let resolver = resolver_with(vec![entry_with(Some(descriptor), "amp")], false);
        let factory = resolver
            .resolve("agent/amp")
            .await
            .expect("ApiKeyEnv resolves without a gateway");
        assert_eq!(factory.family(), "agent/amp");
        assert_eq!(factory.provider_info().flow_kind, AuthFlowKind::UserToken);

        let flow = factory
            .begin(&BTreeMap::new(), "http://127.0.0.1:0/cb")
            .await
            .unwrap();
        assert!(matches!(
            flow.initial_challenge(),
            AuthChallenge::Form { .. }
        ));
        let mut fields = BTreeMap::new();
        fields.insert("key".to_string(), "sk-amp-test".to_string());
        match flow.step(AuthStepInput::Fields(fields)).await.unwrap() {
            AuthStepOutcome::Completed(outcome) => {
                assert_eq!(outcome.credential_ref, "agent/amp/AMP_API_KEY");
                assert_eq!(outcome.credential_blob, "sk-amp-test");
                assert_eq!(outcome.slot, CredentialSlotKind::Derived);
            }
            AuthStepOutcome::Challenge(_) => panic!("a pasted key completes in one step"),
        }
    }

    /// AcpAuthenticate method selection: one method auto-selects; multiple require `method_id`;
    /// an unknown `method_id` errors with the offered ids. Completion writes the `acp:` marker.
    #[tokio::test]
    async fn acp_family_selects_methods_and_completes_with_the_marker() {
        let mut entry = entry_with(
            Some(AgentAuthDescriptor {
                scheme: AgentAuthScheme::AcpAuthenticate,
                rejection: None,
            }),
            "gemini",
        );
        entry.auth = Some(AgentAuth {
            state: AgentAuthState::Unknown,
            methods: vec![
                AgentAuthMethod {
                    id: "oauth".into(),
                    label: "Log in with Google".into(),
                    kind: AuthFlowKind::AcpAuthenticate,
                    family: "agent/gemini".into(),
                },
                AgentAuthMethod {
                    id: "api-key".into(),
                    label: "API key".into(),
                    kind: AuthFlowKind::AcpAuthenticate,
                    family: "agent/gemini".into(),
                },
            ],
        });
        let resolver = resolver_with(vec![entry], true);
        let factory = resolver.resolve("agent/gemini").await.expect("resolves");

        // Two methods, none named: refused with the offered ids.
        let err = factory
            .begin(&BTreeMap::new(), "http://127.0.0.1:0/cb")
            .await
            .err()
            .expect("multiple methods require method_id");
        assert!(err.to_string().contains("oauth") && err.to_string().contains("api-key"));

        // Unknown method: refused.
        let mut params = BTreeMap::new();
        params.insert("method_id".to_string(), "bogus".to_string());
        assert!(factory
            .begin(&params, "http://127.0.0.1:0/cb")
            .await
            .is_err());

        // A named method runs: Message challenge, then the poll completes with the marker.
        params.insert("method_id".to_string(), "oauth".to_string());
        let flow = factory
            .begin(&params, "http://127.0.0.1:0/cb")
            .await
            .unwrap();
        assert!(matches!(
            flow.initial_challenge(),
            AuthChallenge::Message { .. }
        ));
        match flow.step(AuthStepInput::Poll).await.unwrap() {
            AuthStepOutcome::Completed(outcome) => {
                assert_eq!(outcome.credential_ref, "agent/gemini/acp:oauth");
            }
            AuthStepOutcome::Challenge(_) => panic!("the poll completes"),
        }
    }

    /// The namespace refuses what it must: unknown agents, descriptor-less rows, vendor schemes
    /// (exact-registered dedicated families), AcpAuthenticate without a gateway, non-slug names.
    #[tokio::test]
    async fn unservable_families_do_not_resolve() {
        let oauth = AgentAuthDescriptor {
            scheme: AgentAuthScheme::OAuthFamily {
                family: "agent/claude".into(),
            },
            rejection: None,
        };
        let acp_descriptor = AgentAuthDescriptor {
            scheme: AgentAuthScheme::AcpAuthenticate,
            rejection: None,
        };
        let resolver = resolver_with(
            vec![
                entry_with(None, "plain"),
                entry_with(Some(oauth), "claude"),
                entry_with(Some(acp_descriptor), "gemini"),
            ],
            false, // no gateway
        );
        assert!(resolver.resolve("agent/unknown").await.is_none());
        assert!(resolver.resolve("agent/plain").await.is_none());
        assert!(resolver.resolve("agent/claude").await.is_none());
        assert!(resolver.resolve("agent/gemini").await.is_none());
        assert!(resolver.resolve("agent/UPPER").await.is_none());
        assert!(resolver.resolve("not-the-namespace").await.is_none());
    }
}
