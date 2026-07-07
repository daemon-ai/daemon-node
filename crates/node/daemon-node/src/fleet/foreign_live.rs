// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Foreign-engine resolution for the LIVE interactive session seam: a profile whose
//! `engine = Foreign { agent }` resolves its catalog entry BY NAME at spawn time and materializes
//! the protocol-appropriate backend behind the host's transport-agnostic `AgentSession` seam —
//! a [`daemon_acp::AcpSession`] for `AgentProtocol::Acp`, or a stream-json child process driven by
//! the generic [`CodecSession`] for `AgentProtocol::StreamJson`.
//!
//! The security invariant lives here: profiles carry only the catalog NAME. The launch recipe is
//! read from the node's own sources — the durable manual registrations (`agent_register`, operator
//! authz) and the curated builtin table — never from the profile, so a `ProfileCreate` can never
//! smuggle an arbitrary binary spawn. Installed-ness is re-checked at spawn (it can change after
//! profile validation); a vanished or uninstalled agent fails the session open with a clear
//! [`ApiError`] instead of a dead actor.

use std::path::PathBuf;
use std::sync::Arc;

use daemon_api::{from_cbor, AgentEntry, AgentProtocol, ApiError};
use daemon_common::SessionId;
use daemon_host::{
    AgentDiscovery, AgentSession, CodecSession, ForeignSessionFactory, StreamJsonCodec,
};
use daemon_protocol::{AgentCommand, AgentEvent};
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_store::SessionStore;
use tokio::sync::broadcast;

use crate::{GatewayBinding, GatewayCoords, GatewayLease};

/// An [`AgentSession`] decorator that owns a [`GatewayLease`] for a `NodeProvider`-routed foreign
/// session: it forwards every §17 command/subscribe to the inner session and, when it is dropped
/// (the session closed / the node shut down), the held lease revokes the per-session gateway token.
/// This is how "mint on session open, revoke on close" is tied to the session's lifetime without
/// touching the protocol-specific `AcpSession`/`CodecSession` backends.
struct GatewayLeasedSession {
    inner: Arc<dyn AgentSession>,
    /// Revokes the per-session gateway token on drop. Never read; held for its `Drop`.
    _lease: GatewayLease,
}

#[async_trait::async_trait]
impl AgentSession for GatewayLeasedSession {
    async fn submit(&self, cmd: AgentCommand) {
        self.inner.submit(cmd).await;
    }

    fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.inner.subscribe()
    }

    fn rewindable(&self) -> bool {
        self.inner.rewindable()
    }
}

/// Wrap a produced foreign session in a [`GatewayLeasedSession`] when a per-session gateway lease
/// was minted for it, so the token is revoked when the session drops; a `None` lease (AgentNative,
/// or a non-routed session) returns the session unchanged.
fn with_lease(inner: Arc<dyn AgentSession>, lease: Option<GatewayLease>) -> Arc<dyn AgentSession> {
    match lease {
        Some(lease) => Arc::new(GatewayLeasedSession {
            inner,
            _lease: lease,
        }),
        None => inner,
    }
}

/// Build the deferred foreign-session factory for a live session bound to catalog agent `agent`:
/// the factory resolves the recipe (durable manual registration first, curated builtin fallback —
/// exactly the `agent_catalog` precedence), re-checks installed-ness, and materializes the
/// protocol-appropriate backend with the host's parking handler answering its permission
/// callbacks (ACP `session/request_permission` and stream-json `control_request` both park as
/// ordinary host approval requests).
///
/// `model` steers the agent's own backend via ACP (the `AgentNative` path); `extra_env` repoints an
/// OpenAI-wire agent at the node gateway (the `NodeProvider` path) and `lease` (when present) ties
/// the injected per-session token's lifetime to the produced session.
pub(crate) fn foreign_session_factory(
    agent: String,
    model: Option<String>,
    session: SessionId,
    store: Arc<dyn SessionStore>,
    extra_env: Vec<(String, String)>,
    lease: Option<GatewayLease>,
) -> ForeignSessionFactory {
    Box::new(move |host| {
        Box::pin(async move {
            let mut entry = resolve_entry(&agent, &store).await?;
            // NodeProvider injection: append the per-session gateway env (never replacing the
            // catalog recipe — only adding env). Empty for AgentNative and non-OpenAI-wire agents,
            // so the recipe-by-name security invariant is preserved.
            if !extra_env.is_empty() {
                entry.recipe.env.extend(extra_env);
            }
            // Spawn-time re-check: validation at create/update proved installed-ness THEN; the
            // binary can have been removed since. `recipe_installed` is a cheap PATH/file probe
            // (protocol-independent — both ACP and stream-json agents are PATH binaries).
            if !daemon_acp::recipe_installed(&entry.recipe) {
                return Err(ApiError::Other(format!(
                    "agent `{agent}` is not installed (its catalog recipe no longer resolves \
                     to a runnable program)"
                )));
            }
            match entry.protocol {
                AgentProtocol::Acp => {
                    let launch = daemon_acp::launch_from_recipe(&entry.recipe)
                        .ok_or_else(|| {
                            ApiError::Unsupported(format!(
                                "agent `{agent}` has an endpoint-only recipe; endpoint agents are \
                             not spawnable as session engines yet"
                            ))
                        })?
                        // AgentNative model steer: steer the agent to the profile's
                        // node-validated model (best-effort inside daemon-acp after session/new);
                        // the recipe still comes only from the catalog by name.
                        .model(model);
                    Ok(with_lease(
                        daemon_acp::AcpSession::connect(launch, host),
                        lease,
                    ))
                }
                AgentProtocol::StreamJson => {
                    let program = entry.recipe.program.clone().ok_or_else(|| {
                        ApiError::Unsupported(format!(
                            "agent `{agent}` has an endpoint-only recipe; endpoint agents are \
                             not spawnable as session engines yet"
                        ))
                    })?;
                    // NDJSON over the line transport, driven by the generic codec session driver —
                    // the same provisioner + codec wiring the fleet's `ProfileChildSpawner` uses
                    // for `ForeignProtocol::StreamJson` children.
                    let spec = PlacementSpec {
                        program: PathBuf::from(program),
                        args: entry.recipe.args.clone(),
                        env: entry.recipe.env.clone(),
                    };
                    let placement = ProcessProvisioner
                        .place_lines(&session, spec)
                        .await
                        .map_err(|e| {
                            ApiError::Other(format!(
                                "spawning stream-json agent `{agent}` failed: {e}"
                            ))
                        })?;
                    let daemon_provision::Placement { channel, child } = placement;
                    let session = Arc::new(CodecSession::from_channel(
                        channel,
                        Some(child),
                        host,
                        StreamJsonCodec::new(),
                        Some(agent.clone()),
                    )) as Arc<dyn AgentSession>;
                    Ok(with_lease(session, lease))
                }
            }
        })
    })
}

/// The OpenAI-wire foreign agents that honor `OPENAI_BASE_URL` / `OPENAI_API_KEY` env repointing.
/// Deliberately a small, explicit allowlist (start with codex + opencode): a non-OpenAI-wire agent
/// (claude/gemini) is never repointed here — its native backend is left untouched (the Anthropic
/// `/v1/messages` route is a documented follow-up).
pub(crate) fn is_openai_wire_agent(agent: &str) -> bool {
    matches!(agent, "codex" | "opencode")
}

/// Build the `NodeProvider` gateway injection for `agent`: mint a PER-SESSION gateway token bound to
/// `binding` (via [`GatewayCoords::minter`]) and return the OpenAI-wire env repointing the agent at
/// the node gateway (`OPENAI_BASE_URL` + the minted `OPENAI_API_KEY` + `OPENAI_MODEL`) plus a
/// [`GatewayLease`] that revokes the token when the session drops. A non-OpenAI-wire agent is never
/// repointed (empty env, no lease minted) — its native backend is left untouched, so no orphan
/// token is created.
///
/// The agent only ever holds the opaque loopback token; the real provider credential is resolved
/// node-side by the gateway from the token's binding, preserving the keys-stay-node-side invariant.
pub(crate) fn node_provider_injection(
    agent: &str,
    coords: &GatewayCoords,
    binding: GatewayBinding,
) -> (Vec<(String, String)>, Option<GatewayLease>) {
    if !is_openai_wire_agent(agent) {
        return (Vec::new(), None);
    }
    let model = binding.model.clone();
    let token = coords.minter.mint(binding);
    let mut env = vec![
        ("OPENAI_BASE_URL".to_string(), coords.base_url.clone()),
        ("OPENAI_API_KEY".to_string(), token.clone()),
    ];
    if !model.is_empty() {
        env.push(("OPENAI_MODEL".to_string(), model));
    }
    let lease = GatewayLease::new(coords.minter.clone(), token);
    (env, Some(lease))
}

/// Resolve `agent` to its catalog entry: durable manual registrations take precedence over the
/// curated builtin table (mirrors the `agent_catalog` merge order — Manual wins over Builtin).
async fn resolve_entry(agent: &str, store: &Arc<dyn SessionStore>) -> Result<AgentEntry, ApiError> {
    for stored in store.acp_list().await {
        if stored.name != agent {
            continue;
        }
        if let Ok(entry) = from_cbor::<AgentEntry>(&stored.entry) {
            return Ok(entry);
        }
    }
    daemon_acp::AcpDiscoverer::new()
        .builtin(agent)
        .ok_or_else(|| {
            ApiError::Other(format!(
                "profile engine references unknown agent `{agent}` — register it via \
                 agent_register or run AgentDiscover first"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GatewayTokenMinter;
    use daemon_api::ProviderSelector;
    use std::sync::Mutex;

    /// A test minter recording the live token→binding table: `mint` registers a deterministic
    /// token, `revoke` drops it, so a test can assert both the injected env and the revoke-on-drop.
    #[derive(Default)]
    struct RecordingMinter {
        table: Mutex<std::collections::HashMap<String, GatewayBinding>>,
        next: std::sync::atomic::AtomicU64,
    }

    impl GatewayTokenMinter for RecordingMinter {
        fn mint(&self, binding: GatewayBinding) -> String {
            let n = self.next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let token = format!("sess-token-{n}");
            self.table.lock().unwrap().insert(token.clone(), binding);
            token
        }

        fn revoke(&self, token: &str) {
            self.table.lock().unwrap().remove(token);
        }
    }

    fn coords(minter: Arc<RecordingMinter>) -> GatewayCoords {
        GatewayCoords {
            base_url: "http://127.0.0.1:8081/v1".into(),
            minter,
        }
    }

    fn binding(model: &str) -> GatewayBinding {
        GatewayBinding {
            provider: ProviderSelector::GenAi,
            model: model.into(),
            credential_ref: None,
        }
    }

    #[test]
    fn openai_wire_agents_get_a_per_session_gateway_token() {
        for agent in ["codex", "opencode"] {
            let minter = Arc::new(RecordingMinter::default());
            let (env, lease) =
                node_provider_injection(agent, &coords(minter.clone()), binding("gpt-4o"));
            let map: std::collections::HashMap<_, _> = env.into_iter().collect();
            assert_eq!(
                map.get("OPENAI_BASE_URL").map(String::as_str),
                Some("http://127.0.0.1:8081/v1"),
                "{agent} should get OPENAI_BASE_URL"
            );
            let token = map
                .get("OPENAI_API_KEY")
                .cloned()
                .expect("a per-session token is injected");
            assert_eq!(
                map.get("OPENAI_MODEL").map(String::as_str),
                Some("gpt-4o"),
                "{agent} should get OPENAI_MODEL"
            );
            // The token is registered against its binding while the lease is alive...
            let lease = lease.expect("an openai-wire agent mints a lease");
            assert_eq!(
                minter.table.lock().unwrap().get(&token),
                Some(&binding("gpt-4o")),
                "the minted token resolves to its routing binding"
            );
            // ...and revoked when the lease (i.e. the session) drops — keys never outlive the session.
            drop(lease);
            assert!(
                !minter.table.lock().unwrap().contains_key(&token),
                "dropping the lease revokes the per-session token"
            );
        }
    }

    #[test]
    fn non_openai_wire_agents_are_not_repointed_and_mint_nothing() {
        // claude (Anthropic) / gemini and unknown agents keep their own backend — no injection, and
        // crucially no orphan token is minted.
        for agent in ["claude", "gemini", "goose", "unknown"] {
            let minter = Arc::new(RecordingMinter::default());
            let (env, lease) =
                node_provider_injection(agent, &coords(minter.clone()), binding("gpt-4o"));
            assert!(env.is_empty(), "{agent} must not be repointed");
            assert!(lease.is_none(), "{agent} must not mint a token");
            assert!(
                minter.table.lock().unwrap().is_empty(),
                "{agent} must leave the registry untouched"
            );
        }
    }

    #[test]
    fn empty_model_omits_model_hint() {
        let minter = Arc::new(RecordingMinter::default());
        let (env, _lease) = node_provider_injection("codex", &coords(minter), binding(""));
        let map: std::collections::HashMap<_, _> = env.into_iter().collect();
        assert!(map.contains_key("OPENAI_BASE_URL"));
        assert!(map.contains_key("OPENAI_API_KEY"));
        // No model hint when the routing carries no model.
        assert!(!map.contains_key("OPENAI_MODEL"));
    }
}
