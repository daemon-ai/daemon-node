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
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_store::SessionStore;

/// Build the deferred foreign-session factory for a live session bound to catalog agent `agent`:
/// the factory resolves the recipe (durable manual registration first, curated builtin fallback —
/// exactly the `agent_catalog` precedence), re-checks installed-ness, and materializes the
/// protocol-appropriate backend with the host's parking handler answering its permission
/// callbacks (ACP `session/request_permission` and stream-json `control_request` both park as
/// ordinary host approval requests).
pub(crate) fn foreign_session_factory(
    agent: String,
    session: SessionId,
    store: Arc<dyn SessionStore>,
    extra_env: Vec<(String, String)>,
) -> ForeignSessionFactory {
    Box::new(move |host| {
        Box::pin(async move {
            let mut entry = resolve_entry(&agent, &store).await?;
            // Layer 2 injection: append the operator-opted-in gateway env (never replacing the
            // catalog recipe — only adding env). Empty when the gateway is off or the agent is not
            // an OpenAI-wire agent, so the recipe-by-name security invariant is preserved.
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
                    let launch =
                        daemon_acp::launch_from_recipe(&entry.recipe).ok_or_else(|| {
                            ApiError::Unsupported(format!(
                                "agent `{agent}` has an endpoint-only recipe; endpoint agents are \
                             not spawnable as session engines yet"
                            ))
                        })?;
                    Ok(daemon_acp::AcpSession::connect(launch, host))
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
                    Ok(Arc::new(CodecSession::from_channel(
                        channel,
                        Some(child),
                        host,
                        StreamJsonCodec::new(),
                        Some(agent.clone()),
                    )) as Arc<dyn AgentSession>)
                }
            }
        })
    })
}

/// The OpenAI-wire foreign agents that honor `OPENAI_BASE_URL` / `OPENAI_API_KEY` env repointing.
/// Deliberately a small, explicit allowlist (start with codex + opencode): a non-OpenAI-wire agent
/// (claude/gemini) is never repointed here — its native backend is left untouched (the Anthropic
/// `/v1/messages` route is a documented follow-up).
fn is_openai_wire_agent(agent: &str) -> bool {
    matches!(agent, "codex" | "opencode")
}

/// The gateway env vars to inject for `agent`, given the node gateway [`GatewayCoords`] and the
/// profile `model`. Empty for a non-OpenAI-wire agent (no injection). This is the per-agent mapping
/// table: OpenAI-wire agents read `OPENAI_BASE_URL` + `OPENAI_API_KEY` (and `OPENAI_MODEL` as a
/// hint) so they run on the node-configured provider without holding a real key.
pub(crate) fn foreign_gateway_env(
    agent: &str,
    coords: &crate::GatewayCoords,
    model: &str,
) -> Vec<(String, String)> {
    if !is_openai_wire_agent(agent) {
        return Vec::new();
    }
    let mut env = vec![
        ("OPENAI_BASE_URL".to_string(), coords.base_url.clone()),
        ("OPENAI_API_KEY".to_string(), coords.token.clone()),
    ];
    if !model.is_empty() {
        env.push(("OPENAI_MODEL".to_string(), model.to_string()));
    }
    env
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
    use crate::GatewayCoords;

    fn coords() -> GatewayCoords {
        GatewayCoords {
            base_url: "http://127.0.0.1:8081/v1".into(),
            token: "gw-token".into(),
        }
    }

    #[test]
    fn openai_wire_agents_get_gateway_env() {
        for agent in ["codex", "opencode"] {
            let env = foreign_gateway_env(agent, &coords(), "gpt-4o");
            let map: std::collections::HashMap<_, _> = env.into_iter().collect();
            assert_eq!(
                map.get("OPENAI_BASE_URL").map(String::as_str),
                Some("http://127.0.0.1:8081/v1"),
                "{agent} should get OPENAI_BASE_URL"
            );
            assert_eq!(
                map.get("OPENAI_API_KEY").map(String::as_str),
                Some("gw-token"),
                "{agent} should get OPENAI_API_KEY"
            );
            assert_eq!(
                map.get("OPENAI_MODEL").map(String::as_str),
                Some("gpt-4o"),
                "{agent} should get OPENAI_MODEL"
            );
        }
    }

    #[test]
    fn non_openai_wire_agents_are_not_repointed() {
        // claude (Anthropic) / gemini and unknown agents keep their own backend — no injection.
        for agent in ["claude", "gemini", "goose", "unknown"] {
            assert!(
                foreign_gateway_env(agent, &coords(), "gpt-4o").is_empty(),
                "{agent} must not be repointed"
            );
        }
    }

    #[test]
    fn empty_model_omits_model_hint() {
        let env = foreign_gateway_env("codex", &coords(), "");
        let map: std::collections::HashMap<_, _> = env.into_iter().collect();
        assert!(map.contains_key("OPENAI_BASE_URL"));
        assert!(map.contains_key("OPENAI_API_KEY"));
        // No model hint when the profile carries no model.
        assert!(!map.contains_key("OPENAI_MODEL"));
    }
}
