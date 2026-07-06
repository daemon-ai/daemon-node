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
) -> ForeignSessionFactory {
    Box::new(move |host| {
        Box::pin(async move {
            let entry = resolve_entry(&agent, &store).await?;
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
                    )) as Arc<dyn AgentSession>)
                }
            }
        })
    })
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
