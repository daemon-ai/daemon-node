// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Foreign-engine (ACP) resolution for the LIVE interactive session seam: a profile whose
//! `engine = Acp { agent }` resolves its catalog entry BY NAME at spawn time and materializes a
//! [`daemon_acp::AcpSession`] behind the host's transport-agnostic `AgentSession` seam.
//!
//! The security invariant lives here: profiles carry only the catalog NAME. The launch recipe is
//! read from the node's own sources — the durable manual registrations (`acp_register`, operator
//! authz) and the curated builtin table — never from the profile, so a `ProfileCreate` can never
//! smuggle an arbitrary binary spawn. Installed-ness is re-checked at spawn (it can change after
//! profile validation); a vanished or uninstalled agent fails the session open with a clear
//! [`ApiError`] instead of a dead actor.

use std::sync::Arc;

use daemon_api::{from_cbor, AcpAgentEntry, ApiError};
use daemon_host::{AcpDiscovery, ForeignSessionFactory};
use daemon_store::SessionStore;

/// Build the deferred foreign-session factory for a profile bound to ACP agent `agent`: the
/// factory resolves the recipe (durable manual registration first, curated builtin fallback —
/// exactly the `acp_catalog` precedence), re-checks installed-ness, and connects the ACP adapter
/// session with the host's parking handler answering its permission callbacks.
pub(crate) fn acp_session_factory(
    agent: String,
    store: Arc<dyn SessionStore>,
) -> ForeignSessionFactory {
    Box::new(move |host| {
        Box::pin(async move {
            let entry = resolve_entry(&agent, &store).await?;
            // Spawn-time re-check: validation at create/update proved installed-ness THEN; the
            // binary can have been removed since. `recipe_installed` is a cheap PATH/file probe.
            if !daemon_acp::recipe_installed(&entry.recipe) {
                return Err(ApiError::Other(format!(
                    "ACP agent `{agent}` is not installed (its catalog recipe no longer resolves \
                     to a runnable program)"
                )));
            }
            let launch = daemon_acp::launch_from_recipe(&entry.recipe).ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "ACP agent `{agent}` has an endpoint-only recipe; endpoint agents are not \
                     spawnable as session engines yet"
                ))
            })?;
            Ok(daemon_acp::AcpSession::connect(launch, host))
        })
    })
}

/// Resolve `agent` to its catalog entry: durable manual registrations take precedence over the
/// curated builtin table (mirrors the `acp_catalog` merge order — Manual wins over Builtin).
async fn resolve_entry(
    agent: &str,
    store: &Arc<dyn SessionStore>,
) -> Result<AcpAgentEntry, ApiError> {
    for stored in store.acp_list().await {
        if stored.name != agent {
            continue;
        }
        if let Ok(entry) = from_cbor::<AcpAgentEntry>(&stored.entry) {
            return Ok(entry);
        }
    }
    daemon_acp::AcpDiscoverer::new()
        .builtin(agent)
        .ok_or_else(|| {
            ApiError::Other(format!(
                "profile engine references unknown ACP agent `{agent}` — register it via \
                 acp_register or run AcpDiscover first"
            ))
        })
}
