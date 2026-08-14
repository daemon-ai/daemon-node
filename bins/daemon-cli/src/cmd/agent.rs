// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `agent` subcommand: the foreign-agent catalog surface (`agent_catalog` /
//! `agent_register` / `agent_remove`). `register` covers the FULL wire contract — recipe,
//! ApiKeyEnv auth descriptor, and the rejection classifier the GUI form deliberately omits — so
//! scripted setups drive the same ingress the clients do, with the node's validation intact.

use daemon_api::{
    AgentAuthDescriptor, AgentAuthScheme, AgentEntry, AgentProtocol, AgentRecipe, AgentSource,
    ApiRequest, RejectionClassifier,
};
use daemon_host::ApiClient;

use crate::cli::AgentCmd;
use crate::render::render;

/// Split one `KEY=VALUE` argument (the shared shape of `--env` and `--rejection`).
fn split_pair(raw: &str, flag: &str) -> anyhow::Result<(String, String)> {
    let Some((k, v)) = raw.split_once('=') else {
        anyhow::bail!("{flag} wants KEY=VALUE, got {raw:?}");
    };
    if k.trim().is_empty() {
        anyhow::bail!("{flag} wants a non-empty key in {raw:?}");
    }
    Ok((k.trim().to_string(), v.trim().to_string()))
}

/// Dispatch an `agent` subcommand over the api mirror.
pub(super) async fn run(client: &ApiClient, cmd: AgentCmd) -> anyhow::Result<()> {
    let req = match cmd {
        AgentCmd::Catalog => ApiRequest::AgentCatalog,
        AgentCmd::Register {
            name,
            protocol,
            program,
            args,
            env,
            endpoint,
            auth_var,
            auth_label,
            rejection,
        } => {
            let env = env
                .iter()
                .map(|raw| split_pair(raw, "--env"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let rejection = rejection
                .as_deref()
                .map(|raw| {
                    let (field, code) = split_pair(raw, "--rejection")?;
                    Ok::<_, anyhow::Error>(RejectionClassifier { field, code })
                })
                .transpose()?;
            // A descriptor travels when either auth flag is present; a rejection classifier
            // without a scheme is legal on the wire (scheme stays None, the probe may add one).
            let auth_descriptor = if auth_var.is_some() || rejection.is_some() {
                Some(AgentAuthDescriptor {
                    scheme: match auth_var {
                        Some(var) => AgentAuthScheme::ApiKeyEnv {
                            var,
                            label: auth_label.unwrap_or_else(|| "API key".to_string()),
                        },
                        None => AgentAuthScheme::None,
                    },
                    rejection,
                })
            } else {
                None
            };
            ApiRequest::AgentRegister {
                entry: AgentEntry {
                    name,
                    recipe: AgentRecipe {
                        program,
                        args,
                        env,
                        endpoint,
                    },
                    // The node forces Manual regardless; sent to match the round-trip.
                    source: AgentSource::Manual,
                    protocol: match protocol.as_str() {
                        "stream-json" => AgentProtocol::StreamJson,
                        _ => AgentProtocol::Acp,
                    },
                    // Derived fields stay at their defaults: the node re-probes installed-ness
                    // and recomputes verification/auth — caller-supplied verdicts are ignored.
                    installed: false,
                    version: None,
                    capabilities: Vec::new(),
                    verification: Default::default(),
                    auth_descriptor,
                    auth: None,
                },
            }
        }
        AgentCmd::Remove { name } => ApiRequest::AgentRemove { name },
    };
    render(client.call(req).await?);
    Ok(())
}
