// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The exhaustive tail of the render chain: the foreign-agent catalog, provider/tool listings,
//! config dump, and the generic `{:?}` fallback for variants without a first-class CLI rendering
//! (e.g. the filesystem surface). Because this arm is total, the compiler still proves every
//! `ApiResponse` variant is handled across the render chain.

use daemon_api::{AgentAuthState, AgentVerification, ApiResponse};

pub(super) fn render_rest(resp: ApiResponse) {
    match resp {
        ApiResponse::AgentCatalog(entries) => {
            println!("foreign agents: {}", entries.len());
            for e in entries {
                // The node derives the verification verdict once (from installed/protocol/version)
                // and ships it on the wire; the client just projects it to a display string.
                let status = match e.verification {
                    AgentVerification::Verified => "verified",
                    AgentVerification::Unverified => "unverified",
                    AgentVerification::NotInstalled => "not-installed",
                };
                // The auth verdict + runnable method ids (wire v47), verbatim from the node —
                // scripted callers (the e2e auth journey) parse this line, keep it stable.
                let auth = match e.auth.as_ref().map(|a| a.state).unwrap_or_default() {
                    AgentAuthState::NotRequired => "not-required",
                    AgentAuthState::Unknown => "unknown",
                    AgentAuthState::Required => "required",
                    AgentAuthState::Authenticated => "authenticated",
                    AgentAuthState::Expired => "expired",
                };
                let methods = e
                    .auth
                    .as_ref()
                    .map(|a| {
                        a.methods
                            .iter()
                            .map(|m| m.id.as_str())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .unwrap_or_default();
                println!(
                    "  - {} [{:?}/{:?}] {} version={:?} auth={} methods=[{}]",
                    e.name, e.source, e.protocol, status, e.version, auth, methods
                );
            }
        }
        ApiResponse::Providers(providers) => {
            for p in providers {
                println!("  - {} available={}", p.name, p.available);
            }
        }
        ApiResponse::Tools(tools) => {
            for t in tools {
                println!("  - {}", t.name);
            }
        }
        ApiResponse::Config(c) => println!("config ({}):\n{}", c.format, c.body),
        // Filesystem-surface responses (daemon-fs-surface-spec.md) and any other variant: the CLI
        // has no first-class fs command yet, so render the debug form generically.
        other => println!("{other:?}"),
    }
}
