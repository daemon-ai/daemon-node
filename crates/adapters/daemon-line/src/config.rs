// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Resolved LINE adapter configuration.
//!
//! Like the Matrix adapter, the *route table* is the config surface: it carries no account secrets
//! (accounts + their credential blobs come from `ProfileSpec.bound_accounts` + `CredentialStore`,
//! enumerated at bring-up via `AccountProvisioning`). A route only narrows *which* conversations the
//! adapter engages and *how it classifies addressing* (mention-gating). Isolation is pinned to
//! `PerThread` so the ingest gate's derived session id matches the host's routing resolution.
//!
//! The one LINE-specific config knob is the inbound webhook: LINE is webhook-push, so the platform
//! POSTs events to a public HTTP endpoint. [`webhook_bind`](LineConfig::webhook_bind) makes the
//! listener *adapter-owned* (bind here + serve `{webhook_path}/<handle>`); when unset, inbound is
//! left for an external ingress to mount [`crate::inbound::webhook_router`] (Phase 2).

use daemon_ingest::IngestPolicy;
use daemon_protocol::IsolationPolicy;
use serde::{Deserialize, Serialize};

/// The resolved `[line]` config the host hands to [`crate::serve`]. The adapter owns the *shape*; the
/// binary's `NodeConfig` deserializes it (figment) as the `[line]` table / `DAEMON_LINE__*` env.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LineConfig {
    /// Whether the adapter is spawned at all. `false` (default) leaves LINE off.
    #[serde(with = "daemon_common::flex_bool")]
    pub enabled: bool,
    /// The address the adapter-owned webhook listener binds (e.g. `127.0.0.1:8687`). `None` leaves
    /// the listener unbound: inbound is then wired by an external ingress that mounts
    /// [`crate::inbound::webhook_router`] (Phase 2). Outbound push works either way.
    pub webhook_bind: Option<String>,
    /// The base path each account's webhook route is mounted under (the full path a LINE channel is
    /// configured to POST to is `{webhook_path}/<handle>`). Default `/line/webhook`.
    pub webhook_path: String,
    /// The route table — which conversations to engage + how addressing is classified. Empty engages
    /// every conversation of every account with mention-gating on. TOML key `route`
    /// (`[[line.route]]`).
    #[serde(rename = "route")]
    pub routes: Vec<LineRoute>,
}

impl Default for LineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            webhook_bind: None,
            webhook_path: "/line/webhook".to_string(),
            routes: Vec::new(),
        }
    }
}

impl LineConfig {
    /// The account-level ingest policy. Isolation is pinned to `PerThread` to match the host's
    /// no-binding routing resolution, so the gate and the outbound stream key busy-state on the same
    /// session id.
    pub fn ingest_policy(&self) -> IngestPolicy {
        IngestPolicy {
            isolation: IsolationPolicy::PerThread,
            ..IngestPolicy::default()
        }
    }
}

/// One `[[line.route]]` entry. All matchers are optional; an empty route matches every conversation
/// of every account with mention-gating on.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LineRoute {
    /// Match only this account instance (the bare `<handle>`); `None` matches any account.
    pub account: Option<String>,
    /// A glob over the conversation id (LINE user/group/room id) this route applies to; `None`
    /// matches any conversation.
    pub target_glob: Option<String>,
    /// Restrict to direct (user-source) conversations only.
    pub dm_only: bool,
    /// Whether the agent only *turns* on an explicit `!command` / DM (ambient chatter is surfaced as
    /// context via the gate's ambient policy). LINE has no reliable bot-mention signal in group text,
    /// so gating leans on DM + `!command`. When `false`, every message in a matching conversation is
    /// treated as addressed.
    pub mention_gating: bool,
}

impl Default for LineRoute {
    fn default() -> Self {
        Self {
            account: None,
            target_glob: None,
            dm_only: false,
            mention_gating: true,
        }
    }
}

impl LineRoute {
    /// Whether this route matches `account` (the bare handle) and `target` (LINE conversation id).
    pub fn matches(&self, account: &str, target: &str, is_dm: bool) -> bool {
        if let Some(a) = &self.account {
            if a != account {
                return false;
            }
        }
        if self.dm_only && !is_dm {
            return false;
        }
        if let Some(g) = &self.target_glob {
            if !glob_match(g, target) {
                return false;
            }
        }
        true
    }
}

/// Pick the route governing `(account, target, is_dm)`. With no configured routes, every
/// conversation is engaged with mention-gating on (the default route). With a configured table, only
/// conversations matching some route are engaged; a non-matching one returns `None` (ignored).
pub fn route_for<'a>(
    routes: &'a [LineRoute],
    account: &str,
    target: &str,
    is_dm: bool,
) -> Option<std::borrow::Cow<'a, LineRoute>> {
    if routes.is_empty() {
        return Some(std::borrow::Cow::Owned(LineRoute::default()));
    }
    routes
        .iter()
        .find(|r| r.matches(account, target, is_dm))
        .map(std::borrow::Cow::Borrowed)
}

/// A tiny `*`/`?` glob (no character classes) — enough for prefix matchers without pulling a glob
/// crate. `*` matches any run (including empty), `?` one char.
fn glob_match(pattern: &str, text: &str) -> bool {
    glob_inner(pattern.as_bytes(), text.as_bytes())
}

fn glob_inner(pat: &[u8], text: &[u8]) -> bool {
    match pat.first() {
        None => text.is_empty(),
        Some(b'*') => {
            glob_inner(&pat[1..], text) || (!text.is_empty() && glob_inner(pat, &text[1..]))
        }
        Some(b'?') => !text.is_empty() && glob_inner(&pat[1..], &text[1..]),
        Some(&c) => !text.is_empty() && text[0] == c && glob_inner(&pat[1..], &text[1..]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_basics() {
        assert!(glob_match("C*", "Cabcdef"));
        assert!(!glob_match("C*", "Uabcdef"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("U?c", "Uac"));
        assert!(!glob_match("U?c", "Uc"));
    }

    #[test]
    fn empty_table_engages_all_with_gating() {
        let r = route_for(&[], "acme", "Cgroupid", false).unwrap();
        assert!(r.mention_gating);
    }

    #[test]
    fn configured_table_ignores_unmatched_targets() {
        let routes = vec![LineRoute {
            target_glob: Some("C*".into()),
            mention_gating: false,
            ..Default::default()
        }];
        let r = route_for(&routes, "acme", "Cgroup", false).unwrap();
        assert!(!r.mention_gating);
        assert!(route_for(&routes, "acme", "Uuser", true).is_none());
    }

    #[test]
    fn dm_only_route_matches_user_sources() {
        let routes = vec![LineRoute {
            dm_only: true,
            ..Default::default()
        }];
        assert!(route_for(&routes, "acme", "Uuser", true).is_some());
        assert!(route_for(&routes, "acme", "Cgroup", false).is_none());
    }
}
