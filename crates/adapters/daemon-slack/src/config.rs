// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Resolved Slack adapter configuration.
//!
//! Like the Matrix adapter's config, the *route table* is the config surface of the routing registry:
//! it carries no account secrets (accounts + their credential blobs come from
//! `ProfileSpec.bound_accounts` + the `CredentialStore`, enumerated at bring-up via
//! `AccountProvisioning`). A route only narrows *which* channels the adapter engages and *how it
//! classifies addressing* (mention-gating). Isolation is pinned to `PerThread` so the ingest gate and
//! the outbound delivery stream key busy-state on the same session id.

use daemon_ingest::IngestPolicy;
use daemon_protocol::IsolationPolicy;
use serde::{Deserialize, Serialize};

/// The resolved `[slack]` config the host hands to [`crate::serve`]. The adapter owns the *shape*;
/// the binary's `NodeConfig` deserializes it (figment) as the `[slack]` table / `DAEMON_SLACK__*`
/// env.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SlackConfig {
    /// Whether the adapter is spawned at all. `false` (default) leaves Slack off.
    #[serde(with = "daemon_common::flex_bool")]
    pub enabled: bool,
    /// The Slack **app-level token** (`xapp-…`) used to open Socket Mode connections for bot/app
    /// accounts (Phase 2 wiring: env `DAEMON_SLACK__APP_TOKEN`). An app-level token is a property of
    /// the Slack *app*, not of any single workspace install, so it is node config rather than a
    /// per-account credential. `None` disables Socket Mode inbound (bot accounts still send/manage;
    /// user "stealth" accounts never use Socket Mode).
    #[serde(default)]
    pub app_token: Option<String>,
    /// The route table — which channels to engage + how addressing is classified. Empty engages
    /// every channel of every account with mention-gating on. TOML key `route` (`[[slack.route]]`).
    #[serde(rename = "route")]
    pub routes: Vec<SlackRoute>,
}

impl SlackConfig {
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

/// One `[[slack.route]]` entry. All matchers are optional; an empty route matches every channel of
/// every account with mention-gating on.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SlackRoute {
    /// Match only this account instance (the bare team/label after `slack/`); `None` matches any.
    pub account: Option<String>,
    /// A glob over the channel (id or name) this route applies to; `None` matches any channel.
    pub channel_glob: Option<String>,
    /// Restrict to direct-message (IM) conversations only.
    pub dm_only: bool,
    /// Whether the agent only *turns* on an explicit mention / DM / `!command` (ambient chatter is
    /// surfaced as context via the gate's ambient policy). When `false`, every message in a matching
    /// channel is treated as addressed.
    pub mention_gating: bool,
}

impl Default for SlackRoute {
    fn default() -> Self {
        Self {
            account: None,
            channel_glob: None,
            dm_only: false,
            mention_gating: true,
        }
    }
}

impl SlackRoute {
    /// Whether this route matches `account` (the bare label) and `channel` (id or name).
    pub fn matches(&self, account: &str, channel: &str, is_dm: bool) -> bool {
        if let Some(a) = &self.account {
            if a != account {
                return false;
            }
        }
        if self.dm_only && !is_dm {
            return false;
        }
        if let Some(g) = &self.channel_glob {
            if !glob_match(g, channel) {
                return false;
            }
        }
        true
    }
}

/// Pick the route governing `(account, channel, is_dm)`. With no configured routes, every channel is
/// engaged with mention-gating on (the default route). With a configured table, only channels
/// matching some route are engaged; a non-matching channel returns `None` (the adapter ignores it).
pub fn route_for<'a>(
    routes: &'a [SlackRoute],
    account: &str,
    channel: &str,
    is_dm: bool,
) -> Option<std::borrow::Cow<'a, SlackRoute>> {
    if routes.is_empty() {
        return Some(std::borrow::Cow::Owned(SlackRoute::default()));
    }
    routes
        .iter()
        .find(|r| r.matches(account, channel, is_dm))
        .map(std::borrow::Cow::Borrowed)
}

/// A tiny `*`/`?` glob (no character classes) — enough for `#secops*` style channel matchers without
/// pulling a glob crate into the adapter. `*` matches any run (including empty), `?` one char.
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
        assert!(glob_match("secops*", "secops"));
        assert!(glob_match("secops*", "secops-prod"));
        assert!(!glob_match("secops*", "general"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
    }

    #[test]
    fn empty_table_engages_all_with_gating() {
        let r = route_for(&[], "T123", "C123", false).unwrap();
        assert!(r.mention_gating);
    }

    #[test]
    fn configured_table_ignores_unmatched_channels() {
        let routes = vec![SlackRoute {
            channel_glob: Some("secops*".into()),
            mention_gating: false,
            ..Default::default()
        }];
        let r = route_for(&routes, "T1", "secops-prod", false).unwrap();
        assert!(!r.mention_gating);
        assert!(route_for(&routes, "T1", "general", false).is_none());
    }

    #[test]
    fn dm_only_route_matches_dms() {
        let routes = vec![SlackRoute {
            dm_only: true,
            ..Default::default()
        }];
        assert!(route_for(&routes, "T1", "D1", true).is_some());
        assert!(route_for(&routes, "T1", "C1", false).is_none());
    }
}
