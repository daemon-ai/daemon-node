// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Resolved WhatsApp adapter configuration.
//!
//! Like the Matrix adapter, the route table carries no account secrets — accounts and their session
//! blobs come from `ProfileSpec.bound_accounts` + `CredentialStore`, enumerated at bring-up via
//! `AccountProvisioning`. A route only narrows *which* chats the adapter engages and *how it
//! classifies addressing* (mention/command gating). Isolation is pinned to `PerThread` so the ingest
//! gate and the outbound stream key busy-state on the same session id the host resolves.

use daemon_ingest::IngestPolicy;
use daemon_protocol::IsolationPolicy;
use serde::{Deserialize, Serialize};

/// The resolved `[whatsapp]` config the host hands to [`crate::serve`]. The binary's `NodeConfig`
/// deserializes it (figment) as the `[whatsapp]` table / `DAEMON_WHATSAPP__*` env.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WhatsappConfig {
    /// Whether the adapter is spawned at all. `false` (default) leaves WhatsApp off.
    #[serde(with = "daemon_common::flex_bool")]
    pub enabled: bool,
    /// The route table — which chats to engage + how addressing is classified. Empty engages every
    /// chat of every account with command/mention gating on. TOML key `route` (`[[whatsapp.route]]`).
    #[serde(rename = "route")]
    pub routes: Vec<WhatsappRoute>,
}

impl WhatsappConfig {
    /// The account-level ingest policy. Isolation is pinned to `PerThread` to match the host's
    /// no-binding routing resolution (see the Matrix adapter for the rationale).
    pub fn ingest_policy(&self) -> IngestPolicy {
        IngestPolicy {
            isolation: IsolationPolicy::PerThread,
            ..IngestPolicy::default()
        }
    }
}

/// One `[[whatsapp.route]]` entry. All matchers are optional; an empty route matches every chat of
/// every account with command/mention gating on.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WhatsappRoute {
    /// Match only this account instance (the bare account handle); `None` matches any account.
    pub account: Option<String>,
    /// A glob over the chat id (JID / phone) this route applies to; `None` matches any chat.
    pub chat_glob: Option<String>,
    /// Restrict to direct-message chats only (non-group).
    pub dm_only: bool,
    /// Whether the agent only *turns* on an explicit `!command` (or a DM); ambient group chatter is
    /// surfaced as context via the gate's ambient policy. When `false`, every message in a matching
    /// chat is treated as addressed.
    pub mention_gating: bool,
}

impl Default for WhatsappRoute {
    fn default() -> Self {
        Self {
            account: None,
            chat_glob: None,
            dm_only: false,
            mention_gating: true,
        }
    }
}

impl WhatsappRoute {
    /// Whether this route matches `account` (the bare handle) and `chat` (JID/phone).
    pub fn matches(&self, account: &str, chat: &str, is_dm: bool) -> bool {
        if let Some(a) = &self.account {
            if a != account {
                return false;
            }
        }
        if self.dm_only && !is_dm {
            return false;
        }
        if let Some(g) = &self.chat_glob {
            if !glob_match(g, chat) {
                return false;
            }
        }
        true
    }
}

/// Pick the route governing `(account, chat, is_dm)`. With no configured routes, every chat is
/// engaged with gating on (the default route). With a configured table, only chats matching some
/// route are engaged; a non-matching chat returns `None` (the adapter ignores it).
pub fn route_for<'a>(
    routes: &'a [WhatsappRoute],
    account: &str,
    chat: &str,
    is_dm: bool,
) -> Option<std::borrow::Cow<'a, WhatsappRoute>> {
    if routes.is_empty() {
        return Some(std::borrow::Cow::Owned(WhatsappRoute::default()));
    }
    routes
        .iter()
        .find(|r| r.matches(account, chat, is_dm))
        .map(std::borrow::Cow::Borrowed)
}

/// A tiny `*`/`?` glob (no character classes) — enough for chat-id matchers without a glob crate.
/// `*` matches any run (including empty), `?` one char.
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
        assert!(glob_match("*@g.us", "123-456@g.us"));
        assert!(!glob_match("*@g.us", "123@s.whatsapp.net"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("1?3", "123"));
        assert!(!glob_match("1?3", "13"));
    }

    #[test]
    fn empty_table_engages_all_with_gating() {
        let r = route_for(&[], "acct", "123@s.whatsapp.net", true).unwrap();
        assert!(r.mention_gating);
    }

    #[test]
    fn configured_table_ignores_unmatched_chats() {
        let routes = vec![WhatsappRoute {
            chat_glob: Some("*@g.us".into()),
            mention_gating: false,
            ..Default::default()
        }];
        let r = route_for(&routes, "acct", "123@g.us", false).unwrap();
        assert!(!r.mention_gating);
        assert!(route_for(&routes, "acct", "123@s.whatsapp.net", true).is_none());
    }

    #[test]
    fn dm_only_route_matches_dms() {
        let routes = vec![WhatsappRoute {
            dm_only: true,
            ..Default::default()
        }];
        assert!(route_for(&routes, "a", "1@s.whatsapp.net", true).is_some());
        assert!(route_for(&routes, "a", "1@g.us", false).is_none());
    }
}
