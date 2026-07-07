// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Resolved Telegram adapter configuration.
//!
//! Mirrors `daemon-matrix`'s config shape: the *route table* is the config surface of the routing
//! registry (it carries no account secrets — accounts + their session blobs come from
//! `ProfileSpec.bound_accounts` + the credential store, enumerated at bring-up via
//! `AccountProvisioning`). A route only narrows *which* chats the adapter engages and *how it
//! classifies addressing* (mention-gating). The Telegram app credentials (`api_id` + `api_hash`,
//! obtained from <https://my.telegram.org>) are node-wide and required for any account to connect.

use daemon_ingest::IngestPolicy;
use daemon_protocol::IsolationPolicy;
use serde::{Deserialize, Serialize};

/// The resolved `[telegram]` config the host hands to [`crate::client::serve`]. The adapter owns the
/// *shape*; the binary's `NodeConfig` deserializes it (figment) as the `[telegram]` table /
/// `DAEMON_TELEGRAM__*` env and resolves `store_root` against the node `data_dir`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TelegramConfig {
    /// Whether the adapter is spawned at all. `false` (default) leaves Telegram off.
    #[serde(with = "daemon_common::flex_bool")]
    pub enabled: bool,
    /// The Telegram application id (from <https://my.telegram.org>). Required to connect; anyone can
    /// log in (user or bot) with the developer's `api_id`/`api_hash` — end users don't provide their
    /// own. Node-wide (shared by every account).
    pub api_id: i32,
    /// The Telegram application hash paired with [`Self::api_id`].
    pub api_hash: String,
    /// The per-account session store root (the binary resolves it against the node `data_dir`). Each
    /// account gets its own `<store_root>/<credential_ref>/session.sqlite` holding the grammers
    /// MTProto session (authorization key + peer cache).
    pub store_root: std::path::PathBuf,
    /// The route table — which chats to engage + how addressing is classified. Empty engages every
    /// chat of every account with mention-gating on. TOML key `route` (`[[telegram.route]]`).
    #[serde(rename = "route")]
    pub routes: Vec<TelegramRoute>,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_id: 0,
            api_hash: String::new(),
            store_root: std::path::PathBuf::from("telegram"),
            routes: Vec::new(),
        }
    }
}

impl TelegramConfig {
    /// The account-level ingest policy. Isolation is pinned to `PerThread` to match the host's
    /// no-binding routing resolution, so the gate and the outbound stream key busy-state on the same
    /// session id (same rationale as the Matrix adapter).
    pub fn ingest_policy(&self) -> IngestPolicy {
        IngestPolicy {
            isolation: IsolationPolicy::PerThread,
            ..IngestPolicy::default()
        }
    }
}

/// One `[[telegram.route]]` entry. All matchers are optional; an empty route matches every chat of
/// every account with mention-gating on.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TelegramRoute {
    /// Match only this account instance (the bare account id, e.g. `123456789`); `None` matches any.
    pub account: Option<String>,
    /// A glob over the chat id / `@username` this route applies to; `None` matches any chat.
    pub chat_glob: Option<String>,
    /// Restrict to direct-message (private) chats only.
    pub dm_only: bool,
    /// Whether the agent only *turns* on an explicit mention / DM / `!command` (ambient chatter is
    /// surfaced as context via the gate's ambient policy). When `false`, every message in a matching
    /// chat is treated as addressed.
    pub mention_gating: bool,
}

impl Default for TelegramRoute {
    fn default() -> Self {
        Self {
            account: None,
            chat_glob: None,
            dm_only: false,
            mention_gating: true,
        }
    }
}

impl TelegramRoute {
    /// Whether this route matches `account` (bare id) and `chat` (id or `@username`).
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
/// engaged with mention-gating on (the default route). With a configured table, only chats matching
/// some route are engaged; a non-matching chat returns `None` (the adapter ignores it).
pub fn route_for<'a>(
    routes: &'a [TelegramRoute],
    account: &str,
    chat: &str,
    is_dm: bool,
) -> Option<std::borrow::Cow<'a, TelegramRoute>> {
    if routes.is_empty() {
        return Some(std::borrow::Cow::Owned(TelegramRoute::default()));
    }
    routes
        .iter()
        .find(|r| r.matches(account, chat, is_dm))
        .map(std::borrow::Cow::Borrowed)
}

/// A tiny `*`/`?` glob (no character classes) — enough for `@team*` style chat matchers without
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
        assert!(glob_match("@team*", "@team"));
        assert!(glob_match("@team*", "@team-prod"));
        assert!(!glob_match("@team*", "@general"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("1?3", "123"));
        assert!(!glob_match("1?3", "13"));
    }

    #[test]
    fn empty_table_engages_all_with_gating() {
        let r = route_for(&[], "123", "456", false).unwrap();
        assert!(r.mention_gating);
    }

    #[test]
    fn configured_table_ignores_unmatched_chats() {
        let routes = vec![TelegramRoute {
            chat_glob: Some("@secops*".into()),
            mention_gating: false,
            ..Default::default()
        }];
        let r = route_for(&routes, "123", "@secops-prod", false).unwrap();
        assert!(!r.mention_gating);
        assert!(route_for(&routes, "123", "@general", false).is_none());
    }

    #[test]
    fn dm_only_route_matches_dms() {
        let routes = vec![TelegramRoute {
            dm_only: true,
            ..Default::default()
        }];
        assert!(route_for(&routes, "1", "42", true).is_some());
        assert!(route_for(&routes, "1", "42", false).is_none());
    }
}
