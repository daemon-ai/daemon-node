// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Resolved Discord adapter configuration.
//!
//! Structurally the same route-table surface as `daemon-matrix`: the config carries no account
//! secrets (accounts + their token blobs come from `ProfileSpec.bound_accounts` + `CredentialStore`,
//! enumerated at bring-up via `AccountProvisioning`). A route only narrows *which* channels the
//! adapter engages and *how it classifies addressing* (mention-gating). The account [`DiscordMode`]
//! (`user` | `bot`) selects the interactive-auth flow kind and the label — the token itself is a
//! single form field either way.

use daemon_ingest::IngestPolicy;
use daemon_protocol::IsolationPolicy;
use serde::{Deserialize, Serialize};

/// Whether a Discord account authenticates as a **bot** application token or a **user** account
/// token. `serenity_self` sends whichever token verbatim (it prepends no `Bot ` prefix); the mode
/// only picks the interactive-auth flow kind + human label. See the crate header for the user-token
/// (self-bot) Terms-of-Service / account-ban risk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiscordMode {
    /// A bot application token (the default; `AuthFlowKind::BotToken`).
    #[default]
    Bot,
    /// A user account token — self-bot mode (`AuthFlowKind::UserToken`); ToS/ban risk.
    User,
}

impl DiscordMode {
    /// The stable string tag (`"bot"` / `"user"`) used in blobs and labels.
    pub fn as_str(self) -> &'static str {
        match self {
            DiscordMode::Bot => "bot",
            DiscordMode::User => "user",
        }
    }

    /// Parse a mode tag; unknown/empty defaults to [`DiscordMode::Bot`].
    pub fn parse(tag: &str) -> Self {
        match tag.trim().to_ascii_lowercase().as_str() {
            "user" => DiscordMode::User,
            _ => DiscordMode::Bot,
        }
    }
}

/// The resolved `[discord]` config the host hands to [`crate::serve`]. The adapter owns the *shape*;
/// the binary's `NodeConfig` deserializes it (figment) as the `[discord]` table / `DAEMON_DISCORD__*`
/// env.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DiscordConfig {
    /// Whether the adapter is spawned at all. `false` (default) leaves Discord off.
    #[serde(with = "daemon_common::flex_bool")]
    pub enabled: bool,
    /// The account token mode (`bot` | `user`) — selects the auth flow kind + label. Default `bot`.
    pub mode: DiscordMode,
    /// The route table — which channels to engage + how addressing is classified. Empty engages
    /// every channel of every account with mention-gating on. TOML key `route` (`[[discord.route]]`).
    #[serde(rename = "route")]
    pub routes: Vec<DiscordRoute>,
}

impl Default for DiscordConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: DiscordMode::Bot,
            routes: Vec::new(),
        }
    }
}

impl DiscordConfig {
    /// The account-level ingest policy. Isolation is pinned to `PerThread` to match the host's
    /// no-binding routing resolution, so the gate and the outbound stream key busy-state on the same
    /// session id (mirrors `daemon-matrix`). A Discord channel id is the session/route key, so with
    /// `thread = None` this collapses to one session per `(account, channel)`.
    pub fn ingest_policy(&self) -> IngestPolicy {
        IngestPolicy {
            isolation: IsolationPolicy::PerThread,
            ..IngestPolicy::default()
        }
    }
}

/// One `[[discord.route]]` entry. All matchers are optional; an empty route matches every channel of
/// every account with mention-gating on.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DiscordRoute {
    /// Match only this account instance (the bare Discord user id, e.g. `discord/1234`'s `1234`);
    /// `None` matches any account.
    pub account: Option<String>,
    /// A glob over the channel id this route applies to; `None` matches any channel.
    pub channel_glob: Option<String>,
    /// Restrict to direct-message channels only.
    pub dm_only: bool,
    /// Whether the agent only *turns* on an explicit mention / DM / `!command` (ambient chatter is
    /// surfaced as context via the gate's ambient policy). When `false`, every message in a matching
    /// channel is treated as addressed.
    pub mention_gating: bool,
}

impl Default for DiscordRoute {
    fn default() -> Self {
        Self {
            account: None,
            channel_glob: None,
            dm_only: false,
            mention_gating: true,
        }
    }
}

impl DiscordRoute {
    /// Whether this route matches `account` (bare user id) and `channel` (id) at DM-ness `is_dm`.
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
    routes: &'a [DiscordRoute],
    account: &str,
    channel: &str,
    is_dm: bool,
) -> Option<std::borrow::Cow<'a, DiscordRoute>> {
    if routes.is_empty() {
        return Some(std::borrow::Cow::Owned(DiscordRoute::default()));
    }
    routes
        .iter()
        .find(|r| r.matches(account, channel, is_dm))
        .map(std::borrow::Cow::Borrowed)
}

/// A tiny `*`/`?` glob (no character classes) — enough for channel-id prefixes without pulling a glob
/// crate into the adapter. `*` matches any run (including empty), `?` one char.
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
    fn mode_parse_roundtrips() {
        assert_eq!(DiscordMode::parse("user"), DiscordMode::User);
        assert_eq!(DiscordMode::parse("BOT"), DiscordMode::Bot);
        assert_eq!(DiscordMode::parse("nonsense"), DiscordMode::Bot);
        assert_eq!(DiscordMode::User.as_str(), "user");
        assert_eq!(DiscordMode::Bot.as_str(), "bot");
    }

    #[test]
    fn empty_table_engages_all_with_gating() {
        let r = route_for(&[], "1234", "999", false).unwrap();
        assert!(r.mention_gating);
    }

    #[test]
    fn configured_table_ignores_unmatched_channels() {
        let routes = vec![DiscordRoute {
            channel_glob: Some("55*".into()),
            mention_gating: false,
            ..Default::default()
        }];
        let r = route_for(&routes, "1234", "5501", false).unwrap();
        assert!(!r.mention_gating);
        assert!(route_for(&routes, "1234", "6001", false).is_none());
    }

    #[test]
    fn dm_only_route_matches_dms() {
        let routes = vec![DiscordRoute {
            dm_only: true,
            ..Default::default()
        }];
        assert!(route_for(&routes, "1234", "77", true).is_some());
        assert!(route_for(&routes, "1234", "77", false).is_none());
    }

    #[test]
    fn glob_basics() {
        assert!(glob_match("55*", "5501"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("5?0", "550"));
        assert!(!glob_match("5?0", "50"));
    }
}
