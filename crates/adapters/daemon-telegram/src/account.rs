// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Per-account credential model + on-disk session store keying.
//!
//! Each Telegram **account** is a distinct transport instance (`telegram/<id>`) owning its own
//! grammers MTProto session. Two persistence layers, mirroring the Matrix adapter:
//!
//! - The **grammers session** (authorization key + peer cache) lives in an on-disk SQLite store at
//!   `<store_root>/<credential_ref>/session.sqlite` (grammers' `SqliteSession`). It is keyed by the
//!   **credential ref** — stable and known at both `login` and `serve` time — so the exact session
//!   the login minted is the one bring-up re-opens (device-stability, matching Matrix §6.3).
//! - The **credential-store blob** ([`StoredSession`]) records the account *mode* (user vs bot) and,
//!   for a bot, its token — the small non-key metadata bring-up needs to re-establish the login when
//!   the SQLite store alone is not authoritative. The authorization key itself is never copied into
//!   the wire blob; it stays in the on-disk store.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use daemon_protocol::TransportId;

/// Whether an account logs in as a Telegram **user** (phone + code, optionally 2FA) or a **bot**
/// (a BotFather token). Selected per account at login time; recorded in the [`StoredSession`] so
/// bring-up drives the right restore path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountMode {
    /// A real user account (grammers `request_login_code` / `sign_in` / `check_password`).
    User,
    /// A bot account (grammers `bot_sign_in` with a BotFather token).
    Bot,
}

impl AccountMode {
    /// Parse a mode string (`"user"` / `"bot"`), case-insensitively. `None` for anything else.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "user" => Some(Self::User),
            "bot" => Some(Self::Bot),
            _ => None,
        }
    }
}

/// The credential-store blob for one Telegram account: the login mode and, for a bot, the token used
/// to (re-)sign in. Serialized as JSON under the account's credential-ref. This is **not** the
/// grammers authorization key (that is the on-disk SQLite session store).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredSession {
    /// How this account authenticates.
    pub mode: AccountMode,
    /// The bot token (`Bot` accounts only); `None` for user accounts (whose key lives only in the
    /// on-disk session store).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot_token: Option<String>,
    /// The resolved Telegram account id (`get_me().id()`), recorded for the transport instance id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<i64>,
}

impl StoredSession {
    /// A user-account blob (no token; the key lives in the on-disk store).
    pub fn user(account_id: i64) -> Self {
        Self {
            mode: AccountMode::User,
            bot_token: None,
            account_id: Some(account_id),
        }
    }

    /// A bot-account blob carrying the token so bring-up can re-`bot_sign_in`.
    pub fn bot(token: String, account_id: i64) -> Self {
        Self {
            mode: AccountMode::Bot,
            bot_token: Some(token),
            account_id: Some(account_id),
        }
    }

    /// Serialize to the opaque credential blob.
    pub fn to_blob(&self) -> Result<String> {
        serde_json::to_string(self).context("serializing telegram session blob")
    }

    /// Parse from the opaque credential blob.
    pub fn from_blob(blob: &str) -> Result<Self> {
        serde_json::from_str(blob).context("parsing telegram session blob")
    }
}

/// The bare account id (`<id>`) inside an instance-qualified `telegram/<id>` transport id.
pub fn bare_account(transport: &TransportId) -> &str {
    transport
        .as_str()
        .strip_prefix("telegram/")
        .unwrap_or_else(|| transport.as_str())
}

/// A filesystem-safe directory name for a credential ref / account handle (`telegram/alpha/a` ->
/// `telegram_alpha_a`).
pub fn store_dir_name(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The per-account SQLite session path under `store_root`, keyed by `credential_ref` so `login` and
/// `serve` always open the *same* on-disk session (auth-key stability).
pub fn account_session_path(
    store_root: &std::path::Path,
    credential_ref: &str,
) -> std::path::PathBuf {
    store_root
        .join(store_dir_name(credential_ref))
        .join("session.sqlite")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_strips_family_prefix() {
        let t = TransportId::new("telegram/123456".to_string());
        assert_eq!(bare_account(&t), "123456");
    }

    #[test]
    fn store_dir_name_is_fs_safe() {
        assert_eq!(store_dir_name("telegram/alpha/a"), "telegram_alpha_a");
        assert_eq!(store_dir_name("alpha-1.2"), "alpha-1.2");
    }

    #[test]
    fn mode_parse() {
        assert_eq!(AccountMode::parse("user"), Some(AccountMode::User));
        assert_eq!(AccountMode::parse(" BOT "), Some(AccountMode::Bot));
        assert_eq!(AccountMode::parse("nope"), None);
    }

    #[test]
    fn bot_blob_roundtrips_with_token() {
        let s = StoredSession::bot("123:abc".to_string(), 42);
        let blob = s.to_blob().unwrap();
        let back = StoredSession::from_blob(&blob).unwrap();
        assert_eq!(back, s);
        assert_eq!(back.mode, AccountMode::Bot);
        assert_eq!(back.bot_token.as_deref(), Some("123:abc"));
    }

    #[test]
    fn user_blob_omits_token() {
        let s = StoredSession::user(7);
        let blob = s.to_blob().unwrap();
        assert!(
            !blob.contains("bot_token"),
            "user blob carries no token: {blob}"
        );
        let back = StoredSession::from_blob(&blob).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn session_path_is_keyed_by_credential_ref() {
        let p = account_session_path(std::path::Path::new("/data/tg"), "telegram/@bot");
        assert!(
            p.ends_with("telegram__bot/session.sqlite"),
            "got {}",
            p.display()
        );
    }
}
