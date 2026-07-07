// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Per-account credential material: the token blob the `CredentialStore` is the system of record for.
//!
//! Each Discord **account** is a distinct transport instance (`discord/<user_id>`). Unlike Matrix
//! there is no on-disk crypto/state store — Discord auth is a single opaque token (bot or user). The
//! credential subsystem stores that token (plus the [`DiscordMode`] it was minted under) as a JSON
//! blob under the account's `credential_ref`; `serve` reads it back and dials the gateway with it.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::DiscordMode;

/// The credential-store blob for one Discord account: the raw token plus the mode it was minted
/// under. Serialized as JSON under the account's `credential_ref`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredCredential {
    /// The raw Discord token (never `Bot `/`Bearer `-prefixed — `serenity_self` sends it verbatim).
    pub token: String,
    /// The account mode this token was minted under (`bot` | `user`).
    #[serde(default)]
    pub mode: DiscordMode,
}

impl StoredCredential {
    /// Serialize to the opaque credential blob.
    pub fn to_blob(&self) -> Result<String> {
        serde_json::to_string(self).context("serializing discord credential blob")
    }

    /// Parse from the opaque credential blob.
    pub fn from_blob(blob: &str) -> Result<Self> {
        serde_json::from_str(blob).context("parsing discord credential blob")
    }
}

/// The bare Discord user id (`1234`) inside an instance-qualified `discord/1234` transport id.
pub fn bare_account(transport: &daemon_protocol::TransportId) -> &str {
    transport
        .as_str()
        .strip_prefix("discord/")
        .unwrap_or_else(|| transport.as_str())
}

/// Normalize a pasted Discord token for `serenity_self`, which sends the token verbatim and
/// **panics** on a `Bot `/`Bearer ` prefix (`serenity_self::http::client::parse_token`). Operators
/// commonly paste a bot token with its `Bot ` prefix, so strip a leading `Bot `/`Bearer ` (any case)
/// and trim surrounding whitespace before the token ever reaches the SDK.
pub fn sanitize_token(raw: &str) -> String {
    let t = raw.trim();
    for prefix in ["Bot ", "Bearer "] {
        if t.len() >= prefix.len() && t[..prefix.len()].eq_ignore_ascii_case(prefix) {
            return t[prefix.len()..].trim().to_string();
        }
    }
    t.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_protocol::TransportId;

    #[test]
    fn bare_strips_family_prefix() {
        let t = TransportId::new("discord/1234".to_string());
        assert_eq!(bare_account(&t), "1234");
    }

    #[test]
    fn credential_blob_roundtrips() {
        let c = StoredCredential {
            token: "abc.def.ghi".to_string(),
            mode: DiscordMode::User,
        };
        let blob = c.to_blob().unwrap();
        let back = StoredCredential::from_blob(&blob).unwrap();
        assert_eq!(c, back);
        assert_eq!(back.mode, DiscordMode::User);
    }

    #[test]
    fn sanitize_strips_bot_and_bearer_prefixes() {
        assert_eq!(sanitize_token("  abc.def  "), "abc.def");
        assert_eq!(sanitize_token("Bot abc.def"), "abc.def");
        assert_eq!(sanitize_token("bot ABC"), "ABC");
        assert_eq!(sanitize_token("Bearer xyz"), "xyz");
        // A raw user token (no prefix) is left intact.
        assert_eq!(sanitize_token("mfa.longusertoken"), "mfa.longusertoken");
    }
}
