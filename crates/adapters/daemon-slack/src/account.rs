// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Per-account credential material: the opaque `CredentialStore` blob for one Slack account.
//!
//! A Slack **account** is a transport instance (`slack/<label>`) in one of two modes:
//!
//! - **bot/app** — a workspace install obtained via OAuth. The stored secret is the bot token
//!   (`xoxb-…`) plus the resolved team/bot identity. Inbound for these accounts rides Socket Mode
//!   (which additionally needs the app-level token — a node-config property, [`crate::SlackConfig`]).
//! - **user** — a "stealth" login using a browser-extracted `xoxc` token + `xoxd` cookie, driven by
//!   the young `slacko` crate. No OAuth, no app install; the user drives the Web API as themselves.
//!
//! The credential subsystem is the system of record for the login material; this module only
//! (de)serializes the blob it stores.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use daemon_protocol::TransportId;

/// The transport family this adapter provisions.
pub const FAMILY: &str = "slack";

/// The opaque `CredentialStore` blob for one Slack account — the two authentication modes, tagged.
/// Serialized as JSON under the account's credential-ref.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum StoredCredential {
    /// A bot/app workspace install (OAuth `oauth.v2.access` result).
    Bot {
        /// The workspace bot token (`xoxb-…`) used for Web API + as the Socket Mode auth token pair
        /// with the node's app-level token.
        bot_token: String,
        /// The resolved team/workspace id (`T…`).
        team_id: String,
        /// The resolved bot user id (`U…`) — used to drop the bot's own posts on inbound (self-loop).
        bot_user_id: String,
    },
    /// A user "stealth" login (`slacko`, xoxc/xoxd).
    User {
        /// The browser-extracted session token (`xoxc-…`).
        xoxc_token: String,
        /// The `d` cookie value (`xoxd-…`).
        xoxd_cookie: String,
    },
}

impl StoredCredential {
    /// Serialize to the opaque credential blob.
    pub fn to_blob(&self) -> Result<String> {
        serde_json::to_string(self).context("serializing slack credential blob")
    }

    /// Parse from the opaque credential blob.
    pub fn from_blob(blob: &str) -> Result<Self> {
        serde_json::from_str(blob).context("parsing slack credential blob")
    }
}

/// The bare account label (`<label>`) inside an instance-qualified `slack/<label>` transport id.
pub fn bare_account(transport: &TransportId) -> &str {
    transport
        .as_str()
        .strip_prefix("slack/")
        .unwrap_or_else(|| transport.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_strips_family_prefix() {
        let t = TransportId::new("slack/T12345".to_string());
        assert_eq!(bare_account(&t), "T12345");
    }

    #[test]
    fn credential_blob_roundtrips_both_modes() {
        let bot = StoredCredential::Bot {
            bot_token: "xoxb-not-a-real-token".into(),
            team_id: "T1".into(),
            bot_user_id: "U1".into(),
        };
        let back = StoredCredential::from_blob(&bot.to_blob().unwrap()).unwrap();
        assert_eq!(bot, back);

        let user = StoredCredential::User {
            xoxc_token: "xoxc-not-a-real-token".into(),
            xoxd_cookie: "xoxd-not-a-real-cookie".into(),
        };
        let back = StoredCredential::from_blob(&user.to_blob().unwrap()).unwrap();
        assert_eq!(user, back);
    }
}
