// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `wechatbot` iLink types <-> daemon identity/blob projection.
//!
//! The credential subsystem is the system of record for the login material (spec §6.2). WeChat's
//! iLink session is a small opaque bundle (`token` + poll `base_url` + the bot's own account/user
//! ids); [`StoredSession`] is the on-disk shape we persist under the account's `credential_ref` and
//! restore at bring-up. The `wechatbot::types::Credentials` SDK struct never leaves this crate — we
//! convert it to [`StoredSession`] at the auth boundary and keep our own snake_case blob so the wire
//! contract is ours, not the SDK's (which serializes camelCase and carries a volatile `saved_at`).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use daemon_protocol::TransportId;

use crate::FAMILY;

/// The credential-store blob for one WeChat account: the iLink bot token, the poll base URL the
/// session was minted against, and the bot's own account/user ids. Serialized as JSON under the
/// account's credential-ref. This is the whole session — WeChat iLink has no separate on-disk crypto
/// store (unlike Matrix), so restoring this blob is sufficient to resume the long-poll.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredSession {
    /// The iLink bot token (the `Authorization: Bearer` credential for every API call).
    pub token: String,
    /// The poll base URL the session resolved to (may be an IDC-redirected host, not the QR host).
    pub base_url: String,
    /// The bot's own iLink account id (`ilink_bot_id`).
    pub account_id: String,
    /// The bot's own iLink user id (`ilink_user_id`) — the account identity in the transport id.
    pub user_id: String,
}

impl StoredSession {
    /// Serialize to the opaque credential blob.
    pub fn to_blob(&self) -> Result<String> {
        serde_json::to_string(self).context("serializing wechat session blob")
    }

    /// Parse from the opaque credential blob.
    pub fn from_blob(blob: &str) -> Result<Self> {
        serde_json::from_str(blob).context("parsing wechat session blob")
    }

    /// Adopt the SDK's `Credentials` (produced by a completed QR login) into our stored shape.
    pub fn from_credentials(creds: &wechatbot::types::Credentials) -> Self {
        Self {
            token: creds.token.clone(),
            base_url: creds.base_url.clone(),
            account_id: creds.account_id.clone(),
            user_id: creds.user_id.clone(),
        }
    }

    /// The instance-qualified transport id (`wechat/<user_id>`) this session resolves to.
    pub fn transport_instance(&self) -> TransportId {
        transport_for(&self.user_id)
    }
}

/// The instance-qualified transport id (`wechat/<user_id>`) for the bot account `user_id`.
pub fn transport_for(user_id: &str) -> TransportId {
    TransportId::new(format!("{FAMILY}/{user_id}"))
}

/// The bare account user id (`<user_id>`) inside an instance-qualified `wechat/...` transport id.
pub fn bare_account(transport: &TransportId) -> &str {
    transport
        .as_str()
        .strip_prefix(&format!("{FAMILY}/"))
        .unwrap_or_else(|| transport.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_round_trips() {
        let s = StoredSession {
            token: "tok-123".to_string(),
            base_url: "https://idc.example.weixin.qq.com".to_string(),
            account_id: "bot-abc".to_string(),
            user_id: "u-xyz".to_string(),
        };
        let blob = s.to_blob().expect("serialize");
        let back = StoredSession::from_blob(&blob).expect("parse");
        assert_eq!(s, back);
    }

    #[test]
    fn transport_id_is_family_qualified() {
        let t = transport_for("u-xyz");
        assert_eq!(t.as_str(), "wechat/u-xyz");
    }

    #[test]
    fn bare_strips_family_prefix() {
        let t = transport_for("u-xyz");
        assert_eq!(bare_account(&t), "u-xyz");
        // A bare (unqualified) id is returned untouched.
        let raw = TransportId::new("u-xyz");
        assert_eq!(bare_account(&raw), "u-xyz");
    }

    #[test]
    fn from_credentials_drops_volatile_saved_at() {
        let creds = wechatbot::types::Credentials {
            token: "t".to_string(),
            base_url: "https://b".to_string(),
            account_id: "a".to_string(),
            user_id: "u".to_string(),
            saved_at: Some("1700000000Z".to_string()),
        };
        let s = StoredSession::from_credentials(&creds);
        assert_eq!(s.user_id, "u");
        assert_eq!(s.transport_instance().as_str(), "wechat/u");
    }
}
