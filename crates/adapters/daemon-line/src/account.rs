// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Per-account material: the credential-store blob (channel access token + channel secret), the live
//! bot client, and the stable account handle.
//!
//! Each LINE **channel** is a distinct transport instance (`line/<handle>`) owning its own bot
//! [`LINE`](line_bot_sdk_rust::client::LINE) client (which holds the channel access token used for
//! push send) and its channel secret (used to verify inbound webhook signatures). The credential
//! subsystem is the system of record for the login material (the two secrets); this crate holds only
//! the live, in-memory client + secret between bring-up and shutdown.
//!
//! LINE is **bot-only** here: there is no mature Rust LINE *user* client, so the sole auth mode is a
//! channel access token + channel secret pair (see [`crate::auth`]). The transport instance handle is
//! operator-chosen (a friendly channel id) or, absent one, a stable non-reversible short hash of the
//! channel secret — so the same channel always resolves to the same `line/<handle>` id.

use anyhow::{Context, Result};
use line_bot_sdk_rust::client::LINE;
use serde::{Deserialize, Serialize};

use daemon_protocol::TransportId;

/// The credential-store blob for one LINE channel: the channel access token (push auth) plus the
/// channel secret (webhook signature verification). Serialized as JSON under the account's
/// credential-ref. Both are long-lived bot secrets — never on the wire, never logged.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredCredential {
    /// The long-lived channel access token the Messaging API authenticates push send with.
    pub channel_access_token: String,
    /// The channel secret the webhook `x-line-signature` HMAC-SHA256 is verified against.
    pub channel_secret: String,
}

impl StoredCredential {
    /// Serialize to the opaque credential blob.
    pub fn to_blob(&self) -> Result<String> {
        serde_json::to_string(self).context("serializing line credential blob")
    }

    /// Parse from the opaque credential blob.
    pub fn from_blob(blob: &str) -> Result<Self> {
        serde_json::from_str(blob).context("parsing line credential blob")
    }
}

/// One brought-up LINE account: the instance-qualified transport id, the bare account handle, the
/// channel secret (for inbound signature checks), and the live bot client (for outbound push). Cheap
/// to clone — the [`LINE`] client shares one hyper connection pool behind an `Arc`.
#[derive(Clone)]
pub struct LineAccount {
    /// The instance-qualified transport id (`line/<handle>`).
    pub transport: TransportId,
    /// The bare account handle (the segment after `line/`).
    pub bare: String,
    /// The channel secret the inbound webhook signature is verified against.
    pub channel_secret: String,
    /// The live LINE Messaging API bot client (channel-access-token authenticated).
    pub line: LINE,
}

/// The bare handle (`<handle>`) inside an instance-qualified `line/...` transport id.
pub fn bare_account(transport: &TransportId) -> &str {
    transport
        .as_str()
        .strip_prefix("line/")
        .unwrap_or_else(|| transport.as_str())
}

/// A stable, non-reversible short handle derived from a channel secret — used when the operator
/// supplies no explicit channel id at auth time. Hex of the first 6 bytes of SHA-256(secret), so the
/// same channel always maps to the same `line/<handle>` without echoing the secret.
pub fn derive_handle(channel_secret: &str) -> String {
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    sha2::Digest::update(&mut hasher, channel_secret.as_bytes());
    let digest = sha2::Digest::finalize(hasher);
    let mut out = String::with_capacity(12);
    for byte in digest.iter().take(6) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_strips_family_prefix() {
        let t = TransportId::new("line/acme-bot".to_string());
        assert_eq!(bare_account(&t), "acme-bot");
    }

    #[test]
    fn derive_handle_is_stable_and_hex() {
        let a = derive_handle("s3cr3t");
        let b = derive_handle("s3cr3t");
        assert_eq!(a, b, "same secret -> same handle");
        assert_eq!(a.len(), 12);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(
            derive_handle("other"),
            a,
            "different secret -> different handle"
        );
        assert!(!a.contains("s3cr3t"), "handle never echoes the secret");
    }

    #[test]
    fn credential_blob_roundtrips() {
        let cred = StoredCredential {
            channel_access_token: "token-xyz".to_string(),
            channel_secret: "secret-abc".to_string(),
        };
        let blob = cred.to_blob().expect("serialize");
        let back = StoredCredential::from_blob(&blob).expect("parse");
        assert_eq!(back.channel_access_token, "token-xyz");
        assert_eq!(back.channel_secret, "secret-abc");
    }
}
