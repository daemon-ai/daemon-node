// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The per-account credential blob and small account helpers.
//!
//! A WhatsApp **account** is a transport instance (`whatsapp/<handle>`) in one of two modes:
//!
//! * **user** — a WhatsApp Web linked device (whatsapp-rust). The credential store is authoritative
//!   over the session: we persist the paired `Device` snapshot (serde JSON) as the blob, and restore
//!   it into an in-memory backend at bring-up. This is the session-of-record (there is no on-disk
//!   SQLite store — see the crate `Cargo.toml` for why).
//! * **bot** — a Meta Cloud API business number (wacloudapi). The blob is the bearer `access_token`
//!   plus the `phone_number_id`.
//!
//! The blob deliberately holds the user-mode `Device` as an opaque `serde_json::Value` so this module
//! (and the persisted schema) stays free of any SDK type — the whatsapp-rust backend does the typed
//! `Device` (de)serialization.

use serde::{Deserialize, Serialize};

use daemon_protocol::TransportId;

/// The transport family this adapter provisions (`AccountProvisioning::bound_accounts`).
pub const FAMILY: &str = "whatsapp";

/// The credential-store blob for one WhatsApp account. Serialized as JSON under the account's
/// credential-ref; the `mode` tag selects the backend at bring-up.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum StoredCredential {
    /// A WhatsApp Web linked-device (whatsapp-rust) session.
    User {
        /// The linked account JID (`<number>@s.whatsapp.net`), for labelling / the transport id.
        jid: String,
        /// The paired `wacore::store::Device` snapshot, serialized (opaque here; the whatsapp-rust
        /// backend deserializes it into an `InMemoryBackend`).
        device: serde_json::Value,
    },
    /// A Meta Cloud API (wacloudapi) bot number.
    Bot {
        /// The Meta Business Platform bearer access token.
        access_token: String,
        /// The WhatsApp Business phone-number id.
        phone_number_id: String,
    },
}

impl StoredCredential {
    /// Serialize to the opaque credential blob.
    pub fn to_blob(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Parse from the opaque credential blob.
    pub fn from_blob(blob: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(blob)
    }
}

/// The bare account handle inside an instance-qualified `whatsapp/...` transport id.
pub fn bare_account(transport: &TransportId) -> &str {
    transport
        .as_str()
        .strip_prefix("whatsapp/")
        .unwrap_or_else(|| transport.as_str())
}

/// The instance-qualified transport id for a bare account `handle` (`whatsapp/<handle>`).
pub fn transport_for(handle: &str) -> TransportId {
    TransportId::new(format!("{FAMILY}/{handle}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_strips_family_prefix() {
        let t = TransportId::new("whatsapp/15551234567".to_string());
        assert_eq!(bare_account(&t), "15551234567");
    }

    #[test]
    fn transport_for_qualifies() {
        assert_eq!(
            transport_for("15551234567").as_str(),
            "whatsapp/15551234567"
        );
    }

    #[test]
    fn bot_blob_roundtrips() {
        let c = StoredCredential::Bot {
            access_token: "tok".into(),
            phone_number_id: "123".into(),
        };
        let blob = c.to_blob().unwrap();
        assert!(blob.contains("\"mode\":\"bot\""));
        match StoredCredential::from_blob(&blob).unwrap() {
            StoredCredential::Bot {
                access_token,
                phone_number_id,
            } => {
                assert_eq!(access_token, "tok");
                assert_eq!(phone_number_id, "123");
            }
            _ => panic!("expected bot"),
        }
    }

    #[test]
    fn user_blob_roundtrips() {
        let c = StoredCredential::User {
            jid: "15551234567@s.whatsapp.net".into(),
            device: serde_json::json!({"stub": true}),
        };
        let blob = c.to_blob().unwrap();
        match StoredCredential::from_blob(&blob).unwrap() {
            StoredCredential::User { jid, device } => {
                assert_eq!(jid, "15551234567@s.whatsapp.net");
                assert_eq!(device, serde_json::json!({"stub": true}));
            }
            _ => panic!("expected user"),
        }
    }
}
