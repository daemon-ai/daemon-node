// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The Meta Cloud API bot backend (`wacloudapi`).
//!
//! A `mode = "bot"` account authenticates with a Meta Business Platform bearer token + phone-number
//! id and sends via the Cloud API. Outbound send is fully wired here; group membership administration
//! is not a Cloud API capability (business numbers cannot add/remove group participants), so
//! [`WaBackend::membership`] reports it off. Inbound for the Cloud API arrives over Meta webhooks,
//! which require the HTTP surface — that is Phase 2 (`serve` wires no inbound for bot accounts).

use async_trait::async_trait;

use daemon_api::{ApiError, MembershipOps};

use crate::backend::WaBackend;

/// A live Cloud API account: the `wacloudapi` client bound to one business phone number.
pub struct BotBackend {
    client: wacloudapi::Client,
}

impl BotBackend {
    /// Construct the backend over a bearer `access_token` + `phone_number_id`.
    pub fn new(access_token: &str, phone_number_id: &str) -> Self {
        Self {
            client: wacloudapi::Client::new(access_token, phone_number_id),
        }
    }
}

#[async_trait]
impl WaBackend for BotBackend {
    async fn send_text(&self, to: &str, text: &str) -> Result<(), ApiError> {
        // The Cloud API expects a country-coded recipient with no `+`/spaces.
        let recipient = crate::mapping::bot_recipient(to);
        self.client
            .messages()
            .send_text(&recipient, text)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("whatsapp cloud send: {e}")))
    }

    fn membership(&self) -> MembershipOps {
        // The Cloud API has no group-membership administration surface.
        MembershipOps::default()
    }
}
