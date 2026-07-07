// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! WhatsApp interactive auth: one family (`whatsapp`), two modes selected by the `mode` param.
//!
//! * **user** (`AuthFlowKind::QrPairing`) — start a WhatsApp Web client, present the pairing QR as a
//!   [`AuthChallenge::Qr`], and [`AuthStepInput::Poll`] until the phone links the device. On success
//!   the paired `Device` snapshot is persisted as the session blob.
//! * **bot** (`AuthFlowKind::BotToken`) — collect the Meta Cloud API `access_token` +
//!   `phone_number_id` via a [`AuthChallenge::Form`] and complete in one step.
//!
//! `step` takes `&self` (flows advance in place); the user flow keeps its live [`Pairing`] across
//! polls behind the flow's own interior mutability.

use std::collections::BTreeMap;

use async_trait::async_trait;

use daemon_api::{
    ApiError, AuthChallenge, AuthFlowKind, AuthParamField, AuthProviderInfo, AuthStepInput,
};
use daemon_host::{
    AuthFlowFactory, AuthOutcome, AuthStepOutcome, CredentialSlotKind, PendingAuthFlow,
};

use crate::account::{transport_for, StoredCredential, FAMILY};
use crate::backend_user::Pairing;

/// The `mode` param naming which flow to run (`user` | `bot`). Required.
pub const PARAM_MODE: &str = "mode";
/// The stable credential/store key the resulting blob is stored under. Required.
pub const PARAM_CREDENTIAL_REF: &str = "credential_ref";
/// The bot-mode Cloud API bearer token (collected in the `Form` step).
pub const PARAM_ACCESS_TOKEN: &str = "access_token";
/// The bot-mode Cloud API business phone-number id (collected in the `Form` step).
pub const PARAM_PHONE_NUMBER_ID: &str = "phone_number_id";

/// How often the client should re-poll while waiting for the phone to scan the QR.
const QR_POLL_INTERVAL_MS: u64 = 3000;

/// The interactive-auth factory registered with the node so a client can drive WhatsApp login over
/// the wire `AuthApi`.
pub struct WhatsappAuthFlowFactory;

impl WhatsappAuthFlowFactory {
    /// Construct the factory (stateless).
    pub fn new() -> Self {
        Self
    }
}

impl Default for WhatsappAuthFlowFactory {
    fn default() -> Self {
        Self::new()
    }
}

fn require<'a>(params: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, ApiError> {
    params
        .get(key)
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::Other(format!("whatsapp auth: missing `{key}`")))
}

#[async_trait]
impl AuthFlowFactory for WhatsappAuthFlowFactory {
    fn family(&self) -> &str {
        FAMILY
    }

    fn provider_info(&self) -> AuthProviderInfo {
        // The primary hint is QR pairing (the user flow); `mode = bot` switches to the Cloud API form.
        AuthProviderInfo {
            family: FAMILY.to_string(),
            flow_kind: AuthFlowKind::QrPairing,
            display_name: "WhatsApp".to_string(),
            params_schema: vec![
                AuthParamField {
                    key: PARAM_MODE.to_string(),
                    label: "Mode (user = WhatsApp Web QR, bot = Cloud API)".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: PARAM_CREDENTIAL_REF.to_string(),
                    label: "Account credential ref".to_string(),
                    required: true,
                },
            ],
        }
    }

    async fn begin(
        &self,
        params: &BTreeMap<String, String>,
        _redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        let credential_ref = require(params, PARAM_CREDENTIAL_REF)?.to_string();
        let mode = params.get(PARAM_MODE).map(String::as_str).unwrap_or("user");
        match mode {
            "bot" => Ok(Box::new(PendingBotFlow { credential_ref })),
            "user" => {
                let pairing = Pairing::start().await?;
                let initial_qr = pairing.current_qr().unwrap_or_default();
                Ok(Box::new(PendingUserFlow {
                    credential_ref,
                    initial_qr,
                    pairing,
                }))
            }
            other => Err(ApiError::Other(format!(
                "whatsapp auth: unknown mode `{other}` (expected `user` or `bot`)"
            ))),
        }
    }
}

/// The bot (Cloud API) flow: a single `Form` collecting the token + phone-number id.
struct PendingBotFlow {
    credential_ref: String,
}

fn bot_form() -> AuthChallenge {
    AuthChallenge::Form {
        title: "WhatsApp Cloud API credentials".to_string(),
        fields: vec![
            AuthParamField {
                key: PARAM_ACCESS_TOKEN.to_string(),
                label: "Access token".to_string(),
                required: true,
            },
            AuthParamField {
                key: PARAM_PHONE_NUMBER_ID.to_string(),
                label: "Phone number id".to_string(),
                required: true,
            },
        ],
    }
}

#[async_trait]
impl PendingAuthFlow for PendingBotFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        bot_form()
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        let AuthStepInput::Fields(fields) = input else {
            return Err(ApiError::Other(
                "whatsapp bot auth expects the credential form fields".into(),
            ));
        };
        let access_token = require(&fields, PARAM_ACCESS_TOKEN)?.to_string();
        let phone_number_id = require(&fields, PARAM_PHONE_NUMBER_ID)?.to_string();
        let blob = StoredCredential::Bot {
            access_token,
            phone_number_id: phone_number_id.clone(),
        }
        .to_blob()
        .map_err(|e| ApiError::Other(format!("whatsapp bot blob: {e}")))?;
        Ok(AuthStepOutcome::Completed(AuthOutcome {
            credential_blob: blob,
            credential_ref: self.credential_ref.clone(),
            account_label: phone_number_id.clone(),
            transport_instance: transport_for(&phone_number_id),
            slot: CredentialSlotKind::Derived,
        }))
    }
}

/// The user (WhatsApp Web) flow: present the pairing QR, poll until linked, then persist the device.
struct PendingUserFlow {
    credential_ref: String,
    initial_qr: String,
    pairing: Pairing,
}

/// The pure decision for one `Poll` step (extracted so it is testable without a live SDK client):
/// once linked the flow completes; otherwise it re-presents the freshest QR challenge.
enum PollStep {
    Complete,
    Retry(AuthChallenge),
}

fn poll_step(linked: bool, latest_qr: Option<String>, initial_qr: &str) -> PollStep {
    if linked {
        return PollStep::Complete;
    }
    let payload = latest_qr.unwrap_or_else(|| initial_qr.to_string());
    PollStep::Retry(AuthChallenge::Qr {
        payload,
        image: None,
        poll_interval_ms: QR_POLL_INTERVAL_MS,
    })
}

#[async_trait]
impl PendingAuthFlow for PendingUserFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        AuthChallenge::Qr {
            payload: self.initial_qr.clone(),
            image: None,
            poll_interval_ms: QR_POLL_INTERVAL_MS,
        }
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        if !matches!(input, AuthStepInput::Poll) {
            return Err(ApiError::Other("whatsapp QR pairing expects a poll".into()));
        }
        match poll_step(
            self.pairing.is_linked(),
            self.pairing.current_qr(),
            &self.initial_qr,
        ) {
            PollStep::Retry(challenge) => Ok(AuthStepOutcome::Challenge(challenge)),
            PollStep::Complete => {
                let (jid, device) = self.pairing.device_blob().await?;
                let handle = jid.split('@').next().unwrap_or(&jid).to_string();
                let blob = StoredCredential::User {
                    jid: jid.clone(),
                    device,
                }
                .to_blob()
                .map_err(|e| ApiError::Other(format!("whatsapp user blob: {e}")))?;
                Ok(AuthStepOutcome::Completed(AuthOutcome {
                    credential_blob: blob,
                    credential_ref: self.credential_ref.clone(),
                    account_label: jid,
                    transport_instance: transport_for(&handle),
                    slot: CredentialSlotKind::Derived,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poll_step_retries_with_qr_until_linked() {
        // Not linked, a fresh QR available: re-present it.
        match poll_step(false, Some("QR-2".to_string()), "QR-1") {
            PollStep::Retry(AuthChallenge::Qr { payload, .. }) => assert_eq!(payload, "QR-2"),
            _ => panic!("expected a QR retry challenge"),
        }
        // Not linked, no fresh QR: fall back to the initial one.
        match poll_step(false, None, "QR-1") {
            PollStep::Retry(AuthChallenge::Qr { payload, .. }) => assert_eq!(payload, "QR-1"),
            _ => panic!("expected the initial QR"),
        }
        // Linked: complete.
        assert!(matches!(poll_step(true, None, "QR-1"), PollStep::Complete));
    }

    #[tokio::test]
    async fn bot_form_step_completes_from_fields() {
        let flow = PendingBotFlow {
            credential_ref: "wa-bot".to_string(),
        };
        assert!(matches!(
            flow.initial_challenge(),
            AuthChallenge::Form { .. }
        ));

        let mut fields = BTreeMap::new();
        fields.insert(PARAM_ACCESS_TOKEN.to_string(), "EAAB-token".to_string());
        fields.insert(PARAM_PHONE_NUMBER_ID.to_string(), "10987654321".to_string());

        let outcome = flow
            .step(AuthStepInput::Fields(fields))
            .await
            .expect("bot form completes");
        match outcome {
            AuthStepOutcome::Completed(o) => {
                assert_eq!(o.credential_ref, "wa-bot");
                assert_eq!(o.account_label, "10987654321");
                assert_eq!(o.transport_instance.as_str(), "whatsapp/10987654321");
                assert!(matches!(o.slot, CredentialSlotKind::Derived));
                assert!(o.credential_blob.contains("\"mode\":\"bot\""));
            }
            AuthStepOutcome::Challenge(_) => panic!("expected completion"),
        }
    }

    #[tokio::test]
    async fn bot_flow_rejects_a_poll() {
        let flow = PendingBotFlow {
            credential_ref: "wa-bot".to_string(),
        };
        assert!(flow.step(AuthStepInput::Poll).await.is_err());
    }
}
