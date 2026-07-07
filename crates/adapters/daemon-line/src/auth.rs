// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! LINE bot login as a client-driven interactive-auth family (`daemon-interactive-auth-spec`).
//!
//! LINE is **bot-only** here (no mature Rust LINE *user* client exists), so there is no browser
//! redirect and no OAuth hop: the operator pastes the channel's long-lived **channel access token** +
//! **channel secret** (from the LINE Developers console) into a form. The flow is therefore a single
//! [`AuthChallenge::Form`] → [`AuthStepInput::Fields`] → completion, with the two secrets stored as
//! the account's opaque [`StoredCredential`](crate::account::StoredCredential) blob.
//!
//! The account handle (`line/<handle>`) is operator-chosen via an optional `channel_id` field (a
//! friendly id shown in the console) or, absent one, derived as a stable non-reversible short hash of
//! the channel secret ([`derive_handle`]). Either way the same channel always resolves to the same
//! transport instance, so a later profile bind / bring-up finds it. No network call happens here, so
//! the step is deterministic and testable offline.

use std::collections::BTreeMap;
use std::sync::Mutex;

use async_trait::async_trait;
use daemon_api::{
    ApiError, AuthChallenge, AuthFlowKind, AuthParamField, AuthProviderInfo, AuthStepInput,
};
use daemon_host::{
    AuthFlowFactory, AuthOutcome, AuthStepOutcome, CredentialSlotKind, PendingAuthFlow,
};
use daemon_protocol::TransportId;

use crate::account::{derive_handle, StoredCredential};
use crate::FAMILY;

/// The form field carrying the channel access token (required; the push-send bearer).
pub const FIELD_CHANNEL_ACCESS_TOKEN: &str = "channel_access_token";
/// The form field carrying the channel secret (required; the webhook signature key).
pub const FIELD_CHANNEL_SECRET: &str = "channel_secret";
/// The form field carrying an optional friendly channel id, used as the instance handle. When
/// omitted, the handle is derived as a short hash of the channel secret.
pub const FIELD_CHANNEL_ID: &str = "channel_id";

/// Build the [`AuthOutcome`] for a completed LINE bot login from the collected form `fields`. Shared
/// by the flow step and the unit tests. Errors when a required secret is missing/empty.
fn complete_from_fields(fields: &BTreeMap<String, String>) -> Result<AuthOutcome, ApiError> {
    let channel_access_token = fields
        .get(FIELD_CHANNEL_ACCESS_TOKEN)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            ApiError::Other(format!("line auth: missing `{FIELD_CHANNEL_ACCESS_TOKEN}`"))
        })?
        .to_string();
    let channel_secret = fields
        .get(FIELD_CHANNEL_SECRET)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::Other(format!("line auth: missing `{FIELD_CHANNEL_SECRET}`")))?
        .to_string();
    let handle = fields
        .get(FIELD_CHANNEL_ID)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| derive_handle(&channel_secret));

    let credential_blob = StoredCredential {
        channel_access_token,
        channel_secret,
    }
    .to_blob()
    .map_err(|e| ApiError::Other(format!("line auth: {e}")))?;

    let transport_instance = TransportId::new(format!("{FAMILY}/{handle}"));
    Ok(AuthOutcome {
        credential_blob,
        // The family-derived default key; the host may override it with a bind-supplied credential_ref.
        credential_ref: format!("{FAMILY}/{handle}"),
        account_label: format!("LINE {handle}"),
        transport_instance,
        slot: CredentialSlotKind::Derived,
    })
}

/// The form challenge the flow presents at `begin`: collect the two channel secrets (+ optional id).
fn bot_token_form() -> AuthChallenge {
    AuthChallenge::Form {
        title: "LINE bot credentials (Messaging API channel)".to_string(),
        fields: vec![
            AuthParamField {
                key: FIELD_CHANNEL_ACCESS_TOKEN.to_string(),
                label: "Channel access token".to_string(),
                required: true,
            },
            AuthParamField {
                key: FIELD_CHANNEL_SECRET.to_string(),
                label: "Channel secret".to_string(),
                required: true,
            },
            AuthParamField {
                key: FIELD_CHANNEL_ID.to_string(),
                label: "Channel id (optional account handle)".to_string(),
                required: false,
            },
        ],
    }
}

/// The LINE interactive-auth factory: registered with the node so a client can drive `line` bot
/// login over the wire `AuthApi`. Stateless — the single step consumes the pasted form fields.
pub struct LineAuthFlowFactory;

impl LineAuthFlowFactory {
    /// A new factory (no captured state; LINE bot auth needs no store root or redirect).
    pub fn new() -> Self {
        Self
    }
}

impl Default for LineAuthFlowFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AuthFlowFactory for LineAuthFlowFactory {
    fn family(&self) -> &str {
        FAMILY
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: FAMILY.to_string(),
            flow_kind: AuthFlowKind::BotToken,
            display_name: "LINE (bot)".to_string(),
            params_schema: vec![
                AuthParamField {
                    key: FIELD_CHANNEL_ACCESS_TOKEN.to_string(),
                    label: "Channel access token".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: FIELD_CHANNEL_SECRET.to_string(),
                    label: "Channel secret".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: FIELD_CHANNEL_ID.to_string(),
                    label: "Channel id (optional account handle)".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn begin(
        &self,
        _params: &BTreeMap<String, String>,
        _redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        // No browser hop: the whole exchange is the one form. `params`/`redirect_uri` are unused
        // (LINE bot auth has no discovery params and no redirect).
        Ok(Box::new(LinePendingFlow {
            done: Mutex::new(false),
        }))
    }
}

/// A parked LINE bot-token flow. Single-step: the first `step(Fields{..})` completes it. The `done`
/// guard makes completion single-use (a second step errors), matching the registry's park semantics.
struct LinePendingFlow {
    done: Mutex<bool>,
}

#[async_trait]
impl PendingAuthFlow for LinePendingFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        bot_token_form()
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        let AuthStepInput::Fields(fields) = input else {
            return Err(ApiError::Other(
                "line bot auth expects the pasted credential form fields".into(),
            ));
        };
        {
            let mut done = self.done.lock().unwrap();
            if *done {
                return Err(ApiError::Other("line auth flow already completed".into()));
            }
            *done = true;
        }
        Ok(AuthStepOutcome::Completed(complete_from_fields(&fields)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[tokio::test]
    async fn token_step_completes_with_derived_handle() {
        let factory = LineAuthFlowFactory::new();
        let flow = factory
            .begin(&BTreeMap::new(), "")
            .await
            .expect("begin succeeds");
        assert!(matches!(
            flow.initial_challenge(),
            AuthChallenge::Form { .. }
        ));

        // Obviously-fake fixtures (no real token/secret, no network).
        let input = AuthStepInput::Fields(fields(&[
            (FIELD_CHANNEL_ACCESS_TOKEN, "fake-access-token-000"),
            (FIELD_CHANNEL_SECRET, "fake-channel-secret-000"),
        ]));
        let outcome = flow.step(input).await.expect("step completes");
        let AuthStepOutcome::Completed(out) = outcome else {
            panic!("expected completion");
        };
        // No channel_id supplied -> handle derived from the secret.
        let handle = derive_handle("fake-channel-secret-000");
        assert_eq!(out.transport_instance.as_str(), format!("line/{handle}"));
        assert_eq!(out.credential_ref, format!("line/{handle}"));
        assert!(matches!(out.slot, CredentialSlotKind::Derived));
        // The blob round-trips to the pasted secrets.
        let cred = StoredCredential::from_blob(&out.credential_blob).expect("blob parses");
        assert_eq!(cred.channel_access_token, "fake-access-token-000");
        assert_eq!(cred.channel_secret, "fake-channel-secret-000");
    }

    #[tokio::test]
    async fn explicit_channel_id_becomes_the_handle() {
        let out = complete_from_fields(&fields(&[
            (FIELD_CHANNEL_ACCESS_TOKEN, "fake-token"),
            (FIELD_CHANNEL_SECRET, "fake-secret"),
            (FIELD_CHANNEL_ID, "acme-bot"),
        ]))
        .expect("completes");
        assert_eq!(out.transport_instance.as_str(), "line/acme-bot");
        assert_eq!(out.account_label, "LINE acme-bot");
    }

    #[tokio::test]
    async fn missing_secret_is_rejected() {
        let result = complete_from_fields(&fields(&[(FIELD_CHANNEL_ACCESS_TOKEN, "fake-token")]));
        assert!(
            matches!(result, Err(ApiError::Other(_))),
            "missing channel_secret is rejected"
        );
    }

    #[tokio::test]
    async fn flow_is_single_use() {
        let flow = LinePendingFlow {
            done: Mutex::new(false),
        };
        let ok = flow
            .step(AuthStepInput::Fields(fields(&[
                (FIELD_CHANNEL_ACCESS_TOKEN, "t"),
                (FIELD_CHANNEL_SECRET, "s"),
            ])))
            .await;
        assert!(ok.is_ok());
        let again = flow
            .step(AuthStepInput::Fields(fields(&[
                (FIELD_CHANNEL_ACCESS_TOKEN, "t"),
                (FIELD_CHANNEL_SECRET, "s"),
            ])))
            .await;
        assert!(again.is_err(), "second completion is rejected");
    }
}
