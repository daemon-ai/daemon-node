// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Discord token login as a client-driven interactive-auth family (`daemon-interactive-auth-spec`).
//!
//! Discord auth is a **single token step** for both account modes — there is no browser hop. The
//! client's `auth_begin` names the target `credential_ref`; the node presents a [`AuthChallenge::Form`]
//! collecting the token; `auth_step` validates it against Discord (`GET /users/@me`), resolves the
//! account's user id, and completes. The account [`DiscordMode`] (fixed per factory from config)
//! selects the advertised [`AuthFlowKind`] (`BotToken` | `UserToken`) and the human label.
//!
//! `serenity_self` sends the token verbatim (it prepends no `Bot ` and panics on a prefixed token),
//! so the token is [`sanitize_token`](crate::account::sanitize_token)'d before it ever reaches the
//! SDK — an operator may paste a `Bot …` bot token and it is normalised to the raw form.

use std::collections::BTreeMap;
use std::sync::Mutex;

use async_trait::async_trait;
use daemon_api::{
    ApiError, AuthChallenge, AuthFieldKind, AuthFlowKind, AuthParamField, AuthProviderInfo,
    AuthStepInput,
};
use daemon_host::{
    AuthFlowFactory, AuthOutcome, AuthStepOutcome, CredentialSlotKind, PendingAuthFlow,
};
use daemon_protocol::TransportId;
use serenity_self::http::Http;

use crate::account::{sanitize_token, StoredCredential};
use crate::config::DiscordMode;
use crate::FAMILY;

/// The `auth_begin` param naming the account's stable credential/store key (required) — where the
/// resulting token blob lands in the `CredentialStore`.
pub const PARAM_CREDENTIAL_REF: &str = "credential_ref";
/// The `Form`-challenge field carrying the pasted token.
pub const FIELD_TOKEN: &str = "token";

/// The Discord interactive-auth factory: registered with the node so a client can drive a `discord`
/// token login over the wire `AuthApi`. Fixed to one [`DiscordMode`] (from the resolved config) so
/// the advertised flow kind + label are honest.
pub struct DiscordAuthFlowFactory {
    mode: DiscordMode,
}

impl DiscordAuthFlowFactory {
    /// A factory advertising the `discord` family in the given account `mode`.
    pub fn new(mode: DiscordMode) -> Self {
        Self { mode }
    }
}

#[async_trait]
impl AuthFlowFactory for DiscordAuthFlowFactory {
    fn family(&self) -> &str {
        FAMILY
    }

    fn provider_info(&self) -> AuthProviderInfo {
        let (flow_kind, display_name) = match self.mode {
            DiscordMode::Bot => (AuthFlowKind::BotToken, "Discord (bot token)"),
            DiscordMode::User => (AuthFlowKind::UserToken, "Discord (user token)"),
        };
        AuthProviderInfo {
            family: FAMILY.to_string(),
            flow_kind,
            display_name: display_name.to_string(),
            params_schema: vec![AuthParamField {
                key: PARAM_CREDENTIAL_REF.to_string(),
                label: "Account credential ref".to_string(),
                required: true,
                ..Default::default()
            }],
        }
    }

    async fn begin(
        &self,
        params: &BTreeMap<String, String>,
        _redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        let credential_ref = params
            .get(PARAM_CREDENTIAL_REF)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                ApiError::Other(format!("discord auth: missing `{PARAM_CREDENTIAL_REF}`"))
            })?
            .clone();
        Ok(Box::new(DiscordPendingFlow {
            credential_ref,
            mode: self.mode,
            spent: Mutex::new(false),
        }))
    }
}

/// A parked Discord token flow: presents a one-field token form, then validates + completes on the
/// single step. `spent` guards the single-use completion under `&self`.
struct DiscordPendingFlow {
    credential_ref: String,
    mode: DiscordMode,
    spent: Mutex<bool>,
}

impl DiscordPendingFlow {
    /// The token-collection form challenge (both modes present the same single field).
    fn token_form(mode: DiscordMode) -> AuthChallenge {
        let title = match mode {
            DiscordMode::Bot => "Paste your Discord bot token",
            DiscordMode::User => "Paste your Discord user token",
        };
        AuthChallenge::Form {
            title: title.to_string(),
            fields: vec![AuthParamField {
                key: FIELD_TOKEN.to_string(),
                label: "Token".to_string(),
                required: true,
                // The bot/user token is a secret pasted into a form — mask it.
                kind: AuthFieldKind::Password,
                ..Default::default()
            }],
        }
    }
}

/// Assemble the completed [`AuthOutcome`] from a validated identity. Split out from the network step
/// so it is unit-testable without a live Discord call: it builds the stored blob (`{token, mode}`),
/// the `discord/<user_id>` transport instance, and the credential-slot mapping.
fn build_outcome(
    credential_ref: String,
    mode: DiscordMode,
    user_id: u64,
    account_label: String,
    token: String,
) -> Result<AuthOutcome, ApiError> {
    let credential_blob = StoredCredential { token, mode }
        .to_blob()
        .map_err(|e| ApiError::Other(format!("discord: serializing credential blob: {e}")))?;
    Ok(AuthOutcome {
        credential_blob,
        credential_ref,
        account_label,
        transport_instance: TransportId::new(format!("{FAMILY}/{user_id}")),
        slot: CredentialSlotKind::Derived,
    })
}

#[async_trait]
impl PendingAuthFlow for DiscordPendingFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        Self::token_form(self.mode)
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        let AuthStepInput::Fields(fields) = input else {
            return Err(ApiError::Other(
                "discord auth expects the token form fields".into(),
            ));
        };
        {
            let mut spent = self.spent.lock().unwrap();
            if *spent {
                return Err(ApiError::Other(
                    "discord auth flow already completed".into(),
                ));
            }
            *spent = true;
        }
        let token = fields
            .get(FIELD_TOKEN)
            .map(|t| sanitize_token(t))
            .filter(|t| !t.is_empty())
            .ok_or_else(|| ApiError::Other(format!("discord auth: missing `{FIELD_TOKEN}`")))?;

        // Validate the token by resolving the account's own user (`GET /users/@me`). `sanitize_token`
        // already stripped any `Bot `/`Bearer ` prefix, so `Http::new` never hits its prefix panic.
        let http = Http::new(&token);
        let me = http
            .get_current_user()
            .await
            .map_err(|e| ApiError::Other(format!("discord auth: token validation failed: {e}")))?;
        let label = format!("{} ({})", me.name, self.mode.as_str());
        let outcome = build_outcome(
            self.credential_ref.clone(),
            self.mode,
            me.id.get(),
            label,
            token,
        )?;
        Ok(AuthStepOutcome::Completed(outcome))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_info_reflects_mode() {
        let bot = DiscordAuthFlowFactory::new(DiscordMode::Bot).provider_info();
        assert_eq!(bot.family, "discord");
        assert_eq!(bot.flow_kind, AuthFlowKind::BotToken);
        assert!(bot
            .params_schema
            .iter()
            .any(|f| f.key == PARAM_CREDENTIAL_REF));

        let user = DiscordAuthFlowFactory::new(DiscordMode::User).provider_info();
        assert_eq!(user.flow_kind, AuthFlowKind::UserToken);
    }

    #[test]
    fn initial_challenge_is_a_token_form() {
        let flow = DiscordPendingFlow {
            credential_ref: "acct".into(),
            mode: DiscordMode::Bot,
            spent: Mutex::new(false),
        };
        match flow.initial_challenge() {
            AuthChallenge::Form { fields, .. } => {
                assert!(fields.iter().any(|f| f.key == FIELD_TOKEN && f.required));
            }
            other => panic!("expected a token Form, got {other:?}"),
        }
    }

    #[test]
    fn build_outcome_assembles_blob_and_transport() {
        // The token-step completion assembly (the half that runs after `GET /users/@me` resolves the
        // identity), exercised without any live network — a validated (user_id, name) is enough.
        let outcome = build_outcome(
            "acct-1".to_string(),
            DiscordMode::User,
            42,
            "someuser (user)".to_string(),
            "raw.user.token".to_string(),
        )
        .expect("outcome assembles");
        assert_eq!(outcome.credential_ref, "acct-1");
        assert_eq!(outcome.transport_instance.as_str(), "discord/42");
        assert!(matches!(outcome.slot, CredentialSlotKind::Derived));
        let stored = StoredCredential::from_blob(&outcome.credential_blob).unwrap();
        assert_eq!(stored.token, "raw.user.token");
        assert_eq!(stored.mode, DiscordMode::User);
    }
}
