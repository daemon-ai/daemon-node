// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Telegram login as a client-driven interactive-auth family (the Phase 0 challenge/response surface).
//!
//! Telegram has two genuinely different logins, selected by the `mode` begin-param:
//!
//! - **user** ([`AccountMode::User`], a real multi-step [`AuthFlowKind::PhoneOtp`]): the first
//!   challenge collects the phone; submitting it requests the login code (a `Form{code}` challenge);
//!   submitting the code signs in, and *if 2FA is enabled* returns a further `Form{password}`
//!   challenge before completing.
//! - **bot** ([`AccountMode::Bot`], a single [`AuthFlowKind::BotToken`] step): the first challenge
//!   collects the BotFather token; submitting it signs the bot in and completes.
//!
//! The flow logic is SDK-agnostic: it drives a [`LoginBackend`] trait whose real implementation
//! (`crate::client`) wraps grammers, so the multi-step state machine is unit testable with a mock
//! backend (no grammers, no network). The completed [`AuthOutcome`] slots as
//! [`CredentialSlotKind::Derived`] — a transport account, not a provider API key.

use std::collections::BTreeMap;
use std::path::PathBuf;
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

use crate::account::AccountMode;
use crate::FAMILY;

/// The `auth_begin` param selecting the login mode (`"user"` | `"bot"`); required.
pub const PARAM_MODE: &str = "mode";
/// The `auth_begin` param naming the account's stable credential/store key; required. Keys both the
/// on-disk session store and where the resulting blob lands.
pub const PARAM_CREDENTIAL_REF: &str = "credential_ref";
/// The `Form` field collecting the user phone number (user flow, step 1).
pub const PARAM_PHONE: &str = "phone";
/// The `Form` field collecting the login code (user flow, step 2).
pub const PARAM_CODE: &str = "code";
/// The `Form` field collecting the 2FA password (user flow, step 3, only when 2FA is on).
pub const PARAM_PASSWORD: &str = "password";
/// The `Form` field collecting the BotFather token (bot flow, single step).
pub const PARAM_TOKEN: &str = "token";

/// Whether submitting the login code completed the sign-in or a 2FA password is still required.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeStep {
    /// Sign-in complete; drive [`LoginBackend::finish`].
    Done,
    /// The account has 2FA; collect a password and call [`LoginBackend::submit_password`].
    PasswordRequired,
}

/// The resolved product of a completed login: the account identity + the credential blob to persist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoginIdentity {
    /// The resolved Telegram account id (`get_me().id()`).
    pub account_id: i64,
    /// A human label for the account (username / display name).
    pub label: String,
    /// The opaque [`crate::StoredSession`] blob (mode + bot token), already serialized by the backend
    /// (which alone knows the mode + token).
    pub credential_blob: String,
}

/// The grammers-agnostic seam the [`TelegramPendingFlow`] drives. The real implementation
/// (`crate::client`) wraps a connected grammers client; a mock implements it for the flow tests. All
/// methods take `&self` so the flow (which steps in place under `&self`) can drive it; the backend
/// carries its continuation state (login token, password token) internally via interior mutability.
#[async_trait]
pub trait LoginBackend: Send + Sync {
    /// User flow: request the login code be sent to `phone` (stashes the login token internally).
    async fn request_code(&self, phone: &str) -> Result<(), ApiError>;
    /// User flow: submit the login `code`, reporting whether a 2FA password is still required.
    async fn submit_code(&self, code: &str) -> Result<CodeStep, ApiError>;
    /// User flow: submit the 2FA `password`.
    async fn submit_password(&self, password: &str) -> Result<(), ApiError>;
    /// Bot flow: sign in with a BotFather `token`.
    async fn bot_sign_in(&self, token: &str) -> Result<(), ApiError>;
    /// Persist the session and resolve the account identity after a successful sign-in.
    async fn finish(&self) -> Result<LoginIdentity, ApiError>;
}

/// The step the flow is parked at (a fieldless, `Copy` state machine; the payloads live on the
/// backend + the completed [`AuthOutcome`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlowState {
    AwaitToken,
    AwaitPhone,
    AwaitCode,
    AwaitPassword,
    Spent,
}

/// A parked Telegram login flow: the grammers-agnostic [`LoginBackend`], the current [`FlowState`]
/// (behind a `Mutex` so steps advance in place under `&self`), and the account keying.
pub struct TelegramPendingFlow {
    backend: std::sync::Arc<dyn LoginBackend>,
    mode: AccountMode,
    credential_ref: String,
    state: Mutex<FlowState>,
}

impl TelegramPendingFlow {
    /// A flow over `backend` for `mode`, keyed by `credential_ref`. Public so the flow tests can
    /// stage a mock backend exactly the way [`TelegramAuthFlowFactory::begin`] would.
    pub fn new(
        backend: std::sync::Arc<dyn LoginBackend>,
        mode: AccountMode,
        credential_ref: impl Into<String>,
    ) -> Self {
        let state = match mode {
            AccountMode::Bot => FlowState::AwaitToken,
            AccountMode::User => FlowState::AwaitPhone,
        };
        Self {
            backend,
            mode,
            credential_ref: credential_ref.into(),
            state: Mutex::new(state),
        }
    }

    fn set_state(&self, s: FlowState) {
        *self.state.lock().unwrap() = s;
    }

    fn current(&self) -> FlowState {
        *self.state.lock().unwrap()
    }

    fn outcome(&self, id: LoginIdentity) -> AuthStepOutcome {
        AuthStepOutcome::Completed(AuthOutcome {
            credential_blob: id.credential_blob,
            credential_ref: self.credential_ref.clone(),
            account_label: id.label,
            transport_instance: TransportId::new(format!("{FAMILY}/{}", id.account_id)),
            slot: CredentialSlotKind::Derived,
        })
    }
}

/// A single-field `Form` challenge. `kind` renders + validates the field (wire v38): a token/2FA
/// password is [`AuthFieldKind::Password`] (masked), a login code is [`AuthFieldKind::Number`].
fn form_field(title: &str, key: &str, label: &str, kind: AuthFieldKind) -> AuthChallenge {
    AuthChallenge::Form {
        title: title.to_string(),
        fields: vec![AuthParamField {
            key: key.to_string(),
            label: label.to_string(),
            required: true,
            kind,
            ..Default::default()
        }],
    }
}

/// Extract a required, non-empty field from a `Fields` step input.
fn required(fields: &BTreeMap<String, String>, key: &str) -> Result<String, ApiError> {
    fields
        .get(key)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::Other(format!("telegram auth: missing `{key}`")))
}

#[async_trait]
impl PendingAuthFlow for TelegramPendingFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        match self.mode {
            AccountMode::Bot => form_field(
                "Enter the bot token",
                PARAM_TOKEN,
                "BotFather token",
                AuthFieldKind::Password,
            ),
            AccountMode::User => form_field(
                "Enter your phone number",
                PARAM_PHONE,
                "Phone number",
                AuthFieldKind::Text,
            ),
        }
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        let AuthStepInput::Fields(fields) = input else {
            return Err(ApiError::Other(
                "telegram auth expects the collected form fields".into(),
            ));
        };
        match self.current() {
            FlowState::AwaitToken => {
                let token = required(&fields, PARAM_TOKEN)?;
                self.backend.bot_sign_in(&token).await?;
                let id = self.backend.finish().await?;
                self.set_state(FlowState::Spent);
                Ok(self.outcome(id))
            }
            FlowState::AwaitPhone => {
                let phone = required(&fields, PARAM_PHONE)?;
                self.backend.request_code(&phone).await?;
                self.set_state(FlowState::AwaitCode);
                Ok(AuthStepOutcome::Challenge(form_field(
                    "Enter the login code we sent you",
                    PARAM_CODE,
                    "Login code",
                    AuthFieldKind::Number,
                )))
            }
            FlowState::AwaitCode => {
                let code = required(&fields, PARAM_CODE)?;
                match self.backend.submit_code(&code).await? {
                    CodeStep::Done => {
                        let id = self.backend.finish().await?;
                        self.set_state(FlowState::Spent);
                        Ok(self.outcome(id))
                    }
                    CodeStep::PasswordRequired => {
                        self.set_state(FlowState::AwaitPassword);
                        Ok(AuthStepOutcome::Challenge(form_field(
                            "Enter your 2FA password",
                            PARAM_PASSWORD,
                            "2FA password",
                            AuthFieldKind::Password,
                        )))
                    }
                }
            }
            FlowState::AwaitPassword => {
                let password = required(&fields, PARAM_PASSWORD)?;
                self.backend.submit_password(&password).await?;
                let id = self.backend.finish().await?;
                self.set_state(FlowState::Spent);
                Ok(self.outcome(id))
            }
            FlowState::Spent => Err(ApiError::Other(
                "telegram auth flow already completed".into(),
            )),
        }
    }
}

/// The Telegram interactive-auth factory: registered with the node so a client can drive `telegram`
/// login over the wire `AuthApi`. Captures the node-wide app credentials + per-account store root;
/// `begin` reads the `mode` + `credential_ref` from the request params and connects a grammers-backed
/// [`LoginBackend`].
pub struct TelegramAuthFlowFactory {
    store_root: PathBuf,
    api_id: i32,
    api_hash: String,
}

impl TelegramAuthFlowFactory {
    /// A factory over the node-wide Telegram app credentials (`api_id`/`api_hash`) whose per-account
    /// session stores live under `store_root` (the same root `serve` uses).
    pub fn new(store_root: impl Into<PathBuf>, api_id: i32, api_hash: impl Into<String>) -> Self {
        Self {
            store_root: store_root.into(),
            api_id,
            api_hash: api_hash.into(),
        }
    }
}

#[async_trait]
impl AuthFlowFactory for TelegramAuthFlowFactory {
    fn family(&self) -> &str {
        FAMILY
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: FAMILY.to_string(),
            // The user flow is the multi-step OTP flow; `mode=bot` switches to the single-step token
            // form. One family advertises the richer kind for capability discovery.
            flow_kind: AuthFlowKind::PhoneOtp,
            display_name: "Telegram (user or bot)".to_string(),
            params_schema: vec![
                AuthParamField {
                    key: PARAM_MODE.to_string(),
                    label: "Account mode (user | bot)".to_string(),
                    required: true,
                    ..Default::default()
                },
                AuthParamField {
                    key: PARAM_CREDENTIAL_REF.to_string(),
                    label: "Account credential ref".to_string(),
                    required: true,
                    ..Default::default()
                },
            ],
        }
    }

    async fn begin(
        &self,
        params: &BTreeMap<String, String>,
        _redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        let mode = params
            .get(PARAM_MODE)
            .and_then(|m| AccountMode::parse(m))
            .ok_or_else(|| {
                ApiError::Other(format!(
                    "telegram auth: `{PARAM_MODE}` must be `user` or `bot`"
                ))
            })?;
        let credential_ref = required(params, PARAM_CREDENTIAL_REF)?;

        let backend = crate::client::connect_login(
            &self.store_root,
            self.api_id,
            &self.api_hash,
            &credential_ref,
            mode,
        )
        .await?;

        Ok(Box::new(TelegramPendingFlow::new(
            backend,
            mode,
            credential_ref,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A scripted mock backend: records the calls made and yields canned outcomes, so the flow's
    /// state machine can be driven end-to-end without grammers or the network.
    #[derive(Default)]
    struct MockBackend {
        password_required: bool,
        mode: Option<AccountMode>,
        token: Mutex<Option<String>>,
        calls: Mutex<Vec<String>>,
    }

    impl MockBackend {
        fn user(password_required: bool) -> Arc<Self> {
            Arc::new(Self {
                password_required,
                mode: Some(AccountMode::User),
                ..Self::default()
            })
        }
        fn bot() -> Arc<Self> {
            Arc::new(Self {
                mode: Some(AccountMode::Bot),
                ..Self::default()
            })
        }
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl LoginBackend for MockBackend {
        async fn request_code(&self, phone: &str) -> Result<(), ApiError> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("request_code:{phone}"));
            Ok(())
        }
        async fn submit_code(&self, code: &str) -> Result<CodeStep, ApiError> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("submit_code:{code}"));
            Ok(if self.password_required {
                CodeStep::PasswordRequired
            } else {
                CodeStep::Done
            })
        }
        async fn submit_password(&self, password: &str) -> Result<(), ApiError> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("submit_password:{password}"));
            Ok(())
        }
        async fn bot_sign_in(&self, token: &str) -> Result<(), ApiError> {
            *self.token.lock().unwrap() = Some(token.to_string());
            self.calls.lock().unwrap().push("bot_sign_in".into());
            Ok(())
        }
        async fn finish(&self) -> Result<LoginIdentity, ApiError> {
            self.calls.lock().unwrap().push("finish".into());
            let blob = match (self.mode.unwrap(), self.token.lock().unwrap().clone()) {
                (AccountMode::Bot, Some(t)) => crate::StoredSession::bot(t, 555).to_blob().unwrap(),
                _ => crate::StoredSession::user(555).to_blob().unwrap(),
            };
            Ok(LoginIdentity {
                account_id: 555,
                label: "tester".into(),
                credential_blob: blob,
            })
        }
    }

    fn fields(pairs: &[(&str, &str)]) -> AuthStepInput {
        AuthStepInput::Fields(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    fn completed(o: AuthStepOutcome) -> AuthOutcome {
        match o {
            AuthStepOutcome::Completed(a) => a,
            AuthStepOutcome::Challenge(c) => panic!("expected completion, got challenge {c:?}"),
        }
    }

    #[tokio::test]
    async fn user_flow_without_2fa_completes_after_phone_then_code() {
        let backend = MockBackend::user(false);
        let flow = TelegramPendingFlow::new(backend.clone(), AccountMode::User, "telegram/u");

        // Initial challenge collects the phone.
        assert!(matches!(
            flow.initial_challenge(),
            AuthChallenge::Form { .. }
        ));

        // phone -> code challenge
        let step1 = flow
            .step(fields(&[(PARAM_PHONE, "+15551234")]))
            .await
            .unwrap();
        match step1 {
            AuthStepOutcome::Challenge(AuthChallenge::Form { fields, .. }) => {
                assert_eq!(fields[0].key, PARAM_CODE);
            }
            _ => panic!("expected a code form"),
        }

        // code -> completed (no 2FA)
        let done = completed(flow.step(fields(&[(PARAM_CODE, "12345")])).await.unwrap());
        assert_eq!(done.transport_instance.as_str(), "telegram/555");
        assert_eq!(done.slot, CredentialSlotKind::Derived);
        assert_eq!(done.credential_ref, "telegram/u");
        assert_eq!(
            backend.calls(),
            vec!["request_code:+15551234", "submit_code:12345", "finish"]
        );
    }

    #[tokio::test]
    async fn user_flow_with_2fa_inserts_a_password_challenge() {
        let backend = MockBackend::user(true);
        let flow = TelegramPendingFlow::new(backend.clone(), AccountMode::User, "telegram/u");

        flow.step(fields(&[(PARAM_PHONE, "+1")])).await.unwrap();
        // code -> password challenge (2FA on)
        let step = flow.step(fields(&[(PARAM_CODE, "999")])).await.unwrap();
        match step {
            AuthStepOutcome::Challenge(AuthChallenge::Form { fields, .. }) => {
                assert_eq!(fields[0].key, PARAM_PASSWORD);
            }
            _ => panic!("expected a password form"),
        }
        // password -> completed
        let done = completed(
            flow.step(fields(&[(PARAM_PASSWORD, "hunter2")]))
                .await
                .unwrap(),
        );
        assert_eq!(done.account_label, "tester");
        assert_eq!(
            backend.calls(),
            vec![
                "request_code:+1",
                "submit_code:999",
                "submit_password:hunter2",
                "finish"
            ]
        );
    }

    #[tokio::test]
    async fn bot_flow_completes_in_one_step() {
        let backend = MockBackend::bot();
        let flow = TelegramPendingFlow::new(backend.clone(), AccountMode::Bot, "telegram/b");
        let done = completed(flow.step(fields(&[(PARAM_TOKEN, "42:ABC")])).await.unwrap());
        assert_eq!(done.transport_instance.as_str(), "telegram/555");
        // The persisted blob is a bot blob carrying the token.
        let stored = crate::StoredSession::from_blob(&done.credential_blob).unwrap();
        assert_eq!(stored.mode, AccountMode::Bot);
        assert_eq!(stored.bot_token.as_deref(), Some("42:ABC"));
        assert_eq!(backend.calls(), vec!["bot_sign_in", "finish"]);
    }

    #[tokio::test]
    async fn spent_flow_rejects_further_steps() {
        let backend = MockBackend::bot();
        let flow = TelegramPendingFlow::new(backend, AccountMode::Bot, "telegram/b");
        flow.step(fields(&[(PARAM_TOKEN, "42:ABC")])).await.unwrap();
        // `AuthStepOutcome` is not `Debug`, so match rather than `expect_err`.
        let err = match flow.step(fields(&[(PARAM_TOKEN, "42:ABC")])).await {
            Err(e) => e,
            Ok(_) => panic!("a completed flow must reject further steps"),
        };
        assert!(matches!(err, ApiError::Other(_)));
    }

    #[tokio::test]
    async fn missing_required_field_errors() {
        let backend = MockBackend::user(false);
        let flow = TelegramPendingFlow::new(backend, AccountMode::User, "telegram/u");
        let err = match flow.step(fields(&[("wrong", "x")])).await {
            Err(e) => e,
            Ok(_) => panic!("a missing phone field must error"),
        };
        assert!(matches!(err, ApiError::Other(_)));
    }
}
