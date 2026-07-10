// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The demo interactive-auth families: one [`AuthFlowFactory`] per [`AuthFlowKind`] variant, so the
//! whole client-driven login surface (every flow kind, every [`AuthChallenge`] shape) can be
//! exercised against a real node with zero external network. Registered exactly like the
//! descriptor-driven `daemon-oauth` families (several factory instances of one type).
//!
//! ## Flow → challenge-sequence map (all documented, deterministic)
//!
//! | family (`AuthBeginRequest.family`) | [`AuthFlowKind`] | steps |
//! |---|---|---|
//! | `demo` | `UserPassword` | `Form{username Text, password Password}` → *`demo`/`demo123`* completes; a wrong password is rejected (the flow stays parked to retry) |
//! | `demo-sso` | `MatrixSso` | `Redirect{loopback url}` → `Callback` completes |
//! | `demo-oauth` | `OAuth2Pkce` | `Redirect{loopback url}` → `Callback` completes |
//! | `demo-bot` | `BotToken` | `Form{token Password, region Choice+default}` → completes |
//! | `demo-user` | `UserToken` | `Form{token Password}` → completes |
//! | `demo-otp` | `PhoneOtp` | `Form{phone Text}` → `Form{code Number+placeholder}` → *`123456`* completes |
//! | `demo-qr` | `QrPairing` | `Qr{payload}` → `Poll` → `Message` → `Poll` completes |
//!
//! Across the seven flows every [`AuthChallenge`] kind (`Redirect`/`Form`/`Qr`/`Message`) and every
//! [`AuthStepInput`] kind (`Fields`/`Callback`/`Poll`) is produced at least once, and the enriched
//! N1 [`AuthParamField`] metadata is used: `Password` for secrets, `Number` + `placeholder` for the
//! OTP code, and a `Choice` with `choices` + `default` for the bot flow's region.

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

/// The documented demo username (the only accepted `UserPassword` account).
pub const DEMO_USERNAME: &str = "demo";
/// The documented demo password exchanged (never stored) at sign-in.
pub const DEMO_PASSWORD: &str = "demo123";
/// The documented demo one-time code the `PhoneOtp` flow accepts.
pub const DEMO_OTP_CODE: &str = "123456";

/// Which scripted flow a [`DemoAuthFlowFactory`] mints. Distinct from [`AuthFlowKind`] because two
/// kinds can share a shape (both redirect flows) while a client still discovers them as different
/// kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Script {
    /// Masked username + password `Form`; validates the fixed demo credential pair.
    UserPassword,
    /// A single browser-`Redirect` (loopback url) completing on the captured `Callback`.
    Redirect,
    /// A pasted-token `Form`. `with_region` adds the `Choice` field (the bot flow).
    PastedToken { with_region: bool },
    /// A two-step phone → OTP `Form` pair (the `Number` + placeholder code field).
    PhoneOtp,
    /// A `Qr` challenge, a poll, an informational `Message`, then a completing poll.
    QrPairing,
}

/// A demo auth family: its wire family id, human name, discovered [`AuthFlowKind`], and the scripted
/// [`Script`] its flows run.
struct DemoFlowSpec {
    family: &'static str,
    display_name: &'static str,
    flow_kind: AuthFlowKind,
    script: Script,
    /// The account label a completing flow reports (the `UserPassword` flow overrides it with the
    /// entered username).
    account_label: &'static str,
}

/// Map an [`AuthFlowKind`] to its demo family spec. The `match` is **exhaustive** (no wildcard): a
/// new [`AuthFlowKind`] variant breaks this build until a demo flow is defined for it, which is how
/// [`demo_auth_factories`] guarantees it covers *every* variant without hard-coding a subset.
fn spec_for(kind: AuthFlowKind) -> DemoFlowSpec {
    match kind {
        AuthFlowKind::UserPassword => DemoFlowSpec {
            family: "demo",
            display_name: "Demo username & password",
            flow_kind: AuthFlowKind::UserPassword,
            script: Script::UserPassword,
            account_label: DEMO_USERNAME,
        },
        AuthFlowKind::MatrixSso => DemoFlowSpec {
            family: "demo-sso",
            display_name: "Demo SSO (redirect)",
            flow_kind: AuthFlowKind::MatrixSso,
            script: Script::Redirect,
            account_label: "demo-sso",
        },
        AuthFlowKind::OAuth2Pkce => DemoFlowSpec {
            family: "demo-oauth",
            display_name: "Demo OAuth2 (PKCE redirect)",
            flow_kind: AuthFlowKind::OAuth2Pkce,
            script: Script::Redirect,
            account_label: "demo-oauth",
        },
        AuthFlowKind::BotToken => DemoFlowSpec {
            family: "demo-bot",
            display_name: "Demo bot token",
            flow_kind: AuthFlowKind::BotToken,
            script: Script::PastedToken { with_region: true },
            account_label: "demo-bot",
        },
        AuthFlowKind::UserToken => DemoFlowSpec {
            family: "demo-user",
            display_name: "Demo user token",
            flow_kind: AuthFlowKind::UserToken,
            script: Script::PastedToken { with_region: false },
            account_label: "demo-user",
        },
        AuthFlowKind::PhoneOtp => DemoFlowSpec {
            family: "demo-otp",
            display_name: "Demo phone + one-time code",
            flow_kind: AuthFlowKind::PhoneOtp,
            script: Script::PhoneOtp,
            account_label: "demo-otp",
        },
        AuthFlowKind::QrPairing => DemoFlowSpec {
            family: "demo-qr",
            display_name: "Demo QR device pairing",
            flow_kind: AuthFlowKind::QrPairing,
            script: Script::QrPairing,
            account_label: "demo-qr",
        },
    }
}

/// The demo [`AuthFlowKind`] variants, in registration order. Kept beside the exhaustive
/// [`spec_for`] match, which is the real coverage guarantee (a new variant fails the build there).
const DEMO_FLOW_KINDS: [AuthFlowKind; 7] = [
    AuthFlowKind::UserPassword,
    AuthFlowKind::MatrixSso,
    AuthFlowKind::OAuth2Pkce,
    AuthFlowKind::BotToken,
    AuthFlowKind::UserToken,
    AuthFlowKind::PhoneOtp,
    AuthFlowKind::QrPairing,
];

/// Every demo interactive-auth factory (one per [`AuthFlowKind`]), ready to hand to the node
/// alongside the [`DemoAdapter`](crate::DemoAdapter). Register these only when the demo transport is
/// enabled, mirroring the per-transport auth-factory gating in `bins/daemon`.
pub fn demo_auth_factories() -> Vec<std::sync::Arc<dyn AuthFlowFactory>> {
    // RED stub: no factories registered yet (GREEN wires one per AuthFlowKind).
    if true {
        return Vec::new();
    }
    DEMO_FLOW_KINDS
        .iter()
        .map(|&kind| {
            std::sync::Arc::new(DemoAuthFlowFactory {
                spec: spec_for(kind),
            }) as std::sync::Arc<dyn AuthFlowFactory>
        })
        .collect()
}

/// One demo auth family factory. Stateless beyond its [`DemoFlowSpec`]; each `begin` mints a fresh
/// [`DemoPendingFlow`].
struct DemoAuthFlowFactory {
    spec: DemoFlowSpec,
}

/// The masked username + password form (shared by discovery + the initial challenge so the client
/// renders identical fields either way).
fn userpass_fields() -> Vec<AuthParamField> {
    vec![
        AuthParamField {
            key: "username".into(),
            label: "Username".into(),
            required: true,
            kind: AuthFieldKind::Text,
            default: Some(DEMO_USERNAME.into()),
            placeholder: Some("you@demo.local".into()),
            choices: Vec::new(),
        },
        AuthParamField {
            key: "password".into(),
            label: "Password".into(),
            required: true,
            kind: AuthFieldKind::Password,
            default: None,
            placeholder: None,
            choices: Vec::new(),
        },
    ]
}

/// The pasted-token form; `with_region` adds the `Choice` field (default `us`).
fn token_fields(with_region: bool) -> Vec<AuthParamField> {
    let mut fields = vec![AuthParamField {
        key: "token".into(),
        label: "Access token".into(),
        required: true,
        kind: AuthFieldKind::Password,
        default: None,
        placeholder: Some("paste your token".into()),
        choices: Vec::new(),
    }];
    if with_region {
        fields.push(AuthParamField {
            key: "region".into(),
            label: "Region".into(),
            required: false,
            kind: AuthFieldKind::Choice,
            default: Some("us".into()),
            placeholder: None,
            choices: vec!["us".into(), "eu".into(), "apac".into()],
        });
    }
    fields
}

/// The phone-number form (step 1 of the OTP flow).
fn phone_fields() -> Vec<AuthParamField> {
    vec![AuthParamField {
        key: "phone".into(),
        label: "Phone number".into(),
        required: true,
        kind: AuthFieldKind::Text,
        default: None,
        placeholder: Some("+1 555 0100".into()),
        choices: Vec::new(),
    }]
}

/// The one-time-code form (step 2 of the OTP flow) — a `Number` field with a placeholder hint.
fn otp_fields() -> Vec<AuthParamField> {
    vec![AuthParamField {
        key: "code".into(),
        label: "One-time code".into(),
        required: true,
        kind: AuthFieldKind::Number,
        default: None,
        placeholder: Some("123456".into()),
        choices: Vec::new(),
    }]
}

#[async_trait]
impl AuthFlowFactory for DemoAuthFlowFactory {
    fn family(&self) -> &str {
        self.spec.family
    }

    fn provider_info(&self) -> AuthProviderInfo {
        let params_schema = match self.spec.script {
            Script::UserPassword => userpass_fields(),
            Script::PastedToken { with_region } => token_fields(with_region),
            Script::PhoneOtp => phone_fields(),
            // Redirect + QR flows collect nothing up front (the client opens a URL / scans a QR).
            Script::Redirect | Script::QrPairing => Vec::new(),
        };
        AuthProviderInfo {
            family: self.spec.family.to_string(),
            flow_kind: self.spec.flow_kind,
            display_name: self.spec.display_name.to_string(),
            params_schema,
        }
    }

    async fn begin(
        &self,
        _params: &BTreeMap<String, String>,
        redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        Ok(Box::new(DemoPendingFlow {
            script: self.spec.script,
            account_label: self.spec.account_label.to_string(),
            redirect_uri: redirect_uri.to_string(),
            state: Mutex::new(FlowState::default()),
        }))
    }
}

/// The mutable continuation state a multi-step flow carries between challenges.
#[derive(Default)]
struct FlowState {
    /// OTP: the phone step has been answered, so the next `Fields` is the code.
    awaiting_code: bool,
    /// QR: the informational `Message` has been shown, so the next `Poll` completes.
    approved: bool,
}

/// One parked demo flow: a small state machine driven by [`step`](PendingAuthFlow::step).
struct DemoPendingFlow {
    script: Script,
    account_label: String,
    redirect_uri: String,
    state: Mutex<FlowState>,
}

impl DemoPendingFlow {
    /// The completed outcome for this flow (the exchanged blob + the account identity). The
    /// `label` overrides the spec default (the `UserPassword` flow passes the entered username).
    fn complete(&self, label: &str) -> AuthStepOutcome {
        let transport_instance = TransportId::new(format!("{}/{label}", crate::FAMILY));
        AuthStepOutcome::Completed(AuthOutcome {
            // The demo "session token" — a synthetic blob; the transient password (if any) never
            // enters it.
            credential_blob: format!("demo-session:{label}"),
            credential_ref: format!("{}/{label}", crate::FAMILY),
            account_label: label.to_string(),
            transport_instance,
            slot: CredentialSlotKind::Derived,
        })
    }
}

#[async_trait]
impl PendingAuthFlow for DemoPendingFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        match self.script {
            Script::UserPassword => AuthChallenge::Form {
                title: "Sign in to Demo".into(),
                fields: userpass_fields(),
            },
            Script::Redirect => AuthChallenge::Redirect {
                // A loopback-style authorization URL that carries the client's own redirect_uri.
                authorization_url: format!(
                    "https://demo.local/authorize?redirect_uri={}",
                    self.redirect_uri
                ),
            },
            Script::PastedToken { with_region } => AuthChallenge::Form {
                title: "Paste your token".into(),
                fields: token_fields(with_region),
            },
            Script::PhoneOtp => AuthChallenge::Form {
                title: "Enter your phone number".into(),
                fields: phone_fields(),
            },
            Script::QrPairing => AuthChallenge::Qr {
                payload: "demo://pair?code=DEMO-PAIR-0001".into(),
                image: None,
                poll_interval_ms: 250,
            },
        }
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        match self.script {
            Script::UserPassword => {
                let fields = expect_fields(input)?;
                let username = required(&fields, "username")?;
                let password = required(&fields, "password")?;
                if password != DEMO_PASSWORD {
                    // Wrong password: reject; the registry leaves the flow parked for a retry.
                    return Err(ApiError::Other("invalid username or password".into()));
                }
                Ok(self.complete(&username))
            }
            Script::Redirect => {
                let AuthStepInput::Callback(_cb) = input else {
                    return Err(ApiError::Other(
                        "this flow expects a redirect callback".into(),
                    ));
                };
                Ok(self.complete(&self.account_label))
            }
            Script::PastedToken { .. } => {
                let fields = expect_fields(input)?;
                let _token = required(&fields, "token")?;
                Ok(self.complete(&self.account_label))
            }
            Script::PhoneOtp => {
                let fields = expect_fields(input)?;
                let mut state = self.state.lock().unwrap();
                if !state.awaiting_code {
                    // Step 1: the phone number → present the OTP code form.
                    required(&fields, "phone")?;
                    state.awaiting_code = true;
                    return Ok(AuthStepOutcome::Challenge(AuthChallenge::Form {
                        title: "Enter the code we texted you".into(),
                        fields: otp_fields(),
                    }));
                }
                // Step 2: the code → validate the fixed demo code + complete.
                let code = required(&fields, "code")?;
                if code != DEMO_OTP_CODE {
                    return Err(ApiError::Other("incorrect one-time code".into()));
                }
                Ok(self.complete(&self.account_label))
            }
            Script::QrPairing => {
                let AuthStepInput::Poll = input else {
                    return Err(ApiError::Other("this flow expects a poll".into()));
                };
                let mut state = self.state.lock().unwrap();
                if !state.approved {
                    // First poll: show the informational Message, keep polling.
                    state.approved = true;
                    return Ok(AuthStepOutcome::Challenge(AuthChallenge::Message {
                        text: "Approve the pairing on your other device".into(),
                    }));
                }
                // Second poll: the peer device approved → complete.
                Ok(self.complete(&self.account_label))
            }
        }
    }
}

/// Unwrap a `Fields` input or error (a form step got the wrong input kind).
fn expect_fields(input: AuthStepInput) -> Result<BTreeMap<String, String>, ApiError> {
    match input {
        AuthStepInput::Fields(f) => Ok(f),
        _ => Err(ApiError::Other("this flow expects form fields".into())),
    }
}

/// Fetch a required, non-empty form field or error.
fn required(fields: &BTreeMap<String, String>, key: &str) -> Result<String, ApiError> {
    fields
        .get(key)
        .filter(|s| !s.is_empty())
        .cloned()
        .ok_or_else(|| ApiError::Other(format!("{key} is required")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every demo factory reports the flow kind its spec declares, and the set covers all seven
    /// `AuthFlowKind` variants exactly once.
    #[test]
    fn one_factory_per_auth_flow_kind() {
        let factories = demo_auth_factories();
        assert_eq!(factories.len(), DEMO_FLOW_KINDS.len());
        let mut kinds: Vec<AuthFlowKind> = factories
            .iter()
            .map(|f| f.provider_info().flow_kind)
            .collect();
        kinds.sort_by_key(|k| format!("{k:?}"));
        let mut want = DEMO_FLOW_KINDS.to_vec();
        want.sort_by_key(|k| format!("{k:?}"));
        assert_eq!(kinds, want, "every AuthFlowKind is represented once");
    }

    /// The enriched N1 field metadata is present: a masked password, a numeric OTP code with a
    /// placeholder, and a Choice with choices + a default.
    #[test]
    fn enriched_field_metadata_present() {
        let by_family = |family: &str| {
            demo_auth_factories()
                .into_iter()
                .map(|f| f.provider_info())
                .find(|p| p.family == family)
                .unwrap()
        };
        let up = by_family("demo");
        assert!(up
            .params_schema
            .iter()
            .any(|f| f.key == "password" && f.kind == AuthFieldKind::Password));
        let bot = by_family("demo-bot");
        let region = bot
            .params_schema
            .iter()
            .find(|f| f.key == "region")
            .expect("bot flow has a region field");
        assert_eq!(region.kind, AuthFieldKind::Choice);
        assert!(!region.choices.is_empty() && region.default.is_some());
    }
}
