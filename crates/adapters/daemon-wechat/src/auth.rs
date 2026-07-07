// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! WeChat QR-pairing login as a client-driven interactive-auth family (`daemon-interactive-auth-spec`).
//!
//! WeChat iLink has no browser-redirect or token-paste path: a bot binds to a phone by displaying a
//! QR code that the user scans in the WeChat app, then confirms (sometimes with a pairing code shown
//! on the phone). This maps onto the [`AuthChallenge::Qr`] + repeated [`AuthStepInput::Poll`] shape:
//! [`begin`](AuthFlowFactory::begin) mints a QR ([`ILinkClient::get_qr_code`]) and parks the poll
//! handle; each [`step`](PendingAuthFlow::step) polls the scan status ([`ILinkClient::poll_qr_status`])
//! until the server reports `confirmed`, at which point the minted [`StoredSession`] is persisted as
//! the account's credential blob (`slot = Derived`). The pairing-code sub-step surfaces as an
//! [`AuthChallenge::Form`] (`verify_code`) — the digits WeChat shows on the phone.
//!
//! `step` takes `&self`; the mutable poll state (the possibly-IDC-redirected base URL and a submitted
//! pairing code) lives behind a `std::sync::Mutex` that is never held across an `await` (values are
//! taken out under the lock, the network call runs unlocked, then the lock is re-taken to update).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{
    ApiError, AuthChallenge, AuthFlowKind, AuthParamField, AuthProviderInfo, AuthStepInput,
};
use daemon_host::{
    AuthFlowFactory, AuthOutcome, AuthStepOutcome, CredentialSlotKind, PendingAuthFlow,
};
use wechatbot::protocol::{ILinkClient, QrStatusResponse};

use crate::account::WECHAT_QR_BASE_URL;
use crate::mapping::StoredSession;
use crate::FAMILY;

/// The `auth_begin` param naming the account's stable credential/store key (required). The minted
/// session blob is stored under it, and it is the account handle the profile binds to.
pub const PARAM_CREDENTIAL_REF: &str = "credential_ref";
/// The optional `auth_begin` param carrying the UA-style `bot_agent` the iLink calls are stamped with.
pub const PARAM_BOT_AGENT: &str = "bot_agent";
/// The [`AuthChallenge::Form`] field key carrying the pairing code shown in WeChat on the phone.
pub const PARAM_VERIFY_CODE: &str = "verify_code";

/// How often the client should re-`Poll` the QR scan status (milliseconds) — the SDK's own cadence.
const DEFAULT_POLL_INTERVAL_MS: u64 = 2000;

/// The decision made from one QR status poll — the pure core of the flow, unit-tested against
/// hand-built [`QrStatusResponse`]s (no live network).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PollDecision {
    /// Not scanned yet — keep showing the QR and poll again.
    KeepWaiting,
    /// Scanned but not confirmed — prompt the user to confirm on their phone, then poll again.
    Scanned,
    /// The server wants the pairing code shown on the phone — collect it via a form.
    NeedVerifyCode,
    /// Polling must move to a new data-center host — switch and keep polling.
    Redirect(String),
    /// Login confirmed — the session to persist.
    Completed(StoredSession),
    /// Terminal failure (expired / blocked / already bound elsewhere).
    Failed(String),
}

/// Classify one QR status response against the current poll `base_url` (the fallback for a session
/// whose `baseurl` the server omitted). Pure; the network poll happens in [`PendingAuthFlow::step`].
pub(crate) fn classify_status(status: &QrStatusResponse, base_url: &str) -> PollDecision {
    match status.status.as_str() {
        "confirmed" => match &status.bot_token {
            Some(token) if !token.is_empty() => PollDecision::Completed(StoredSession {
                token: token.clone(),
                base_url: status
                    .baseurl
                    .clone()
                    .unwrap_or_else(|| base_url.to_string()),
                account_id: status.ilink_bot_id.clone().unwrap_or_default(),
                user_id: status.ilink_user_id.clone().unwrap_or_default(),
            }),
            _ => PollDecision::Failed("server reported confirmed but returned no bot_token".into()),
        },
        "scaned" => PollDecision::Scanned,
        "need_verifycode" => PollDecision::NeedVerifyCode,
        "scaned_but_redirect" => match &status.redirect_host {
            Some(host) if !host.is_empty() => PollDecision::Redirect(format!("https://{host}")),
            _ => PollDecision::KeepWaiting,
        },
        "expired" => PollDecision::Failed("QR code expired".into()),
        "verify_code_blocked" => {
            PollDecision::Failed("pairing code blocked after repeated mismatches".into())
        }
        "binded_redirect" => {
            PollDecision::Failed("this bot is already bound to another client".into())
        }
        // Unknown / still-waiting statuses: keep the QR up and poll again.
        _ => PollDecision::KeepWaiting,
    }
}

/// The WeChat interactive-auth factory: registered with the node so a client can drive `wechat` QR
/// pairing over the wire `AuthApi`. Stateless — each `begin` mints its own iLink client + QR.
pub struct WeChatAuthFlowFactory;

impl WeChatAuthFlowFactory {
    /// A new factory (no captured state; each flow is self-contained).
    pub fn new() -> Self {
        Self
    }
}

impl Default for WeChatAuthFlowFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AuthFlowFactory for WeChatAuthFlowFactory {
    fn family(&self) -> &str {
        FAMILY
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: FAMILY.to_string(),
            flow_kind: AuthFlowKind::QrPairing,
            display_name: "WeChat (QR pairing)".to_string(),
            params_schema: vec![
                AuthParamField {
                    key: PARAM_CREDENTIAL_REF.to_string(),
                    label: "Account credential ref".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: PARAM_BOT_AGENT.to_string(),
                    label: "Bot agent (optional UA string)".to_string(),
                    required: false,
                },
            ],
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
                ApiError::Other(format!("wechat auth: missing `{PARAM_CREDENTIAL_REF}`"))
            })?
            .clone();
        let bot_agent = params.get(PARAM_BOT_AGENT).map(String::as_str);

        let client = Arc::new(ILinkClient::with_bot_agent(bot_agent));
        let qr = client
            .get_qr_code(WECHAT_QR_BASE_URL, &[])
            .await
            .map_err(|e| ApiError::Other(format!("wechat get_qr_code: {e}")))?;

        Ok(Box::new(WeChatPendingFlow {
            client,
            qrcode: qr.qrcode,
            payload: qr.qrcode_img_content,
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            credential_ref,
            state: Mutex::new(PollState {
                base_url: WECHAT_QR_BASE_URL.to_string(),
                verify_code: None,
            }),
        }))
    }
}

/// The mutable, across-step poll state. Behind a `Mutex` so [`PendingAuthFlow::step`] (which only has
/// `&self`) can advance it; the guard is never held across the network `await`.
struct PollState {
    /// The current poll host (updated on an IDC `scaned_but_redirect`).
    base_url: String,
    /// A pairing code submitted via the `verify_code` form, carried onto subsequent polls.
    verify_code: Option<String>,
}

/// A parked WeChat QR-pairing flow: the iLink client + poll handle held across the scan, plus the QR
/// payload the client renders and the credential ref the minted session is keyed by.
struct WeChatPendingFlow {
    client: Arc<ILinkClient>,
    /// The opaque QR poll handle (`get_qrcode_status?qrcode=...`).
    qrcode: String,
    /// The QR content the client renders for the user to scan (`qrcode_img_content`).
    payload: String,
    /// The client's re-poll cadence.
    poll_interval_ms: u64,
    /// The stable credential/store key the minted session is persisted under.
    credential_ref: String,
    /// The across-step mutable poll state.
    state: Mutex<PollState>,
}

impl WeChatPendingFlow {
    /// The QR challenge presented while the login is still pending (rendered + re-polled).
    fn qr_challenge(&self) -> AuthChallenge {
        AuthChallenge::Qr {
            payload: self.payload.clone(),
            image: None,
            poll_interval_ms: self.poll_interval_ms,
        }
    }
}

#[async_trait]
impl PendingAuthFlow for WeChatPendingFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        self.qr_challenge()
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        // A `Fields` step submits the pairing code; a `Poll` step just re-checks. A redirect callback
        // has no meaning for QR pairing.
        let submitted =
            match input {
                AuthStepInput::Poll => None,
                AuthStepInput::Fields(fields) => Some(
                    fields
                        .get(PARAM_VERIFY_CODE)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            ApiError::Other(format!(
                                "wechat auth: verification form requires `{PARAM_VERIFY_CODE}`"
                            ))
                        })?
                        .clone(),
                ),
                AuthStepInput::Callback(_) => return Err(ApiError::Other(
                    "wechat QR pairing expects Poll (or the verification code), not a redirect \
                     callback"
                        .into(),
                )),
            };

        // Read the poll host + effective pairing code under the lock, then poll unlocked.
        let (base_url, code) = {
            let state = self.state.lock().unwrap();
            let code = submitted.clone().or_else(|| state.verify_code.clone());
            (state.base_url.clone(), code)
        };

        let status = self
            .client
            .poll_qr_status(&base_url, &self.qrcode, code.as_deref())
            .await
            .map_err(|e| ApiError::Other(format!("wechat poll_qr_status: {e}")))?;

        // Persist a freshly submitted pairing code so subsequent polls carry it.
        if let Some(code) = &submitted {
            self.state.lock().unwrap().verify_code = Some(code.clone());
        }

        match classify_status(&status, &base_url) {
            PollDecision::Completed(session) => {
                let credential_blob = session
                    .to_blob()
                    .map_err(|e| ApiError::Other(format!("wechat session blob: {e}")))?;
                let account_label = session.user_id.clone();
                let transport_instance = session.transport_instance();
                Ok(AuthStepOutcome::Completed(AuthOutcome {
                    credential_blob,
                    credential_ref: self.credential_ref.clone(),
                    account_label,
                    transport_instance,
                    slot: CredentialSlotKind::Derived,
                }))
            }
            PollDecision::KeepWaiting => Ok(AuthStepOutcome::Challenge(self.qr_challenge())),
            PollDecision::Scanned => Ok(AuthStepOutcome::Challenge(AuthChallenge::Message {
                text: "Scan detected — confirm the login in WeChat on your phone.".to_string(),
            })),
            PollDecision::NeedVerifyCode => Ok(AuthStepOutcome::Challenge(AuthChallenge::Form {
                title: "Enter the pairing code shown in WeChat on your phone".to_string(),
                fields: vec![AuthParamField {
                    key: PARAM_VERIFY_CODE.to_string(),
                    label: "Pairing code".to_string(),
                    required: true,
                }],
            })),
            PollDecision::Redirect(host) => {
                self.state.lock().unwrap().base_url = host;
                Ok(AuthStepOutcome::Challenge(AuthChallenge::Message {
                    text: "Redirecting to your data center — keep this dialog open.".to_string(),
                }))
            }
            PollDecision::Failed(msg) => Err(ApiError::Other(format!("wechat QR login: {msg}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(state: &str) -> QrStatusResponse {
        QrStatusResponse {
            status: state.to_string(),
            bot_token: None,
            ilink_bot_id: None,
            ilink_user_id: None,
            baseurl: None,
            redirect_host: None,
        }
    }

    #[test]
    fn waiting_and_scanned_keep_the_flow_open() {
        assert_eq!(
            classify_status(&status("new"), WECHAT_QR_BASE_URL),
            PollDecision::KeepWaiting
        );
        assert_eq!(
            classify_status(&status("waiting"), WECHAT_QR_BASE_URL),
            PollDecision::KeepWaiting
        );
        assert_eq!(
            classify_status(&status("scaned"), WECHAT_QR_BASE_URL),
            PollDecision::Scanned
        );
    }

    #[test]
    fn need_verifycode_and_terminal_failures() {
        assert_eq!(
            classify_status(&status("need_verifycode"), WECHAT_QR_BASE_URL),
            PollDecision::NeedVerifyCode
        );
        assert!(matches!(
            classify_status(&status("expired"), WECHAT_QR_BASE_URL),
            PollDecision::Failed(_)
        ));
        assert!(matches!(
            classify_status(&status("verify_code_blocked"), WECHAT_QR_BASE_URL),
            PollDecision::Failed(_)
        ));
        assert!(matches!(
            classify_status(&status("binded_redirect"), WECHAT_QR_BASE_URL),
            PollDecision::Failed(_)
        ));
    }

    #[test]
    fn redirect_switches_the_poll_host() {
        let mut s = status("scaned_but_redirect");
        s.redirect_host = Some("idc-sh.weixin.qq.com".to_string());
        assert_eq!(
            classify_status(&s, WECHAT_QR_BASE_URL),
            PollDecision::Redirect("https://idc-sh.weixin.qq.com".to_string())
        );
        // A redirect status without a host cannot switch — stay waiting rather than erroring.
        assert_eq!(
            classify_status(&status("scaned_but_redirect"), WECHAT_QR_BASE_URL),
            PollDecision::KeepWaiting
        );
    }

    #[test]
    fn confirmed_mints_a_session_and_defaults_the_base_url() {
        let mut s = status("confirmed");
        s.bot_token = Some("tok-xyz".to_string());
        s.ilink_bot_id = Some("bot-1".to_string());
        s.ilink_user_id = Some("user-1".to_string());
        // No `baseurl` in the response: falls back to the current poll host.
        match classify_status(&s, "https://idc-sh.weixin.qq.com") {
            PollDecision::Completed(session) => {
                assert_eq!(session.token, "tok-xyz");
                assert_eq!(session.base_url, "https://idc-sh.weixin.qq.com");
                assert_eq!(session.user_id, "user-1");
                assert_eq!(session.transport_instance().as_str(), "wechat/user-1");
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn confirmed_without_token_is_a_failure() {
        assert!(matches!(
            classify_status(&status("confirmed"), WECHAT_QR_BASE_URL),
            PollDecision::Failed(_)
        ));
    }

    #[test]
    fn provider_info_advertises_qr_pairing() {
        let info = WeChatAuthFlowFactory::new().provider_info();
        assert_eq!(info.family, FAMILY);
        assert_eq!(info.flow_kind, AuthFlowKind::QrPairing);
        assert!(info
            .params_schema
            .iter()
            .any(|f| f.key == PARAM_CREDENTIAL_REF && f.required));
    }
}
