// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Matrix SSO as a client-driven interactive-auth family (`daemon-interactive-auth-spec`).
//!
//! Matrix SSO is a browser-redirect flow: the homeserver issues an authorization URL, the user logs
//! in there, and the homeserver redirects to a caller-chosen `redirect_uri` carrying a single-use
//! `loginToken`. The daemon is headless, so it cannot host the browser or the redirect — a decoupled
//! client does. This module exposes the SSO flow as two primitives, [`sso_begin`] / [`sso_complete`],
//! split exactly at the browser hop, and wires them into the host's family-agnostic auth seam via
//! [`MatrixAuthFlowFactory`] (begin) + [`MatrixPendingFlow`] (complete).
//!
//! The on-disk client (state + E2EE crypto store) is created at `begin` keyed by the account's
//! `credential_ref`, and the *same* `Client` is held across the browser hop and consumed at
//! `complete` — so the device the crypto store is created for is the device the session is minted on,
//! and `serve` (which re-opens `account_store_dir(store_root, credential_ref)`) restores it (spec
//! §6.3 device-id constraint). The CLI `matrix login` rebases onto the same two primitives over a
//! local loopback redirect, so the operator path and the GUI path share one implementation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use matrix_sdk::utils::UrlOrQuery;
use matrix_sdk::Client;

use async_trait::async_trait;
use daemon_api::{ApiError, AuthFlowKind, AuthParamField, AuthProviderInfo};
use daemon_host::{AuthFlowFactory, AuthOutcome, PendingAuthFlow};
use daemon_protocol::TransportId;

use crate::account::{account_store_dir, build_client, StoredSession};
use crate::FAMILY;

/// The `auth_begin` param naming the homeserver to dial (required).
pub const PARAM_HOMESERVER: &str = "homeserver";
/// The `auth_begin` param naming the account's stable credential/store key (required). It keys both
/// the on-disk store dir (device-id stability) and where the session blob lands.
pub const PARAM_CREDENTIAL_REF: &str = "credential_ref";
/// The `auth_begin` param naming an optional identity-provider id (multi-IdP homeservers).
pub const PARAM_IDP_ID: &str = "idp_id";

/// The default `initial_device_display_name` minted on login.
const DEVICE_DISPLAY_NAME: &str = "daemon";

/// An SSO flow paused at the browser hop: the on-disk client built at `begin` (held across the hop and
/// consumed at `complete`), the homeserver it dials, the account's stable `credential_ref`, and the
/// authorization URL the caller opens.
pub struct SsoSession {
    client: Client,
    homeserver: String,
    credential_ref: String,
    /// The authorization URL the caller opens in a browser.
    pub authorization_url: String,
}

/// The product of a completed Matrix login: the opaque session blob to persist, the stable
/// `credential_ref` it is keyed by, the resolved bare user id, and the instance-qualified transport id.
pub struct MatrixLogin {
    /// The opaque [`StoredSession`] blob (homeserver + matrix-sdk session/tokens).
    pub credential_blob: String,
    /// The stable credential/store key this account is keyed by.
    pub credential_ref: String,
    /// The resolved bare user id (`@bot:hs.org`).
    pub user_id: String,
    /// The instance-qualified transport id (`matrix/@bot:hs.org`).
    pub transport_instance: TransportId,
}

/// Begin SSO: build the on-disk client for `homeserver` (keyed by `credential_ref`) and mint the
/// authorization URL pointing at the caller-owned `redirect_uri`. The returned [`SsoSession`] is
/// driven to [`sso_complete`] after the caller captures the redirect.
pub async fn sso_begin(
    store_root: &Path,
    homeserver: &str,
    credential_ref: &str,
    redirect_uri: &str,
    idp_id: Option<&str>,
) -> Result<SsoSession> {
    let store_dir = account_store_dir(store_root, credential_ref);
    let client = build_client(homeserver, &store_dir).await?;
    let authorization_url = client
        .matrix_auth()
        .get_sso_login_url(redirect_uri, idp_id)
        .await
        .map_err(|e| anyhow!("building matrix SSO login url: {e}"))?;
    Ok(SsoSession {
        client,
        homeserver: homeserver.to_string(),
        credential_ref: credential_ref.to_string(),
        authorization_url,
    })
}

/// Complete SSO from a captured `callback` (the full redirect URL or just its query string), minting
/// the session on the *same* device the `begin` client created and producing the blob to persist.
pub async fn sso_complete(session: SsoSession, callback: &str) -> Result<MatrixLogin> {
    // matrix-sdk's `login_with_sso_callback` reads the `loginToken` from a query string; accept either
    // a full redirect URL or a bare query and hand it the query part.
    let query = callback
        .split_once('?')
        .map(|(_, q)| q)
        .unwrap_or(callback)
        .to_string();

    let SsoSession {
        client,
        homeserver,
        credential_ref,
        ..
    } = session;

    client
        .matrix_auth()
        .login_with_sso_callback(UrlOrQuery::Query(query))
        .map_err(|e| anyhow!("matrix SSO callback invalid: {e}"))?
        .initial_device_display_name(DEVICE_DISPLAY_NAME)
        .await
        .map_err(|e| anyhow!("matrix SSO token login failed: {e}"))?;

    let matrix_session = client
        .matrix_auth()
        .session()
        .ok_or_else(|| anyhow!("no session present after SSO login"))?;
    let user_id = client
        .user_id()
        .ok_or_else(|| anyhow!("client has no user id after login"))?
        .to_string();

    let credential_blob = StoredSession {
        homeserver,
        session: matrix_session,
    }
    .to_blob()
    .context("serializing matrix session blob")?;

    let transport_instance = TransportId::new(format!("{FAMILY}/{user_id}"));
    Ok(MatrixLogin {
        credential_blob,
        credential_ref,
        user_id,
        transport_instance,
    })
}

/// The Matrix interactive-auth factory: registered with the node so a client can drive `matrix` SSO
/// over the wire `AuthApi`. Captures the per-account store root (`<data_dir>/<matrix.store_root>`);
/// `begin` reads the homeserver / credential-ref / optional IdP from the request params.
pub struct MatrixAuthFlowFactory {
    store_root: PathBuf,
}

impl MatrixAuthFlowFactory {
    /// A factory whose per-account on-disk stores live under `store_root` (the same root `serve` uses).
    pub fn new(store_root: impl Into<PathBuf>) -> Self {
        Self {
            store_root: store_root.into(),
        }
    }
}

#[async_trait]
impl AuthFlowFactory for MatrixAuthFlowFactory {
    fn family(&self) -> &str {
        FAMILY
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: FAMILY.to_string(),
            flow_kind: AuthFlowKind::MatrixSso,
            display_name: "Matrix (SSO)".to_string(),
            params_schema: vec![
                AuthParamField {
                    key: PARAM_HOMESERVER.to_string(),
                    label: "Homeserver URL".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: PARAM_CREDENTIAL_REF.to_string(),
                    label: "Account credential ref".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: PARAM_IDP_ID.to_string(),
                    label: "Identity provider id (optional)".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn begin(
        &self,
        params: &BTreeMap<String, String>,
        redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        let homeserver = params
            .get(PARAM_HOMESERVER)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ApiError::Other(format!("matrix auth: missing `{PARAM_HOMESERVER}`")))?;
        let credential_ref = params
            .get(PARAM_CREDENTIAL_REF)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                ApiError::Other(format!("matrix auth: missing `{PARAM_CREDENTIAL_REF}`"))
            })?;
        let idp_id = params.get(PARAM_IDP_ID).map(String::as_str);

        let session = sso_begin(
            &self.store_root,
            homeserver,
            credential_ref,
            redirect_uri,
            idp_id,
        )
        .await
        .map_err(|e| ApiError::Other(format!("matrix SSO begin: {e}")))?;

        Ok(Box::new(MatrixPendingFlow { session }))
    }
}

/// A parked Matrix SSO flow: holds the [`SsoSession`] across the browser hop and finishes it.
struct MatrixPendingFlow {
    session: SsoSession,
}

#[async_trait]
impl PendingAuthFlow for MatrixPendingFlow {
    fn authorization_url(&self) -> &str {
        &self.session.authorization_url
    }

    fn flow_kind(&self) -> AuthFlowKind {
        AuthFlowKind::MatrixSso
    }

    async fn complete(self: Box<Self>, callback: &str) -> Result<AuthOutcome, ApiError> {
        let login = sso_complete(self.session, callback)
            .await
            .map_err(|e| ApiError::Other(format!("matrix SSO complete: {e}")))?;
        Ok(AuthOutcome {
            credential_blob: login.credential_blob,
            credential_ref: login.credential_ref,
            account_label: login.user_id,
            transport_instance: login.transport_instance,
            slot: daemon_host::CredentialSlotKind::Derived,
        })
    }
}
