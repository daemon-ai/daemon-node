// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The admin access-control sub-surface ([`AccessControlApi`]) over the node's [`AuthStore`].
//!
//! Every op except [`who_am_i`](AccessControlApi::who_am_i) requires `access_admin`: the per-request
//! capability gate ([`crate::authz`]) already enforces this at the transport, and each handler
//! **re-checks** it (defense in depth) so the in-process / FFI path — and any future caller — cannot
//! reach an admin mutation without the capability. `who_am_i` is allowed for any authenticated
//! principal (it returns the caller's own [`PrincipalView`]). The reserved `resource_grant_*` ops
//! keep the trait's default [`ApiError::Unsupported`].
//!
//! Mutations are recorded onto the verifiable `node-auth` journal stream via the shared
//! [`AuthAudit`](crate::auth_audit::AuthAudit), audit-on-success, and **never** carry credential
//! material. Last-admin lockout is enforced atomically by the store's guarded mutations
//! ([`AuthStore::set_disabled_guarded`] / [`AuthStore::set_roles_guarded`]); `user_set_password`
//! re-derives SCRAM (inside `set_password`) and revokes the user's sessions so a reset forces re-login.

use super::*;

use crate::authn::principal_view;
use crate::request_context::current_principal;
use daemon_api::{AccessControlApi, AccessUser, PrincipalView, RoleInfo};
use daemon_auth::{AuthStore, Capability, Principal, Role, UserRecord};

/// Map a `daemon-auth` store error onto the wire [`ApiError`]. The last-admin guard becomes a
/// `Forbidden` (the caller is allowed to administer, but this *specific* change is refused); an
/// unknown row becomes `Other` (there is no dedicated "unknown user" wire variant).
fn auth_err(e: daemon_auth::Error) -> ApiError {
    match e {
        daemon_auth::Error::LastAdmin => {
            ApiError::Forbidden("refusing to remove the last administrator".into())
        }
        daemon_auth::Error::NotFound => ApiError::Other("unknown user".into()),
        daemon_auth::Error::Disabled => ApiError::Other("account disabled".into()),
        other => ApiError::Other(format!("access control: {other}")),
    }
}

/// The stable snake_case wire name of a capability (its serde representation — the same source of
/// truth the `PrincipalView` capability list uses).
fn cap_name(cap: Capability) -> String {
    serde_json::to_value(cap)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// Parse wire role names into [`Role`]s, rejecting any unknown string (fail-closed: never silently
/// drop an unrecognized role to "no role").
fn parse_roles(roles: &[String]) -> Result<Vec<Role>, ApiError> {
    roles
        .iter()
        .map(|r| Role::from_wire(r).ok_or_else(|| ApiError::Other(format!("unknown role: {r}"))))
        .collect()
}

/// Defense-in-depth `access_admin` check (the gate already enforced it on the network path; this
/// guards the in-process/FFI path and any future caller).
fn require_admin() -> Result<(), ApiError> {
    match current_principal() {
        None => Err(ApiError::Unauthenticated(
            "no authenticated principal bound to this request".into(),
        )),
        Some(p) if p.has(Capability::AccessAdmin) => Ok(()),
        Some(_) => Err(ApiError::Forbidden(
            "operation requires capability AccessAdmin".into(),
        )),
    }
}

impl NodeApiImpl {
    fn auth_store(&self) -> Result<&Arc<AuthStore>, ApiError> {
        self.auth_store
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("access control not available".into()))
    }

    /// Tear down `user_id`'s live mux connections (Cluster F, Part A): bump the principal's
    /// revocation epoch so a connection holding the old epoch is closed and its live stream pumps
    /// end. Called **after** the store mutation has committed (its `conn` lock already released), so
    /// the epoch bump never runs under the store lock. A no-op when no revocation registry is wired.
    fn revoke_principal(&self, user_id: &str) {
        if let Some(revocations) = &self.revocations {
            revocations.revoke(user_id);
        }
    }

    /// Project a store [`UserRecord`] onto the wire [`AccessUser`] (resolving its roles). No secrets.
    fn access_user(store: &AuthStore, rec: UserRecord) -> Result<AccessUser, ApiError> {
        let roles = store
            .roles_of(&rec.id)
            .map_err(auth_err)?
            .iter()
            .map(|r| r.as_str().to_string())
            .collect();
        Ok(AccessUser {
            user_id: rec.id,
            username: rec.username,
            disabled: rec.disabled,
            created_at: rec.created_at,
            roles,
        })
    }
}

#[async_trait]
impl AccessControlApi for NodeApiImpl {
    async fn user_create(
        &self,
        username: String,
        password: String,
        roles: Vec<String>,
    ) -> Result<AccessUser, ApiError> {
        require_admin()?;
        let store = self.auth_store()?;
        let parsed = parse_roles(&roles)?;
        let rec = store
            .create_user(&username, &password, &parsed)
            .map_err(auth_err)?;
        let user = Self::access_user(store, rec)?;
        if let Some(a) = &self.auth_audit {
            a.user_created(&user.user_id, &user.username, &user.roles)
                .await;
        }
        Ok(user)
    }

    async fn user_list(&self) -> Result<Vec<AccessUser>, ApiError> {
        require_admin()?;
        let store = self.auth_store()?;
        let mut out = Vec::new();
        for rec in store.list_users().map_err(auth_err)? {
            out.push(Self::access_user(store, rec)?);
        }
        Ok(out)
    }

    async fn user_disable(&self, user_id: String, disabled: bool) -> Result<(), ApiError> {
        require_admin()?;
        let store = self.auth_store()?;
        // Atomic last-admin lockout (+ session revoke on disable) in one transaction.
        store
            .set_disabled_guarded(&user_id, disabled)
            .map_err(auth_err)?;
        // Disabling revokes the user's store sessions; also tear down any live mux connection so a
        // disabled account cannot keep acting on an already-open connection. (Re-enabling need not
        // revoke — no live connection to invalidate.)
        if disabled {
            self.revoke_principal(&user_id);
        }
        if let Some(a) = &self.auth_audit {
            a.user_disabled(&user_id, disabled).await;
        }
        Ok(())
    }

    async fn user_set_roles(&self, user_id: String, roles: Vec<String>) -> Result<(), ApiError> {
        require_admin()?;
        let store = self.auth_store()?;
        let parsed = parse_roles(&roles)?;
        // Atomic last-admin lockout: refuses to demote the final admin in one transaction.
        store
            .set_roles_guarded(&user_id, &parsed)
            .map_err(auth_err)?;
        // A role change alters the effective capability set; tear down live connections so they
        // cannot keep acting under the pre-change capabilities.
        self.revoke_principal(&user_id);
        if let Some(a) = &self.auth_audit {
            a.roles_changed(&user_id, &roles).await;
        }
        Ok(())
    }

    async fn user_set_password(&self, user_id: String, password: String) -> Result<(), ApiError> {
        require_admin()?;
        let store = self.auth_store()?;
        // `set_password` re-derives the SCRAM material from the new password (PLAIN + SCRAM stay
        // coherent); we additionally revoke the user's sessions so a reset forces re-login.
        store.set_password(&user_id, &password).map_err(auth_err)?;
        store.revoke_user_sessions(&user_id).map_err(auth_err)?;
        // A password reset revokes store sessions; also tear down live mux connections so the old
        // credential cannot keep acting on an already-open connection.
        self.revoke_principal(&user_id);
        if let Some(a) = &self.auth_audit {
            a.password_reset(&user_id).await;
        }
        Ok(())
    }

    async fn role_list(&self) -> Result<Vec<RoleInfo>, ApiError> {
        require_admin()?;
        Ok(Role::ALL
            .iter()
            .map(|r| RoleInfo {
                role: r.as_str().to_string(),
                capabilities: r.capabilities().into_iter().map(cap_name).collect(),
            })
            .collect())
    }

    async fn who_am_i(&self) -> Result<PrincipalView, ApiError> {
        // Any authenticated principal: no `access_admin` required. Reads the request principal.
        let principal: Principal = current_principal().ok_or_else(|| {
            ApiError::Unauthenticated("no authenticated principal bound to this request".into())
        })?;
        Ok(principal_view(&principal))
    }

    async fn session_revoke(&self, user_id: String) -> Result<(), ApiError> {
        require_admin()?;
        let store = self.auth_store()?;
        store.revoke_user_sessions(&user_id).map_err(auth_err)?;
        // Tear down any live mux connection for this user (Cluster F): the store delete alone only
        // blocks reconnect; the epoch bump closes an already-open connection.
        self.revoke_principal(&user_id);
        if let Some(a) = &self.auth_audit {
            a.sessions_revoked(&user_id).await;
        }
        Ok(())
    }

    // resource_grant_* keep the trait's default `ApiError::Unsupported` (reserved, option B).
}
