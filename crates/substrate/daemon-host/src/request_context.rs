// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The per-request task-local identity scope (authz core, Auth 2).
//!
//! Every server-side request runs *inside* a [`with_request_context`] scope that binds the
//! authenticated [`Principal`] (plus its provenance) as a task-local. The capability gate
//! ([`crate::authz::authorize`]) and the per-resource ownership checks read it back through
//! [`current_principal`] / [`current_context`].
//!
//! **Fail-closed is structural.** The task-local has no value outside a scope, so
//! [`current_principal`] returns [`None`] *only* when no context is active — and `None` means "no
//! capabilities", i.e. DENY. A context can only be entered with a concrete [`Principal`]
//! (either network-authenticated via [`RequestContext::authenticated`] or the deliberate
//! local-trust [`RequestContext::system`]); there is no "context present but identity absent"
//! middle state. Mirrors the [`with_trace`](daemon_telemetry) task-local pattern.

use daemon_auth::{Principal, Role};
use daemon_protocol::Origin;
use std::future::Future;

tokio::task_local! {
    static REQUEST_CONTEXT: RequestContext;
}

/// The reserved usernames of the two synthetic in-process principals — [`RequestContext::system`]
/// (`"system"`) and [`RequestContext::internal`] (`"internal"`). Re-exported from `daemon-auth`,
/// which owns the reservation and rejects creating a real store user with either name (so neither
/// synthetic identity can be forged by a network user whose ownership stamp would then collide).
pub use daemon_auth::{INTERNAL_USERNAME, SYSTEM_USERNAME};

/// How the principal bound to the current request proved its identity. Advisory — carried for
/// audit/telemetry only; the capability gate keys off [`Principal::capabilities`], not this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthMethod {
    /// In-process / FFI / local-Unix trust — no network authentication was performed. The marker
    /// of a [`RequestContext::system`] principal.
    LocalTrust,
    /// SASL SCRAM-SHA-256 (the primary network mechanism).
    Scram,
    /// SASL PLAIN (password over an already-encrypted channel).
    Plain,
    /// SASL EXTERNAL (mutual-TLS client certificate).
    External,
    /// A resumed server-side session token (`AuthResume`).
    Token,
}

/// The identity + provenance bound to one in-flight request.
///
/// Established once — post-authentication on a network transport, or at a deliberate local-trust
/// site — and read by the gate/ownership checks for the duration of the request. The `principal`
/// is **non-optional**: the absence of identity is modeled by the *absence of a scope*, not by a
/// null principal (see the module docs).
#[derive(Clone, Debug)]
pub struct RequestContext {
    /// The resolved caller identity + effective capability set.
    pub principal: Principal,
    /// The inbound chat/transport attribution, when the request arrived via an adapter. `None` for
    /// direct node clients and the local-trust path.
    pub origin: Option<Origin>,
    /// The connection this request rode in on (server-assigned; audit/telemetry correlation).
    pub conn_id: Option<u64>,
    /// How `principal` authenticated (audit/telemetry).
    pub auth_method: Option<AuthMethod>,
}

impl RequestContext {
    /// A network-authenticated request context: bind a resolved [`Principal`] and its inbound
    /// [`Origin`]. The primary entry point for the transport/handshake layer (Auth 3). `conn_id` /
    /// `auth_method` default to `None`; set them with the builders below.
    pub fn authenticated(principal: Principal, origin: Option<Origin>) -> Self {
        Self {
            principal,
            origin,
            conn_id: None,
            auth_method: None,
        }
    }

    /// The deliberate **local-trust** escape hatch: a full-capability principal (the complete
    /// [`Role::Admin`] capability set) under the reserved [`SYSTEM_USERNAME`].
    ///
    /// This is the *only* constructor that injects [`Role::Admin`] without consulting the identity
    /// store, so it is the single audit point for unauthenticated full trust. Construct it ONLY at
    /// deployment-trusted in-process / FFI / local-Unix sites (the binary decides via
    /// `[api].local_trust` whether the Unix socket adopts it); NEVER on a TCP/network path, where
    /// every request must carry a store-authenticated principal instead.
    pub fn system() -> Self {
        Self {
            principal: Principal::from_roles("system", SYSTEM_USERNAME, vec![Role::Admin]),
            origin: None,
            conn_id: None,
            auth_method: Some(AuthMethod::LocalTrust),
        }
    }

    /// The in-process **embedded-caller** marker: trusted node internals that legitimately cross
    /// session ownership without a request principal — the mux/HTTP stream pumps, chat ingest
    /// ([`daemon-ingest`]), outbound delivery, and background input injection. Constructed ONLY here
    /// (never derivable from wire input), so after the ownership layer flips `None` from allow to
    /// deny, these paths carry an explicit identity instead of the old "no principal ⇒ full trust".
    ///
    /// Distinct from [`system`](Self::system): `internal` holds exactly the operator-tier session
    /// overrides ([`Role::Operator`] ⇒ `SessionSeeAll` + `SessionControlAny`) — enough to read/drive
    /// any session for delivery/ingest — but NOT `AccessAdmin`, and it stamps ownership as the
    /// reserved user id/username `"internal"` (see [`INTERNAL_USERNAME`]) so audit and roster reads
    /// can tell it apart from `system` and from real operators.
    pub fn internal() -> Self {
        Self {
            principal: Principal::from_roles("internal", INTERNAL_USERNAME, vec![Role::Operator]),
            origin: None,
            conn_id: None,
            auth_method: Some(AuthMethod::LocalTrust),
        }
    }

    /// Attach the server-assigned connection id (audit/telemetry correlation).
    pub fn with_conn_id(mut self, conn_id: u64) -> Self {
        self.conn_id = Some(conn_id);
        self
    }

    /// Record how the principal authenticated (audit/telemetry).
    pub fn with_auth_method(mut self, auth_method: AuthMethod) -> Self {
        self.auth_method = Some(auth_method);
        self
    }
}

/// Run `fut` with `ctx` bound as the task-local request context. Within the scope (and any task
/// that inherits it via `.await`, *not* a freshly `spawn`ed task), [`current_principal`] /
/// [`current_context`] resolve to `ctx`; once `fut` completes the binding is gone (deny again).
pub async fn with_request_context<F, T>(ctx: RequestContext, fut: F) -> T
where
    F: Future<Output = T>,
{
    REQUEST_CONTEXT.scope(ctx, fut).await
}

/// The [`Principal`] bound to the current request, or [`None`] when no context is active.
///
/// `None` is the fail-closed default: it means no identity has been established for this task, so
/// the caller holds no capabilities and every gated operation must be denied.
pub fn current_principal() -> Option<Principal> {
    REQUEST_CONTEXT.try_with(|ctx| ctx.principal.clone()).ok()
}

/// The full current [`RequestContext`] (identity + origin + provenance), or [`None`] when no
/// context is active. Used by the ownership layer (origin/`conn_id`) and audit (`auth_method`).
pub fn current_context() -> Option<RequestContext> {
    REQUEST_CONTEXT.try_with(Clone::clone).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_auth::Capability;
    use std::sync::Arc;
    use tokio::sync::Barrier;

    fn user(name: &str) -> Principal {
        Principal::from_roles(name, name, vec![Role::User])
    }

    #[tokio::test]
    async fn default_context_denies() {
        // Outside any scope: no principal == no capabilities == deny.
        assert!(current_principal().is_none());
        assert!(current_context().is_none());
    }

    #[tokio::test]
    async fn scope_binds_then_resets_to_deny() {
        assert!(current_principal().is_none());
        let ctx = RequestContext::authenticated(user("alice"), None);
        with_request_context(ctx, async {
            let p = current_principal().expect("principal bound inside scope");
            assert_eq!(p.username, "alice");
        })
        .await;
        // Scope dropped -> back to fail-closed.
        assert!(current_principal().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_tasks_are_isolated() {
        // Two tasks, each in its own scope with a distinct principal, forced to interleave on a
        // shared barrier. Neither must ever observe the other's principal, and the spawner (no
        // scope) must observe none.
        let barrier = Arc::new(Barrier::new(2));
        let b1 = barrier.clone();
        let t1 = tokio::spawn(async move {
            with_request_context(
                RequestContext::authenticated(user("alice"), None),
                async move {
                    b1.wait().await;
                    tokio::task::yield_now().await;
                    current_principal().unwrap().username
                },
            )
            .await
        });
        let b2 = barrier.clone();
        let t2 = tokio::spawn(async move {
            with_request_context(
                RequestContext::authenticated(user("bob"), None),
                async move {
                    b2.wait().await;
                    tokio::task::yield_now().await;
                    current_principal().unwrap().username
                },
            )
            .await
        });
        let (a, b) = (t1.await.unwrap(), t2.await.unwrap());
        assert_eq!(a, "alice");
        assert_eq!(b, "bob");
        // The spawning task never entered a scope.
        assert!(current_principal().is_none());
    }

    #[tokio::test]
    async fn system_principal_is_full_local_trust() {
        let ctx = RequestContext::system();
        assert_eq!(ctx.principal.username, SYSTEM_USERNAME);
        assert_eq!(ctx.auth_method, Some(AuthMethod::LocalTrust));
        assert_eq!(ctx.principal.roles, vec![Role::Admin]);
        // Holds *every* capability, including the admin and operator-override caps.
        for cap in ALL_CAPABILITIES {
            assert!(ctx.principal.has(cap), "system principal must hold {cap:?}");
        }
        assert!(ctx.principal.has(Capability::AccessAdmin));
    }

    #[tokio::test]
    async fn system_is_the_only_full_trust_constructor() {
        // A network-authenticated non-admin principal must NOT pick up full trust: only `system()`
        // injects `Role::Admin`. This is the "only constructible at intended sites" guard — the
        // sole code path to a full-capability principal without a store lookup is `system()`.
        let ctx = RequestContext::authenticated(user("mallory"), None);
        assert!(!ctx.principal.has(Capability::AccessAdmin));
        assert!(!ctx.principal.has(Capability::ControlWrite));
        assert_ne!(ctx.principal.username, SYSTEM_USERNAME);
    }

    /// The full capability vocabulary, used to assert `system()` is genuinely full-trust.
    const ALL_CAPABILITIES: [Capability; 24] = [
        Capability::SessionRead,
        Capability::SessionWrite,
        Capability::SessionSeeAll,
        Capability::SessionControlAny,
        Capability::ControlRead,
        Capability::ControlWrite,
        Capability::FleetRead,
        Capability::FleetWrite,
        Capability::ModelsRead,
        Capability::ModelsWrite,
        Capability::ProfileRead,
        Capability::ProfileWrite,
        Capability::CredentialRead,
        Capability::CredentialWrite,
        Capability::CronRead,
        Capability::CronWrite,
        Capability::RoutingRead,
        Capability::RoutingWrite,
        Capability::MessagingRead,
        Capability::MessagingWrite,
        Capability::RegistryRead,
        Capability::RegistryWrite,
        Capability::FsRead,
        Capability::FsWrite,
    ];
}
