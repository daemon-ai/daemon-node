// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Compile-time per-resource ownership (Auth 4, Cluster A): the [`AuthorizedFor<Session>`] capability
//! token and the ownership check that mints it.
//!
//! Phase 1 made the per-resource ownership check a *runtime* gate that every session-touching handler
//! had to remember to call ([`NodeApiImpl::require_session_access`]). This module upgrades that from
//! "must remember to call it" to "the type system won't let you skip it": the guarded
//! [`LiveSessions`](super::internals::LiveSessions) primitives that read/mutate one session's live
//! state now *require* an [`AuthorizedFor<Session>`] argument, and the **only** way to obtain one is
//! [`require_session_access`](NodeApiImpl::require_session_access) — a passing ownership check.
//!
//! **Un-forgeability.** [`AuthorizedFor`] has a private field and its only constructor
//! ([`AuthorizedFor::mint`]) is private to this module, so [`authorize_ownership`] /
//! [`require_session_access`] are the sole producers anywhere in `daemon-host`. Elsewhere in the
//! crate you may *name*, *hold*, and *pass* a token, but you cannot *create* one without passing the
//! check. The crate is `#![forbid(unsafe_code)]`, so there is no `transmute` backdoor either.
//!
//! This composes with — does not duplicate — the Phase 1 runtime check: the check is *reused as the
//! mint site*. It makes exactly the same decision it did before; it now returns the *proof* of that
//! decision instead of `()`, at zero added runtime cost.
//!
//! ## Why the enforcement lives here and not on the wire trait
//!
//! The object-safe [`NodeApi`](daemon_api::NodeApi) trait has nine implementors across crates (the
//! FFI, `daemon-api`'s demo, and the matrix/ingest/http/delivery test mocks). Threading a token into
//! the trait would break all of them and force the token to be `pub` and constructible inside
//! `daemon_api::dispatch` — exactly where it would become forgeable. Enforcing *below* the trait, at
//! the `NodeApiImpl → LiveSessions` boundary inside `daemon-host`, achieves compile-time ownership
//! with no trait change, no wire change, and no impact on those implementors.
//!
//! ## Documented compile-fail (the whole point)
//!
//! A guarded primitive cannot be reached without a token, and a token cannot be forged:
//!
//! ```text
//! // Does NOT compile — `mint` is private to this module, so no other daemon-host code can build one:
//! self.live.submit(&AuthorizedFor::mint(session.clone()), command)   // error: `mint` is private
//!
//! // Does NOT compile — the guarded signature requires `&AuthorizedFor<Session>`, not a `SessionId`:
//! self.live.submit(session, command)                                 // error: mismatched types
//!
//! // The ONLY form that type-checks: obtain the proof from the ownership check first.
//! let auth = self.require_session_access(&session, true).await?;
//! self.live.submit(&auth, command).await
//! ```
//!
//! The whole-workspace build (`cargo clippy -D warnings` in the gate) is that compile-fail check for
//! real code: any daemon-host path that tried to touch guarded session state without a token would
//! fail to compile. See `HARDENING-PLAN.md` for why a standalone `trybuild` case is deferred to the
//! Phase 4 tooling track (the token is `pub(crate)`, so a separate-crate `trybuild`/doctest cannot
//! see it, and adding `trybuild` is a Nix/dependency change better bundled there).

use super::roster::SessionOwnership;
use super::*;

/// Resource-class marker for a session capability token: an uninhabited, zero-sized type used only
/// as the `Resource` parameter of [`AuthorizedFor`]. Never constructed. Kept in this module so
/// `AuthorizedFor<Session>` reads naturally alongside the `SessionId`/`SessionApi` names already in
/// scope (there is no bare `Session` type in `node_api` to collide with).
pub(crate) enum Session {}

/// A capability token proving the Auth 4 per-resource ownership check passed for one specific
/// resource of class `R`. It **carries the [`SessionId`] it authorizes**, so a guarded primitive
/// derives the target from the *proof* rather than a separately-passed id that could disagree with
/// what was checked — closing authorize-A-act-on-B.
///
/// Un-forgeable: the field is private and [`mint`](Self::mint) is private to this module, so the
/// ownership check is the sole producer (see the module docs).
pub(crate) struct AuthorizedFor<R> {
    session: SessionId,
    _resource: std::marker::PhantomData<R>,
}

// A hand-written `Debug` (test diagnostics only) so the `Resource` marker need not be `Debug`; it
// inspects, never constructs, so it does not weaken the mint guarantee.
impl<R> std::fmt::Debug for AuthorizedFor<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthorizedFor")
            .field("session", &self.session)
            .finish()
    }
}

impl<R> AuthorizedFor<R> {
    /// The session this token authorizes. Guarded primitives read the target id from here.
    pub(crate) fn session(&self) -> &SessionId {
        &self.session
    }
}

impl AuthorizedFor<Session> {
    /// Module-private mint — **not** `pub(crate)`. Only [`authorize_ownership`] (this module) may call
    /// it, so a token can only come into existence *after* a passing ownership check.
    fn mint(session: SessionId) -> Self {
        Self {
            session,
            _resource: std::marker::PhantomData,
        }
    }
}

/// The pure ownership decision → token: the single mint site for the interaction/read gate. Store-free
/// (it takes the already-resolved [`SessionOwnership`]), so it is directly unit-testable without a
/// whole [`NodeApiImpl`]. `ownership` is consulted **only** on the owner-comparison arm (an
/// authenticated non-override principal); the None and operator-override decisions are made first and
/// ignore it (which is why [`require_session_access`](NodeApiImpl::require_session_access) can skip
/// the store read on those paths).
///
/// Mirrors the Phase 1 semantics exactly: a `None` principal is DENIED (fail-closed); an override-cap
/// holder passes ([`SessionControlAny`](daemon_auth::Capability::SessionControlAny) for `control`,
/// [`SessionSeeAll`](daemon_auth::Capability::SessionSeeAll) for a read); an `Absent` session passes
/// (the create/not-found path runs downstream); an `Owned` session passes iff owned by the caller; a
/// `LegacyUnowned` (owner-NULL) row is reachable only via the override.
fn authorize_ownership(
    session: &SessionId,
    principal: &Option<daemon_auth::Principal>,
    control: bool,
    ownership: SessionOwnership,
) -> Result<AuthorizedFor<Session>, ApiError> {
    // Fail-closed: an unscoped call (no bound principal) is DENIED — no token is minted.
    let Some(principal) = principal else {
        return Err(ApiError::Unauthenticated(
            "no authenticated principal bound to this request".into(),
        ));
    };
    let override_cap = if control {
        daemon_auth::Capability::SessionControlAny
    } else {
        daemon_auth::Capability::SessionSeeAll
    };
    if principal.has(override_cap) {
        return Ok(AuthorizedFor::mint(session.clone()));
    }
    // Exhaustive over every `SessionOwnership` variant — NO `_` arm (the ownership-layer
    // no-wildcard discipline): a future variant forces a compile-time decision here rather than
    // silently folding into a catch-all. Both non-owner arms remain fail-closed (`Forbidden`).
    match ownership {
        // No such session yet: let the normal create / not-found path handle it downstream.
        SessionOwnership::Absent => Ok(AuthorizedFor::mint(session.clone())),
        // Owned by the caller: authorized.
        SessionOwnership::Owned(owner) if owner == principal.user_id => {
            Ok(AuthorizedFor::mint(session.clone()))
        }
        // Owned by someone else: a non-override principal never crosses ownership.
        SessionOwnership::Owned(_) => Err(ApiError::Forbidden(format!(
            "session {session} is not owned by the caller"
        ))),
        // Legacy owner-NULL row: reachable only via the override cap (decided above); a
        // non-override principal is denied — deny-closed on an unknown owner.
        SessionOwnership::LegacyUnowned => Err(ApiError::Forbidden(format!(
            "session {session} has no owner and is reachable only via an operator override"
        ))),
    }
}

impl NodeApiImpl {
    /// The per-resource ownership gate (Auth 4), enforced *beneath* Auth 2's coarse capability gate,
    /// now returning an [`AuthorizedFor<Session>`] **proof** the guarded
    /// [`LiveSessions`](super::internals::LiveSessions) primitives require. The caller must own
    /// `session`, or hold the relevant override capability:
    /// [`SessionControlAny`](daemon_auth::Capability::SessionControlAny) for an interaction op
    /// (`control = true`) or [`SessionSeeAll`](daemon_auth::Capability::SessionSeeAll) for a
    /// read-of-one (`control = false`). An `Absent` session passes so the create/`NotFound` flow runs
    /// downstream; a `LegacyUnowned` (owner-NULL) session is reachable only via the override.
    ///
    /// **Fail-closed on a missing principal.** A `None` principal (no request context bound) is
    /// DENIED — no token is produced: every legitimate in-process caller enters an explicit
    /// [`RequestContext::system`](crate::RequestContext::system) or
    /// [`RequestContext::internal`](crate::RequestContext::internal) scope, so an unscoped call here
    /// is a bug, never implicit full trust.
    ///
    /// Preserves the Phase 1 short-circuit: the `None` and operator-override decisions need no store
    /// read; only an authenticated non-override principal resolves the durable owner
    /// ([`session_ownership`](Self::session_ownership)).
    pub(crate) async fn require_session_access(
        &self,
        session: &SessionId,
        control: bool,
    ) -> Result<AuthorizedFor<Session>, ApiError> {
        let principal = crate::request_context::current_principal();
        // Only an authenticated principal that lacks the override cap needs the durable owner; the
        // None (deny) and override (allow) arms are decided by `authorize_ownership` without it.
        let needs_owner = matches!(
            &principal,
            Some(p) if !p.has(if control {
                daemon_auth::Capability::SessionControlAny
            } else {
                daemon_auth::Capability::SessionSeeAll
            })
        );
        // A placeholder on the no-store-read paths (None/override): `authorize_ownership` returns
        // before ever inspecting `ownership` on those arms.
        let ownership = if needs_owner {
            self.session_ownership(session).await
        } else {
            SessionOwnership::Absent
        };
        authorize_ownership(session, &principal, control, ownership)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_context::RequestContext;
    use daemon_auth::{Principal, Role};

    fn session() -> SessionId {
        SessionId::new("s-test")
    }

    fn user(name: &str) -> Option<Principal> {
        Some(Principal::from_roles(name, name, vec![Role::User]))
    }

    fn operator() -> Option<Principal> {
        // Operator grants both session overrides (SessionSeeAll + SessionControlAny).
        Some(Principal::from_roles("op", "op", vec![Role::Operator]))
    }

    // ---- the core new guarantee: a token is minted IFF the ownership check passes -----------------

    #[test]
    fn owner_mints_token_carrying_the_checked_session() {
        let s = session();
        for control in [true, false] {
            let tok = authorize_ownership(
                &s,
                &user("alice"),
                control,
                SessionOwnership::Owned("alice".into()),
            )
            .expect("owner is authorized");
            // The proof carries the checked id, so a guarded primitive cannot be pointed elsewhere.
            assert_eq!(
                tok.session(),
                &s,
                "token must carry the authorized session id"
            );
        }
    }

    #[test]
    fn non_owner_is_denied_and_yields_no_token() {
        let err = authorize_ownership(
            &session(),
            &user("alice"),
            true,
            SessionOwnership::Owned("bob".into()),
        )
        .expect_err("a non-owner must be denied");
        assert!(matches!(err, ApiError::Forbidden(_)));
    }

    #[test]
    fn none_principal_is_denied_fail_closed() {
        // The fail-closed default (Phase 1): no bound principal → no token, for read and write alike.
        for control in [true, false] {
            let err = authorize_ownership(&session(), &None, control, SessionOwnership::Absent)
                .expect_err("an unscoped call must be denied");
            assert!(matches!(err, ApiError::Unauthenticated(_)));
        }
    }

    #[test]
    fn legacy_unowned_denied_for_non_operator() {
        let err = authorize_ownership(
            &session(),
            &user("alice"),
            false,
            SessionOwnership::LegacyUnowned,
        )
        .expect_err("a legacy owner-NULL row is hidden from a non-operator");
        assert!(matches!(err, ApiError::Forbidden(_)));
    }

    #[test]
    fn see_all_override_mints_for_a_read_of_a_foreign_session() {
        // control = false → the read override is SessionSeeAll (held by Operator).
        let tok = authorize_ownership(
            &session(),
            &operator(),
            false,
            SessionOwnership::Owned("bob".into()),
        )
        .expect("an operator read override crosses ownership");
        assert_eq!(tok.session(), &session());
    }

    #[test]
    fn control_any_override_mints_for_an_interaction_on_a_foreign_session() {
        // control = true → the interaction override is SessionControlAny (held by Operator).
        authorize_ownership(
            &session(),
            &operator(),
            true,
            SessionOwnership::Owned("bob".into()),
        )
        .expect("an operator interaction override crosses ownership");
    }

    #[test]
    fn absent_session_mints_so_the_create_path_runs() {
        // A brand-new (Absent) session passes: the token lets the create/not-found path run, then
        // ownership is stamped downstream.
        authorize_ownership(&session(), &user("alice"), true, SessionOwnership::Absent)
            .expect("an absent session passes the gate");
    }

    #[test]
    fn internal_marker_mints_like_an_operator() {
        // The synthetic in-process `internal` principal (Operator ⇒ both overrides) crosses ownership
        // for the legitimate embedded callers (mux/HTTP pumps, ingest, delivery, injection).
        let internal = Some(RequestContext::internal().principal);
        authorize_ownership(
            &session(),
            &internal,
            true,
            SessionOwnership::Owned("bob".into()),
        )
        .expect("the internal marker crosses ownership like an operator");
        authorize_ownership(
            &session(),
            &internal,
            false,
            SessionOwnership::LegacyUnowned,
        )
        .expect("the internal marker reads a legacy-unowned row like an operator");
    }
}
