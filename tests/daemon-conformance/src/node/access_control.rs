// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Admin access-control surface (Auth 5): the `AccessControlApi` handlers over a real assembled node
//! with an attached identity store + auth-audit sink. Proves the safety guards end-to-end:
//! every admin op denies a non-`AccessAdmin` caller (`Forbidden`); `WhoAmI` works for any
//! authenticated principal; last-admin lockout; `UserSetPassword` revokes sessions; `ResourceGrant*`
//! -> `Unsupported`; and admin mutations are audited with NO credential material in any payload.

use super::harness::*;

use daemon_api::{AccessControlApi, ApiError};
use daemon_auth::{AuthStore, Principal, Role, DEFAULT_SESSION_TTL_SECS};
use daemon_common::{JournalStreamId, UnitId};
use daemon_host::{
    auth_audit::AUTH_JOURNAL_UNIT, with_request_context, AuthAudit, NodeApiImpl, RequestContext,
};
use daemon_store::SessionStore;
use daemon_telemetry::{decode_entry, JournalPayload, TraceSigner};

/// An assembled node with an in-memory identity store + auth-audit sink attached. The audit chain is
/// recorded onto a dedicated in-memory store so the test can read it back and verify it.
struct Fixture {
    node: NodeApiImpl,
    store: Arc<AuthStore>,
    audit_store: Arc<dyn SessionStore>,
}

fn fixture() -> Fixture {
    let (node, _handle) = assemble();
    let store = Arc::new(AuthStore::open_in_memory().expect("auth store"));
    let audit_store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let signer = Arc::new(TraceSigner::generate());
    let audit = AuthAudit::shared(audit_store.clone(), signer);
    // NodeApiImpl is Clone; attach the access-control seam to an owned copy.
    let node = (*node)
        .clone()
        .with_auth_store(store.clone())
        .with_auth_audit(audit);
    Fixture {
        node,
        store,
        audit_store,
    }
}

/// Run `fut` with `principal` bound as the request context (so the handlers' gate + `who_am_i` see it).
async fn as_principal<F, T>(principal: Principal, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    with_request_context(RequestContext::authenticated(principal, None), fut).await
}

fn admin(store: &AuthStore) -> Principal {
    let rec = store
        .create_user("root", "rootpw", &[Role::Admin])
        .expect("create admin");
    store
        .principal_for_user(&rec.id, &rec.username)
        .expect("admin principal")
}

fn viewer() -> Principal {
    Principal::from_roles("v", "viewer", vec![Role::Viewer])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn who_am_i_works_for_any_authenticated_principal() {
    let f = fixture();
    let p = viewer();
    let view = as_principal(p.clone(), f.node.who_am_i())
        .await
        .expect("who_am_i ok for a viewer");
    assert_eq!(view.username, "viewer");
    assert!(view.capabilities.contains(&"session_read".to_string()));
    assert!(!view.capabilities.contains(&"access_admin".to_string()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_admin_op_denies_a_non_admin_caller() {
    let f = fixture();
    let v = viewer();
    // Each handler re-checks `AccessAdmin` (defense in depth): a non-admin principal is Forbidden.
    assert!(matches!(
        as_principal(v.clone(), f.node.user_list()).await,
        Err(ApiError::Forbidden(_))
    ));
    assert!(matches!(
        as_principal(
            v.clone(),
            f.node
                .user_create("x".into(), "pw".into(), vec!["user".into()])
        )
        .await,
        Err(ApiError::Forbidden(_))
    ));
    assert!(matches!(
        as_principal(v.clone(), f.node.user_disable("u".into(), true)).await,
        Err(ApiError::Forbidden(_))
    ));
    assert!(matches!(
        as_principal(
            v.clone(),
            f.node.user_set_roles("u".into(), vec!["user".into()])
        )
        .await,
        Err(ApiError::Forbidden(_))
    ));
    assert!(matches!(
        as_principal(v.clone(), f.node.session_revoke("u".into())).await,
        Err(ApiError::Forbidden(_))
    ));
    assert!(matches!(
        as_principal(v.clone(), f.node.role_list()).await,
        Err(ApiError::Forbidden(_))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_can_create_and_list_users() {
    let f = fixture();
    let a = admin(&f.store);
    let created = as_principal(
        a.clone(),
        f.node
            .user_create("alice".into(), "pw".into(), vec!["operator".into()]),
    )
    .await
    .expect("create");
    assert_eq!(created.username, "alice");
    assert_eq!(created.roles, vec!["operator".to_string()]);

    let users = as_principal(a.clone(), f.node.user_list())
        .await
        .expect("list");
    assert!(users.iter().any(|u| u.username == "alice"));
    assert!(users.iter().any(|u| u.username == "root"));

    // Unknown role string is rejected (fail-closed).
    assert!(matches!(
        as_principal(
            a,
            f.node
                .user_create("bob".into(), "pw".into(), vec!["wizard".into()])
        )
        .await,
        Err(ApiError::Other(_))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn last_admin_lockout_is_enforced() {
    let f = fixture();
    let a = admin(&f.store);
    let root_id = a.user_id.clone();
    // Sole admin: cannot demote or disable it.
    assert!(matches!(
        as_principal(
            a.clone(),
            f.node
                .user_set_roles(root_id.clone(), vec!["operator".into()])
        )
        .await,
        Err(ApiError::Forbidden(_))
    ));
    assert!(matches!(
        as_principal(a.clone(), f.node.user_disable(root_id.clone(), true)).await,
        Err(ApiError::Forbidden(_))
    ));
    // With a second admin, the first can be demoted.
    as_principal(
        a.clone(),
        f.node
            .user_create("root2".into(), "pw".into(), vec!["admin".into()]),
    )
    .await
    .expect("second admin");
    as_principal(a, f.node.user_set_roles(root_id, vec!["operator".into()]))
        .await
        .expect("demote allowed now");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn user_set_password_revokes_sessions() {
    let f = fixture();
    let a = admin(&f.store);
    let u = f
        .store
        .create_user("dave", "old", &[Role::User])
        .expect("user");
    let token = f
        .store
        .mint_session(&u.id, DEFAULT_SESSION_TTL_SECS, "scram")
        .expect("token");
    assert!(f.store.principal_for_token(&token).is_ok());

    as_principal(a, f.node.user_set_password(u.id.clone(), "new".into()))
        .await
        .expect("set password");

    // The reset revoked the user's sessions: the old token no longer resolves.
    assert!(f.store.principal_for_token(&token).is_err());
    // SCRAM material was re-derived, so the new password verifies under PLAIN/Argon2 too.
    assert!(f.store.authenticate_password("dave", "new").is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resource_grants_are_unsupported() {
    let f = fixture();
    let a = admin(&f.store);
    assert!(matches!(
        as_principal(
            a.clone(),
            f.node.resource_grant_create(
                "u".into(),
                "session".into(),
                "s".into(),
                "session_read".into(),
            )
        )
        .await,
        Err(ApiError::Unsupported(_))
    ));
    assert!(matches!(
        as_principal(a.clone(), f.node.resource_grant_list(None)).await,
        Err(ApiError::Unsupported(_))
    ));
    assert!(matches!(
        as_principal(a, f.node.resource_grant_revoke("g".into())).await,
        Err(ApiError::Unsupported(_))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_mutations_are_audited_without_secret_material() {
    let f = fixture();
    let a = admin(&f.store);
    // A password that must NOT appear anywhere in the audit chain.
    let secret = "correct horse battery staple";
    as_principal(
        a,
        f.node
            .user_create("alice".into(), secret.into(), vec!["user".into()]),
    )
    .await
    .expect("create");

    let stream = JournalStreamId::unit(&UnitId::new(AUTH_JOURNAL_UNIT));
    let page = f.audit_store.load_journal(&stream, 0, 100).await;
    assert!(!page.entries.is_empty(), "the create was audited");
    let mut saw_created = false;
    for je in &page.entries {
        let view = decode_entry(&je.entry.bytes).expect("decode audit entry");
        if view.kind == "auth.user_created" {
            saw_created = true;
        }
        if let JournalPayload::Management { detail } = view.payload {
            assert!(
                !detail.contains(secret),
                "audit payload must NOT contain the password: {detail}"
            );
        }
    }
    assert!(saw_created, "an auth.user_created record was written");
}
