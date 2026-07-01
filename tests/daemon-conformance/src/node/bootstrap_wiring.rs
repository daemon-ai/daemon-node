// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Track A wiring guards (#2 + #3): binding the `AuthStore` + shared `AuthAudit` onto the assembled
//! node makes the admin `AccessControl` surface functional for the local-trust `system` principal
//! (previously `Unsupported`), and the first-admin bootstrap seeds exactly one admin into an empty
//! store (idempotently). These exercise the in-process node the binary assembles + re-wraps in
//! `run_as_host`; the over-the-wire SCRAM path is covered in `positive_e2e`.

use super::harness::*;

use daemon_api::{AccessControlApi, ApiError};
use daemon_auth::{AdminSeed, AuthStore, Role};
use daemon_common::{JournalStreamId, UnitId};
use daemon_host::{
    auth_audit::AUTH_JOURNAL_UNIT, with_request_context, AuthAudit, NodeApiImpl, RequestContext,
};
use daemon_store::SessionStore;
use daemon_telemetry::{decode_entry, JournalPayload, TraceSigner};

/// An assembled node with the identity store + auth-audit sink bound the way `run_as_host` does
/// (deref-clone-rewrap of the `Arc<NodeApiImpl>`). The audit records onto a caller-owned in-memory
/// store so the test can read the chain back.
struct Bound {
    node: NodeApiImpl,
    store: Arc<AuthStore>,
    audit_store: Arc<dyn SessionStore>,
}

fn bound_node() -> Bound {
    let (node, _handle) = assemble();
    let store = Arc::new(AuthStore::open_in_memory().expect("auth store"));
    let audit_store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let signer = Arc::new(TraceSigner::generate());
    let audit = AuthAudit::shared(audit_store.clone(), signer);
    let node = (*node)
        .clone()
        .with_auth_store(store.clone())
        .with_auth_audit(audit);
    Bound {
        node,
        store,
        audit_store,
    }
}

/// Run `fut` as the local-trust `system` principal (the default `local_trust=system` Unix path).
async fn as_system<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    with_request_context(RequestContext::system(), fut).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_ops_work_for_the_local_trust_system_principal_once_bound() {
    let b = bound_node();
    // The system principal (full-trust local operator) can now drive user CRUD: previously these
    // resolved to `Unsupported` because no `AuthStore` was bound.
    let created = as_system(
        b.node
            .user_create("alice".into(), "pw".into(), vec!["user".into()]),
    )
    .await
    .expect("user_create succeeds for the bound system principal");
    assert_eq!(created.username, "alice");

    let users = as_system(b.node.user_list())
        .await
        .expect("user_list succeeds for the bound system principal");
    assert!(users.iter().any(|u| u.username == "alice"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_ops_are_unsupported_without_a_bound_store() {
    // A plain assembled node (no `with_auth_store`) gates admin ops as `Unsupported` — the
    // regression anchor the #3 bind removes.
    let (node, _handle) = assemble();
    assert!(matches!(
        as_system(node.user_list()).await,
        Err(ApiError::Unsupported(_))
    ));
    assert!(matches!(
        as_system(node.user_create("x".into(), "pw".into(), vec!["user".into()])).await,
        Err(ApiError::Unsupported(_))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bootstrap_seeded_admin_is_listed_and_is_admin() {
    let b = bound_node();
    // Seed via the #2 bootstrap into the (empty) bound store, then observe it through the surface.
    let seeded = b
        .store
        .seed_first_admin_if_empty(AdminSeed::Generate)
        .expect("seed")
        .expect("seeded on empty store");
    // Idempotent: a second boot does not add another user.
    assert!(b
        .store
        .seed_first_admin_if_empty(AdminSeed::Generate)
        .expect("second seed")
        .is_none());

    let users = as_system(b.node.user_list()).await.expect("list");
    assert_eq!(users.len(), 1, "exactly one seeded admin");
    let admin = &users[0];
    assert_eq!(admin.username, seeded.username);
    assert!(
        admin.roles.contains(&Role::Admin.as_str().to_string()),
        "seeded user holds the admin role, got {:?}",
        admin.roles
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_op_on_system_path_is_audited_without_secret_material() {
    let b = bound_node();
    let secret = "correct horse battery staple";
    as_system(
        b.node
            .user_create("carol".into(), secret.into(), vec!["user".into()]),
    )
    .await
    .expect("create");

    let stream = JournalStreamId::unit(&UnitId::new(AUTH_JOURNAL_UNIT));
    let page = b.audit_store.load_journal(&stream, 0, 100).await;
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
