// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE NODE-AUTHORITATIVE FIRST-RUN REPLACEMENT GATE (wire v47): the first-boot placeholder the
//! daemon seeds carries `seeded = true`, and its retirement is a NODE decision — an operator
//! `ProfileCreate` removes a still-seeded, session-free placeholder and promotes the new profile
//! to default. Clients never guess which default is disposable (the app-side
//! `remove(defaultProfileId)` heuristic this replaces deleted real user profiles).
//!
//! Also proves the marker's integrity rails: `seeded` is minted ONLY by `ProfileStore::seed` —
//! a wire create claiming it is normalized to `false`; an operator update targeting the
//! placeholder ADOPTS it (clears the marker, blocking later retirement); a bound session (any
//! state) blocks retirement; and a clone of the placeholder never inherits the marker.

use std::sync::Arc;

use daemon_api::{ProfileApi, ProfileSpec, ProviderSelector, SessionApi};
use daemon_common::ProfileRef;
use daemon_core::{MockProvider, Provider, ProviderBuilder, ProviderRegistry};
use daemon_host::{HostConfig, MemProfileStore, ProfileStore};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_store::InMemoryStore;

/// Assemble a full node over a caller-provided (possibly pre-seeded) profile store — the same
/// surface the daemon binary wires: `ProfileStore::seed` runs BEFORE assembly there too.
fn assemble_node(
    profiles: Arc<dyn ProfileStore>,
) -> (Arc<daemon_host::NodeApiImpl>, daemon_host::SupervisorHandle) {
    let resolver: ProviderResolver = Arc::new(move |_spec: &ProfileSpec| {
        let builder: ProviderBuilder =
            Arc::new(|| Arc::new(MockProvider::completing("native reply")) as Arc<dyn Provider>);
        builder
    });
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: daemon_common::PartitionId::DEFAULT,
        host_config: HostConfig::default(),
        providers,
        credentials: None,
        profile: ProfileRef::new("default"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x47; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(profiles),
        provider_resolver: Some(resolver),
        credential_store: None,
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: None,
        blob_root: None,
        fs: Default::default(),
        processes: Default::default(),
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
    });
    (node, handle)
}

/// A store holding exactly what a fresh daemon boot leaves behind: the seeded placeholder,
/// marked and active (the daemon binary calls `ProfileStore::seed` with the boot spec).
fn seeded_store(id: &str) -> Arc<dyn ProfileStore> {
    let store = Arc::new(MemProfileStore::new());
    assert!(store
        .seed(ProfileSpec::new(id, ProviderSelector::Mock, "m"))
        .expect("seed the fresh store"));
    store as Arc<dyn ProfileStore>
}

/// A plain operator profile spec (Core engine, mock provider — validation-clean).
fn operator_profile(id: &str) -> ProfileSpec {
    ProfileSpec::new(id, ProviderSelector::Mock, "m")
}

/// Fresh boot -> the placeholder is marked + active; the operator's first `ProfileCreate`
/// retires it and the new profile takes over as the active default. The client sends ONLY the
/// create — no remove, no select.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn operator_create_retires_the_seeded_placeholder_and_takes_default() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let (node, _handle) = assemble_node(seeded_store("default"));

        let before = node.profile_list().await;
        assert_eq!(before.len(), 1);
        assert!(before[0].seeded, "the seeded placeholder must be marked");
        assert!(before[0].is_active, "the seeded placeholder starts active");

        node.profile_create(operator_profile("mine"))
            .await
            .expect("the operator's first create");

        let after = node.profile_list().await;
        assert_eq!(
            after.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
            vec!["mine"],
            "the untouched placeholder is retired by the create"
        );
        assert!(after[0].is_active, "the new profile inherits the default");
        assert!(!after[0].seeded, "a created profile is never seeded");
    })
    .await;
}

/// An operator update targeting the placeholder ADOPTS it: the marker clears (even when the
/// incoming wire spec claims `seeded = true`), so a later create no longer retires it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_update_adopts_the_placeholder_and_blocks_retirement() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let (node, _handle) = assemble_node(seeded_store("default"));

        // The update claims `seeded = true` on the wire — normalization must ignore it.
        let mut configured = operator_profile("default");
        configured.model = "configured-model".into();
        configured.seeded = true;
        node.profile_update(configured)
            .await
            .expect("configure (adopt) the placeholder");

        let adopted = node
            .profile_get("default".into())
            .await
            .expect("get the adopted profile")
            .expect("the adopted profile exists");
        assert!(
            !adopted.seeded,
            "adoption clears the marker, wire claim ignored"
        );

        node.profile_create(operator_profile("mine"))
            .await
            .expect("a later create");
        let mut ids: Vec<String> = node
            .profile_list()
            .await
            .into_iter()
            .map(|p| p.id)
            .collect();
        ids.sort();
        assert_eq!(
            ids,
            vec!["default", "mine"],
            "an adopted profile is never retired"
        );
    })
    .await;
}

/// A wire create claiming `seeded = true` is normalized to `false` — so it can never be
/// retired by a subsequent create (only `ProfileStore::seed` mints the marker).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wire_seeded_claim_is_normalized_on_create() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let (node, _handle) = assemble_node(Arc::new(MemProfileStore::new()));

        let mut sneaky = operator_profile("sneaky");
        sneaky.seeded = true;
        node.profile_create(sneaky).await.expect("create");

        let stored = node
            .profile_get("sneaky".into())
            .await
            .expect("get")
            .expect("exists");
        assert!(!stored.seeded, "the wire claim is normalized node-side");

        node.profile_create(operator_profile("second"))
            .await
            .expect("a second create");
        let mut ids: Vec<String> = node
            .profile_list()
            .await
            .into_iter()
            .map(|p| p.id)
            .collect();
        ids.sort();
        assert_eq!(
            ids,
            vec!["second", "sneaky"],
            "a normalized row is never retired"
        );
    })
    .await;
}

/// A session bound to the placeholder blocks retirement — the create still succeeds, but the
/// placeholder row survives (node-side safety: never delete a profile in use).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bound_session_blocks_retirement() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let (node, _handle) = assemble_node(seeded_store("default"));

        node.session_create(None, Some(ProfileRef::new("default")))
            .await
            .expect("bind a session to the placeholder");

        node.profile_create(operator_profile("mine"))
            .await
            .expect("the create itself must not fail");
        let mut ids: Vec<String> = node
            .profile_list()
            .await
            .into_iter()
            .map(|p| p.id)
            .collect();
        ids.sort();
        assert_eq!(
            ids,
            vec!["default", "mine"],
            "a placeholder with a bound session is never deleted"
        );
        let active: Vec<String> = node
            .profile_list()
            .await
            .into_iter()
            .filter(|p| p.is_active)
            .map(|p| p.id)
            .collect();
        assert_eq!(
            active,
            vec!["default"],
            "the default selection is untouched"
        );
    })
    .await;
}

/// A clone of the placeholder never inherits the marker (cloning must not mint a second
/// retireable row), and cloning triggers no retirement (only `ProfileCreate` does).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_clone_of_the_placeholder_is_not_seeded() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let (node, _handle) = assemble_node(seeded_store("default"));

        node.profile_clone("default".into(), "copy".into())
            .await
            .expect("clone the placeholder");

        let copy = node
            .profile_get("copy".into())
            .await
            .expect("get")
            .expect("exists");
        assert!(!copy.seeded, "a clone never inherits the seeded marker");

        let mut ids: Vec<String> = node
            .profile_list()
            .await
            .into_iter()
            .map(|p| p.id)
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["copy", "default"], "a clone retires nothing");
    })
    .await;
}
