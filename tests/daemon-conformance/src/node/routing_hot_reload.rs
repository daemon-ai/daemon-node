// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE §5.9.4 ACCOUNT→PROFILE HOT-RELOAD GATE (EIO-3 / EIO-11 "routing_builder unwired"): editing a
//! profile's `bound_accounts` must refresh the live routing registry's `instance_profiles` baseline
//! **without a restart**. The chat-pin half of hot-reload was already proven
//! (`routing_auth::routing_pin_resolves_to_bound_session`); this suite pins the *account* half,
//! which only works when the assembling composition root installs the routing **rebuild hook**
//! ([`daemon_host::RoutingBuilder`]) instead of a boot-time static snapshot — the actual
//! `routing_builder` hole the user stories call out.
//!
//! Both tests are RED on a tree whose `install_routing` derives `instance_profiles` once at
//! assembly, and GREEN once the builder recomputes it from the live profile store on every
//! `rebuild_routing()` (fired by `profile_update` / `auth_complete` / the `routing_*` ops).
//!
//! Sticky-profile discipline: `submit_routed` binds the resolved profile sticky-on-first-open, so a
//! session that already ran keeps its agent. Every resolution asserted below therefore targets a
//! FRESH origin scope (a fresh `PerThread` session), isolating the registry's answer from the
//! stickiness of previously-opened sessions.

use super::harness::*;

use daemon_api::{BoundAccount, Outbound, ProfileApi, ProfileSpec, ProviderSelector, SessionApi};
use daemon_common::ReqId;
use daemon_host::{MemProfileStore, ProfileStore};
use daemon_protocol::{AgentCommand, AgentEvent, Origin, OriginScope, TransportId, UserMsg};

/// An echoing resolver: the reply reveals which profile (agent) ran the session.
fn echo_resolver() -> daemon_node::ProviderResolver {
    Arc::new(|spec: &ProfileSpec| {
        let reply = format!("[{}]", spec.id);
        let builder: daemon_core::ProviderBuilder = Arc::new(move || {
            Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
        });
        builder
    })
}

/// Assemble a node over `profiles` with the echo resolver and NO config routing table, exactly as
/// `bins/daemon` does when `[routing]` is empty — the account→profile baseline must then be wholly
/// profile-derived (and hot-reloadable).
fn assemble_with_profiles(
    profiles: Arc<MemProfileStore>,
) -> (Arc<NodeApiImpl>, daemon_host::SupervisorHandle) {
    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("alpha"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x66; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(profiles),
        provider_resolver: Some(echo_resolver()),
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
    });
    (node, handle)
}

/// Drive a routed submit for `origin` and return the finished turn's final text (which agent ran).
async fn route_text(node: &Arc<NodeApiImpl>, origin: Origin) -> String {
    let session = node
        .submit_routed(
            origin,
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
        )
        .await
        .expect("routed submit");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        for item in node.poll(session.clone(), 0).await.expect("poll") {
            if let Outbound::Event(AgentEvent::TurnFinished { summary, .. }) = item {
                return summary.final_text.unwrap_or_default();
            }
        }
        assert!(Instant::now() < deadline, "routed turn never finished");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A group-chat origin on `account`, scoped to `chat` (each asserted resolution uses a fresh chat).
fn origin(account: &str, chat: &str) -> Origin {
    Origin::new(
        TransportId::new(format!("matrix/{account}")),
        OriginScope::Group {
            chat: chat.into(),
            thread: None,
        },
    )
}

/// BIND hot-reload: a profile that gains a `BoundAccount` after boot must route that account's new
/// inbound to itself without a restart. RED while `install_routing` snapshots `instance_profiles`
/// at assembly (the post-boot `rebuild_routing()` reclones the stale base); GREEN once the rebuild
/// hook recomputes the baseline from the live profile store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bound_account_bind_hot_reloads_instance_routing() {
    as_system(bound_account_bind_hot_reloads_instance_routing_impl()).await;
}
async fn bound_account_bind_hot_reloads_instance_routing_impl() {
    // Two profiles, NEITHER with a bound account at boot (the boot-time derived baseline is empty).
    let profiles = Arc::new(MemProfileStore::new());
    profiles
        .create(ProfileSpec::new(
            "alpha",
            ProviderSelector::GenAi,
            "model-a",
        ))
        .expect("create alpha");
    profiles
        .create(ProfileSpec::new("beta", ProviderSelector::GenAi, "model-b"))
        .expect("create beta");
    profiles.set_active("alpha").expect("set active");

    let (node, handle) = assemble_with_profiles(profiles);

    // Sanity: with no binding, the account's inbound falls to the active default (alpha).
    let before = route_text(&node, origin("@b:hs", "#before")).await;
    assert!(
        before.contains("[alpha]"),
        "unbound account -> active default, got {before:?}"
    );

    // BIND after boot: beta declares the account through the wire `profile_update` op.
    let mut beta = node
        .profile_get("beta".into())
        .await
        .expect("profile_get")
        .expect("beta exists");
    beta.bound_accounts = vec![BoundAccount::new("matrix/@b:hs", "matrix/beta/b")];
    node.profile_update(beta).await.expect("profile_update");

    // A FRESH chat on the account must now resolve to beta — with no restart.
    let after = route_text(&node, origin("@b:hs", "#after")).await;
    assert!(
        after.contains("[beta]"),
        "binding a bound_account must hot-reload instance routing (expected [beta]), got {after:?}"
    );

    handle.shutdown().await;
}

/// UNBIND hot-reload (EIO-7's disconnect leg): a profile that LOSES its `BoundAccount` must stop
/// receiving that account's new inbound without a restart — stale routes must not keep delivering.
/// RED while the boot-time snapshot keeps the removed binding alive; GREEN with the rebuild hook.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bound_account_unbind_hot_reloads_instance_routing() {
    as_system(bound_account_unbind_hot_reloads_instance_routing_impl()).await;
}
async fn bound_account_unbind_hot_reloads_instance_routing_impl() {
    // beta declares its account AT BOOT, so the assembly-time derivation binds it.
    let profiles = Arc::new(MemProfileStore::new());
    profiles
        .create(ProfileSpec::new(
            "alpha",
            ProviderSelector::GenAi,
            "model-a",
        ))
        .expect("create alpha");
    profiles
        .create(
            ProfileSpec::new("beta", ProviderSelector::GenAi, "model-b")
                .with_bound_accounts(vec![BoundAccount::new("matrix/@c:hs", "matrix/beta/c")]),
        )
        .expect("create beta");
    profiles.set_active("alpha").expect("set active");

    let (node, handle) = assemble_with_profiles(profiles);

    // Sanity: the boot-time binding routes the account to beta.
    let bound = route_text(&node, origin("@c:hs", "#one")).await;
    assert!(
        bound.contains("[beta]"),
        "boot-declared bound_account -> beta, got {bound:?}"
    );

    // UNBIND after boot: beta drops the account through the wire `profile_update` op.
    let mut beta = node
        .profile_get("beta".into())
        .await
        .expect("profile_get")
        .expect("beta exists");
    beta.bound_accounts = Vec::new();
    node.profile_update(beta).await.expect("profile_update");

    // A FRESH chat on the account must fall back to the active default — the stale instance
    // binding must not keep delivering to beta until a restart.
    let after = route_text(&node, origin("@c:hs", "#two")).await;
    assert!(
        after.contains("[alpha]"),
        "unbinding a bound_account must hot-reload instance routing (expected [alpha] fallback), \
         got {after:?}"
    );

    handle.shutdown().await;
}
