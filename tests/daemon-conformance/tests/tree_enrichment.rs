// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Fleet tree enrichment (wire v29, F3): `UnitNode` carries the declared delegation `lifetime`
//! (server-derived from the durable role, so clients never re-implement the role->lifetime rule)
//! and the bound profile's `engine` selector (denormalized so a tree render needs no per-node
//! `ProfileGet`) — proven through the real durable projection (`Unit`/`Tree` over session meta +
//! the profile store).

use std::sync::Arc;

use daemon_api::{
    AgentEntry, AgentProtocol, AgentRecipe, AgentSource, AgentVerification, ControlApi,
    EngineSelector, ProfileApi, ProfileSpec, ProviderSelector,
};
use daemon_common::{PartitionId, ProfileRef, SessionId, UnitId};
use daemon_core::{MockProvider, Provider, ProviderBuilder, ProviderRegistry, Snapshot};
use daemon_host::{HostConfig, MemProfileStore, NodeApiImpl};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_protocol::DelegationLifetime;
use daemon_store::{InMemoryStore, SessionRole, SessionStore};

fn assemble_with_profiles(
    store: Arc<dyn SessionStore>,
) -> (Arc<NodeApiImpl>, daemon_host::SupervisorHandle) {
    let resolver: ProviderResolver = Arc::new(|_spec: &ProfileSpec| {
        let builder: ProviderBuilder =
            Arc::new(|| Arc::new(MockProvider::completing("done")) as Arc<dyn Provider>);
        builder
    });
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("done")) as Arc<dyn Provider>
    }));
    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store,
        partition: PartitionId::DEFAULT,
        host_config: HostConfig::default(),
        providers,
        credentials: None,
        profile: ProfileRef::new("default"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x49; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(Arc::new(MemProfileStore::new())),
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

/// Seed a durable session row with the given meta (the same stamping the node's own
/// child-creation paths perform).
async fn seed_session(
    store: &Arc<dyn SessionStore>,
    id: &SessionId,
    role: Option<SessionRole>,
    profile: Option<&str>,
) {
    let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
    store
        .create_session(id.clone(), PartitionId::DEFAULT, blob)
        .await
        .expect("create session");
    let mut meta = store.session_meta(id).await.unwrap_or_default();
    meta.role = role;
    meta.bound_profile = profile.map(ProfileRef::new);
    store.set_session_meta(id, meta).await.expect("set meta");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tree_nodes_carry_lifetime_and_engine() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        tree_nodes_carry_lifetime_and_engine_impl(),
    )
    .await;
}
async fn tree_nodes_carry_lifetime_and_engine_impl() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let (node, _handle) = assemble_with_profiles(store.clone());

    // A foreign-engine profile bound to a stream-json agent registered BY NAME (the PATH-only
    // probe marks the compiled mock installed, so profile validation passes).
    node.agent_register(AgentEntry {
        name: "sj".into(),
        recipe: AgentRecipe {
            program: Some(env!("CARGO_BIN_EXE_mock_stream_json_agent").to_string()),
            args: Vec::new(),
            env: Vec::new(),
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::StreamJson,
        installed: false,
        version: None,
        capabilities: Vec::new(),
        verification: AgentVerification::NotInstalled, // untrusted; the node re-derives on register
    })
    .await
    .expect("register the mock stream-json agent");
    node.profile_create(ProfileSpec {
        engine: EngineSelector::Foreign { agent: "sj".into() },
        ..ProfileSpec::new("sj-profile", ProviderSelector::Mock, "")
    })
    .await
    .expect("create the foreign profile");
    node.profile_create(ProfileSpec::new(
        "core-profile",
        ProviderSelector::Mock,
        "mock-model",
    ))
    .await
    .expect("create the core profile");

    // Three durable sessions, stamped exactly as the node's own paths do: a primary (no role, no
    // profile), a persistent managed child on the Core profile, an ephemeral subagent on the
    // foreign profile.
    let primary = SessionId::new("t-primary");
    let managed = SessionId::new("t-primary/c1");
    let ephemeral = SessionId::new("t-primary/d1");
    seed_session(&store, &primary, None, None).await;
    seed_session(
        &store,
        &managed,
        Some(SessionRole::ManagedChild),
        Some("core-profile"),
    )
    .await;
    seed_session(
        &store,
        &ephemeral,
        Some(SessionRole::EphemeralSubagent),
        Some("sj-profile"),
    )
    .await;

    // Per-unit reads carry the enrichment.
    let unit = |id: &SessionId| node.unit(UnitId::new(id.as_str()));
    let p = unit(&primary).await.expect("primary node");
    assert_eq!(p.lifetime, None, "a primary unit declares no lifetime");
    assert_eq!(p.engine, None, "an unbound unit denormalizes no engine");

    let m = unit(&managed).await.expect("managed node");
    assert_eq!(m.lifetime, Some(DelegationLifetime::Persistent));
    assert_eq!(m.engine, Some(EngineSelector::Core));

    let e = unit(&ephemeral).await.expect("ephemeral node");
    assert_eq!(e.lifetime, Some(DelegationLifetime::Ephemeral));
    assert_eq!(
        e.engine,
        Some(EngineSelector::Foreign { agent: "sj".into() }),
        "the bound profile's engine selector is denormalized onto the tree node"
    );

    // The paged tree read agrees (same projection).
    let tree = node.tree(None).await;
    let from_tree = tree
        .nodes
        .iter()
        .find(|n| n.session.as_ref() == Some(&ephemeral))
        .expect("the ephemeral child is on the tree");
    assert_eq!(from_tree.lifetime, Some(DelegationLifetime::Ephemeral));
    assert_eq!(
        from_tree.engine,
        Some(EngineSelector::Foreign { agent: "sj".into() })
    );
}
