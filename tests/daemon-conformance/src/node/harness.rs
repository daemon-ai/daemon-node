// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared assembly + socket glue for the node control-surface conformance suite.

pub use daemon_api::{ApiRequest, ApiResponse, ControlApi, SessionState};
pub use daemon_common::{PartitionId, ProfileRef, SessionId};
pub use daemon_core::{MockProvider, Provider, ProviderRegistry};
pub use daemon_host::{serve_api_unix, ApiClient, HostConfig, NodeApiImpl};
pub use daemon_node::{assemble as assemble_node, AssembledNode, NodeAssembly};
pub use daemon_store::{InMemoryStore, SessionStore, SqliteStore};
pub use std::sync::atomic::{AtomicU64, Ordering};
pub use std::sync::Arc;
pub use std::time::{Duration, Instant};
pub use tokio::net::UnixListener;

pub const PARTITION: PartitionId = PartitionId::DEFAULT;

/// Assemble a node through the shared composition root ([`daemon_node::assemble`]) — exactly as
/// `bins/daemon`'s host role does — with the gate's mock providers (an orchestrator that
/// delegates once, completing children, and a completing session default). Returns the in-process
/// surface and the started resident-service handle.
/// The gate's mock provider registry: an orchestrator that delegates once per turn (driving the
/// recursive durable delegation chain, bounded by the orchestrate-tool depth guard), a completing
/// session default, and a legacy `child` provider for the synchronous foreign fallback.
pub fn gate_providers() -> ProviderRegistry {
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    providers.register(
        "orchestrator",
        Arc::new(|| {
            Arc::new(MockProvider::delegating("orchestrate", "fleet done")) as Arc<dyn Provider>
        }),
    );
    providers.register(
        "child",
        Arc::new(|| Arc::new(MockProvider::completing("child done")) as Arc<dyn Provider>),
    );
    providers
}

/// Assemble a node over a caller-supplied durable `store` (so two nodes can share one store to
/// simulate a crash/restart), with `host_config` cadence and a delegation depth cap of
/// `nesting_depth + 1` (see [`daemon_node::assemble`]).
pub fn assemble_over(
    store: Arc<dyn SessionStore>,
    nesting_depth: usize,
    journal_seed: [u8; 32],
    host_config: HostConfig,
) -> AssembledNode {
    assemble_node(NodeAssembly {
        store,
        partition: PARTITION,
        host_config,
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some(journal_seed),
        nesting_depth,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: None,
        provider_resolver: None,
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
    })
}

/// The default resident-service cadence for the gate (fast ticks).
pub fn fast_host_config() -> HostConfig {
    HostConfig {
        partition: PARTITION,
        ..HostConfig::default()
    }
}

pub fn assemble() -> (Arc<NodeApiImpl>, daemon_host::SupervisorHandle) {
    let AssembledNode { node, handle, .. } = assemble_over(
        Arc::new(InMemoryStore::new()),
        0,
        [0x11; 32],
        fast_host_config(),
    );
    (node, handle)
}

/// Assemble a node whose orchestrate-tool depth cap allows `depth + 1` levels of nested durable
/// delegation, so the management tree the GUI projects is genuinely recursive (top -> child ->
/// ... -> leaf). The durable orchestrator delegates once per level; the deepest level completes.
pub fn assemble_nested(depth: usize) -> (Arc<NodeApiImpl>, daemon_host::SupervisorHandle) {
    let AssembledNode { node, handle, .. } = assemble_over(
        Arc::new(InMemoryStore::new()),
        depth,
        [0x22; 32],
        fast_host_config(),
    );
    (node, handle)
}

/// Assemble a node wired for the **Phase 0 GUI-readiness demo gate**: a profile store + a
/// provider resolver (a hermetic mock standing in for the real GenAI client) + a credential
/// store, so the profile/credential/model/session surfaces are all live over one socket. The
/// resolver echoes the active profile's persona so the demo proves the per-session profile
/// resolution path (not a fixed launch profile).
pub fn assemble_demo() -> (Arc<NodeApiImpl>, daemon_host::SupervisorHandle) {
    use daemon_host::{MemCredentialStore, MemProfileStore};
    let resolver: daemon_node::ProviderResolver = Arc::new(|spec: &daemon_api::ProfileSpec| {
        let reply = format!("[{}] hello from {}", spec.id, spec.model);
        let builder: daemon_core::ProviderBuilder = Arc::new(move || {
            Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
        });
        builder
    });
    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x44; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(Arc::new(MemProfileStore::new())),
        provider_resolver: Some(resolver),
        credential_store: Some(Arc::new(MemCredentialStore::new())),
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
    });
    (node, handle)
}

/// Assemble a node whose per-session provider is the **real** `genai` OpenAI adapter (the
/// `DaemonApi` gateway path), pointed at a caller-supplied mock upstream `base_url`, with a working
/// credential broker + profile store — mirroring how `bins/daemon`'s `run_as_host` wires the live
/// agent (`build_multi_profile_broker` + `provider_resolver` + `credential_store` + `profiles`).
///
/// The top-level provider registry stays mock (`gate_providers`), so only the turn's provider hits
/// the upstream (LCM-aux / mnemosyne summarization never do — no context/memory builders are wired).
/// Returns the in-process surface, the shared credential store (so a test can seed/inspect keys
/// directly, e.g. the local-trust path), and the started resident-service handle.
pub fn assemble_daemon_api_gateway(
    base_url: String,
) -> (
    Arc<NodeApiImpl>,
    Arc<dyn daemon_host::CredentialStore>,
    daemon_host::SupervisorHandle,
) {
    assemble_daemon_api_gateway_inner(base_url, None)
}

/// As [`assemble_daemon_api_gateway`], but also wires a `CloudCatalog` discovery hook so the same
/// node serves the setup-picker ops (`ProviderCatalog` / `ProviderModels`) alongside the live turn
/// path. Used by the Track 5 discovery→configure→chat e2e, where the mock upstream doubles as the
/// Daemon Cloud gateway (serving both `GET /models` and `POST /chat/completions`).
pub fn assemble_daemon_api_gateway_with_catalog(
    base_url: String,
    catalog: Arc<dyn daemon_host::CloudCatalog>,
) -> (
    Arc<NodeApiImpl>,
    Arc<dyn daemon_host::CredentialStore>,
    daemon_host::SupervisorHandle,
) {
    assemble_daemon_api_gateway_inner(base_url, Some(catalog))
}

fn assemble_daemon_api_gateway_inner(
    base_url: String,
    cloud_catalog: Option<Arc<dyn daemon_host::CloudCatalog>>,
) -> (
    Arc<NodeApiImpl>,
    Arc<dyn daemon_host::CredentialStore>,
    daemon_host::SupervisorHandle,
) {
    use daemon_api::{ProfileSpec, ProviderSelector};
    use daemon_common::CredMode;
    use daemon_core::{CredentialBuilder, CredentialProvider, ProviderBuilder};
    use daemon_credentials::CapabilitySigner;
    use daemon_host::{
        BrokeredCredentialProvider, CredentialBroker, CredentialStore, MemCredentialStore,
        MemProfileStore, MultiProfileStoreBroker,
    };
    use daemon_providers::GenAiProvider;

    // A shared credential store: a wire `CredentialSet` (or a direct `.set`) writes here, and the
    // per-profile broker leases the stored key into `Request.auth` (Bearer). Mirrors `run_as_host`.
    let cred_store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let broker: Arc<MultiProfileStoreBroker> = Arc::new(MultiProfileStoreBroker::new(
        cred_store.clone(),
        Arc::new(CapabilitySigner::generate()),
        "", // no launch fallback key: the test provisions the profile key explicitly
        ["chat", "embed"],
        Some(1_000),
        CredMode::Bearer,
        60_000,
    ));
    let owner: Arc<dyn CredentialBroker> = broker;
    let credentials: CredentialBuilder = Arc::new(move || {
        Arc::new(BrokeredCredentialProvider::new(owner.clone(), None))
            as Arc<dyn CredentialProvider>
    });

    // The per-session resolver: a `DaemonApi` profile builds the Daemon Cloud provider (genai's
    // OpenAI-compatible adapter pinned at the profile's base URL, falling back to the mock upstream);
    // every other selector stays a mock. `GenAiProvider::daemon_cloud` is the named Daemon Cloud
    // constructor — an OpenAI adapter + endpoint override, byte-identical on the wire.
    let resolver: daemon_node::ProviderResolver = {
        let default_base = base_url;
        Arc::new(move |spec: &ProfileSpec| match spec.provider {
            ProviderSelector::DaemonApi => {
                let base = spec
                    .base_url
                    .clone()
                    .unwrap_or_else(|| default_base.clone());
                let model = spec.model.clone();
                let builder: ProviderBuilder = Arc::new(move || {
                    Arc::new(GenAiProvider::daemon_cloud(model.clone()).with_endpoint(base.clone()))
                        as Arc<dyn Provider>
                });
                builder
            }
            _ => {
                let reply = format!("[{}] mock reply", spec.id);
                let builder: ProviderBuilder = Arc::new(move || {
                    Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
                });
                builder
            }
        })
    };

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: Some(credentials),
        profile: ProfileRef::new("gateway"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x55; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(Arc::new(MemProfileStore::new())),
        provider_resolver: Some(resolver),
        credential_store: Some(cred_store.clone()),
        cloud_catalog,
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
    });
    (node, cred_store, handle)
}

/// `ConvGet` helper: fetch a conversation and unwrap it (panics if absent).
pub async fn conv_get(
    client: &ApiClient,
    transport: &daemon_protocol::TransportId,
    conv: &str,
) -> daemon_api::ConversationInfo {
    match client
        .call(ApiRequest::ConvGet {
            transport: transport.clone(),
            conv: conv.to_string(),
        })
        .await
        .unwrap()
    {
        ApiResponse::Conversation(Some(info)) => info,
        other => panic!("expected Conversation, got {other:?}"),
    }
}

pub fn temp_socket() -> std::path::PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("daemon-api-gate-{}-{}.sock", std::process::id(), n))
}
