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
    });
    (node, handle)
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
