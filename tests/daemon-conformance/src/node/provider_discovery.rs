// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Provider + model discovery (Track 2) node-wiring conformance.
//!
//! Proves the node surfaces the injected provider catalog and threads the LIST credential correctly
//! for `ProviderModels` (a first-run `transient_key` wins; else the stored `credential_ref` is
//! resolved through the credential store; Daemon Cloud lists keyless), and that an *unconfigured*
//! node (no discovery hook wired) still lists providers via the static fallback. The genai/Daemon
//! Cloud network specifics are covered hermetically in `bins/daemon`'s own unit tests.

use super::harness::{
    assemble_node, fast_host_config, gate_providers, AssembledNode, NodeApiImpl, NodeAssembly,
    PARTITION,
};
use daemon_api::{
    ModelApi, ModelDescriptor, ProviderDescriptor, ProviderKindWire, ProviderSelector,
};
use daemon_host::{CloudCatalog, CredentialStore, MemCredentialStore, MemProfileStore};
use daemon_store::InMemoryStore;
use std::sync::{Arc, Mutex};

/// A fake discovery hook: records the key handed to `provider_models` (so the test can assert the
/// transient/stored resolution) and returns a fixed provider list + one synthesized model per call.
struct RecordingCatalog {
    last_key: Arc<Mutex<Option<Option<String>>>>,
}

#[async_trait::async_trait]
impl CloudCatalog for RecordingCatalog {
    async fn list(&self) -> Vec<ModelDescriptor> {
        Vec::new()
    }

    async fn providers(&self) -> Vec<ProviderDescriptor> {
        vec![
            ProviderDescriptor {
                id: "anthropic".into(),
                display_name: "Anthropic".into(),
                kind: ProviderKindWire::Cloud,
                wire_selector: ProviderSelector::GenAi,
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: None,
            },
            ProviderDescriptor {
                id: "daemon_cloud".into(),
                display_name: "Daemon Cloud".into(),
                kind: ProviderKindWire::DaemonCloud,
                wire_selector: ProviderSelector::DaemonApi,
                // Needs a key to run turns; LISTING stays keyless (asserted below).
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: Some("https://api.daemon.ai/api/v1/".into()),
            },
        ]
    }

    async fn provider_models(
        &self,
        provider_id: &str,
        key: Option<String>,
    ) -> Vec<ModelDescriptor> {
        *self.last_key.lock().unwrap() = Some(key.clone());
        vec![ModelDescriptor {
            id: format!("{provider_id}/model-1"),
            provider: ProviderSelector::GenAi,
            display_name: None,
            context_length: None,
            input_price_micros_per_mtok: None,
            output_price_micros_per_mtok: None,
            local: false,
        }]
    }
}

/// Assemble a node with a discovery hook + credential store, mirroring `run_as_host`'s wiring.
fn assemble_with_catalog(
    catalog: Arc<dyn CloudCatalog>,
    creds: Arc<dyn CredentialStore>,
) -> Arc<NodeApiImpl> {
    let AssembledNode { node, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: daemon_common::ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x66; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(Arc::new(MemProfileStore::new())),
        provider_resolver: None,
        credential_store: Some(creds),
        cloud_catalog: Some(catalog),
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
    node
}

#[tokio::test]
async fn provider_catalog_surfaces_the_injected_hook() {
    let last_key = Arc::new(Mutex::new(None));
    let catalog = Arc::new(RecordingCatalog {
        last_key: last_key.clone(),
    });
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let node = assemble_with_catalog(catalog, creds);

    let providers = node.provider_catalog().await;
    let ids: Vec<&str> = providers.iter().map(|p| p.id.as_str()).collect();
    assert!(ids.contains(&"anthropic"), "catalog surfaced: {ids:?}");
    assert!(ids.contains(&"daemon_cloud"), "catalog surfaced: {ids:?}");
}

#[tokio::test]
async fn provider_models_prefers_the_transient_key() {
    let last_key = Arc::new(Mutex::new(None));
    let catalog = Arc::new(RecordingCatalog {
        last_key: last_key.clone(),
    });
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    // A stored credential exists, but the first-run transient key must win.
    creds.set("cred-1", "stored-abc").unwrap();
    let node = assemble_with_catalog(catalog, creds);

    let models = node
        .provider_models(
            "anthropic".into(),
            Some("cred-1".into()),
            Some("transient-xyz".into()),
        )
        .await;
    assert_eq!(models.len(), 1);
    assert_eq!(
        *last_key.lock().unwrap(),
        Some(Some("transient-xyz".into())),
        "the transient key must authenticate the LIST call"
    );
}

#[tokio::test]
async fn provider_models_resolves_the_credential_ref() {
    let last_key = Arc::new(Mutex::new(None));
    let catalog = Arc::new(RecordingCatalog {
        last_key: last_key.clone(),
    });
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    creds.set("cred-1", "stored-abc").unwrap();
    let node = assemble_with_catalog(catalog, creds);

    // No transient key: the stored credential the ref points at is resolved and passed through.
    let _ = node
        .provider_models("anthropic".into(), Some("cred-1".into()), None)
        .await;
    assert_eq!(
        *last_key.lock().unwrap(),
        Some(Some("stored-abc".into())),
        "the stored credential_ref must authenticate the LIST call"
    );
}

#[tokio::test]
async fn provider_models_daemon_cloud_is_keyless() {
    let last_key = Arc::new(Mutex::new(None));
    let catalog = Arc::new(RecordingCatalog {
        last_key: last_key.clone(),
    });
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let node = assemble_with_catalog(catalog, creds);

    let _ = node
        .provider_models("daemon_cloud".into(), None, None)
        .await;
    assert_eq!(
        *last_key.lock().unwrap(),
        Some(None),
        "Daemon Cloud lists keyless (no LIST credential)"
    );
}

/// An unconfigured node with no discovery hook wired still lists providers (the static fallback:
/// local engines + Daemon Cloud), so setup is never blocked.
#[tokio::test]
async fn unconfigured_node_still_lists_providers() {
    let AssembledNode { node, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: daemon_common::ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x67; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(Arc::new(MemProfileStore::new())),
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
    });
    let providers = node.provider_catalog().await;
    let ids: Vec<&str> = providers.iter().map(|p| p.id.as_str()).collect();
    assert!(!providers.is_empty(), "static fallback lists providers");
    assert!(
        ids.contains(&"daemon_cloud"),
        "fallback has Daemon Cloud: {ids:?}"
    );
    assert!(
        ids.contains(&"llama_cpp"),
        "fallback has local engines: {ids:?}"
    );
}
