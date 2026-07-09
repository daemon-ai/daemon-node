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
    CustomProvider, CustomProviderSource, ModelApi, ModelDescriptor, ProviderDescriptor,
    ProviderKindWire, ProviderSelector,
};
use daemon_host::{CloudCatalog, CredentialStore, MemCredentialStore, MemProfileStore};
use daemon_store::{CustomProviderRecord, InMemoryStore, SessionStore};
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
                sign_in: None,
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
                sign_in: None,
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

/// The last `(base_url, key)` an `openai_compat_models` call recorded.
type LastOpenAiCall = Arc<Mutex<Option<(String, Option<String>)>>>;

/// A custom-provider probe: records the (base_url, key) handed to `openai_compat_models` so a test
/// can assert credential-aware routing, and returns one synthesized model. Its `providers()` is
/// empty (custom rows come from the store overlay, not the injected hook).
struct OpenAiProbe {
    last: LastOpenAiCall,
}

#[async_trait::async_trait]
impl CloudCatalog for OpenAiProbe {
    async fn list(&self) -> Vec<ModelDescriptor> {
        Vec::new()
    }
    async fn providers(&self) -> Vec<ProviderDescriptor> {
        Vec::new()
    }
    async fn provider_models(&self, _: &str, _: Option<String>) -> Vec<ModelDescriptor> {
        Vec::new()
    }
    async fn openai_compat_models(
        &self,
        base_url: &str,
        key: Option<String>,
    ) -> Vec<ModelDescriptor> {
        *self.last.lock().unwrap() = Some((base_url.to_string(), key.clone()));
        vec![ModelDescriptor {
            id: "gw/model-1".into(),
            provider: ProviderSelector::DaemonApi,
            display_name: None,
            context_length: None,
            input_price_micros_per_mtok: None,
            output_price_micros_per_mtok: None,
            local: false,
        }]
    }
}

/// A wire `CustomProvider` (`source = User`) for seeding the store in tests.
fn custom(id: &str, name: &str, base: &str) -> CustomProvider {
    CustomProvider {
        id: id.into(),
        display_name: name.into(),
        base_url: base.into(),
        wire_selector: ProviderSelector::DaemonApi,
        requires_key: true,
        credential_ref: None,
        source: CustomProviderSource::User,
    }
}

/// Upsert a wire `CustomProvider` straight into the durable store (the seeding a boot / a wire set
/// performs), so `provider_catalog`/`provider_models` overlay it.
async fn seed_custom(store: &Arc<InMemoryStore>, p: &CustomProvider) {
    store
        .custom_provider_set(CustomProviderRecord {
            id: p.id.clone(),
            entry: daemon_api::to_cbor(p),
        })
        .await
        .unwrap();
}

/// Assemble a node with a discovery hook + credential store, mirroring `run_as_host`'s wiring.
fn assemble_with_catalog(
    catalog: Arc<dyn CloudCatalog>,
    creds: Arc<dyn CredentialStore>,
) -> Arc<NodeApiImpl> {
    assemble_with_catalog_store(catalog, creds, Arc::new(InMemoryStore::new()))
}

/// As [`assemble_with_catalog`], but over a caller-supplied store so a test can pre-seed custom
/// providers (the durable half of the provider catalog).
fn assemble_with_catalog_store(
    catalog: Arc<dyn CloudCatalog>,
    creds: Arc<dyn CredentialStore>,
    store: Arc<InMemoryStore>,
) -> Arc<NodeApiImpl> {
    let AssembledNode { node, .. } = assemble_node(NodeAssembly {
        store,
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
        fs: Default::default(),
        processes: Default::default(),
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
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
            None,
        )
        .await;
    assert_eq!(models.items.len(), 1);
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
        .provider_models("anthropic".into(), Some("cred-1".into()), None, None)
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
        .provider_models("daemon_cloud".into(), None, None, None)
        .await;
    assert_eq!(
        *last_key.lock().unwrap(),
        Some(None),
        "Daemon Cloud lists keyless (no LIST credential)"
    );
}

/// Vision-projector (mmproj) records never surface as chat models: `ProviderModels` for the local
/// engine and the merged `models` page both exclude them, and `ModelActivate` rejects them with
/// the actionable error — while the text record stays offered and activatable.
#[tokio::test]
async fn projector_records_are_excluded_and_activate_rejects() {
    use daemon_common::{InstalledModel, ModelEngine, ModelRef, ModelSource};

    // Seed a registry with one text model and one projector (arch=clip + mmproj name), then open
    // a real ModelManager over it (no network is touched by catalog/activate paths).
    let dir = std::env::temp_dir().join(format!("daemon-conf-projector-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let registry_path = dir.join("catalog.json");
    let text_ref = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::hf_file("org/smolvlm", "SmolVLM-256M-Instruct-f16.gguf"),
    );
    let proj_ref = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::hf_file("org/smolvlm", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"),
    );
    let record = |model: &ModelRef, name: &str, arch: &str| InstalledModel {
        id: daemon_models::model_id(model),
        model: model.clone(),
        display_name: name.into(),
        local_path: dir.join(name),
        size_bytes: 1,
        quant: None,
        installed_at_ms: 1,
        arch: Some(arch.into()),
        context_length: None,
        file_type: None,
        mmproj_path: None,
        sha256: None,
    };
    let text_id = daemon_models::model_id(&text_ref);
    let proj_id = daemon_models::model_id(&proj_ref);
    {
        let registry = daemon_models::Registry::open(&registry_path).await.unwrap();
        registry
            .upsert(record(&text_ref, "SmolVLM-256M-Instruct-f16.gguf", "llama"))
            .await
            .unwrap();
        registry
            .upsert(record(
                &proj_ref,
                "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
                "clip",
            ))
            .await
            .unwrap();
    }
    let manager = daemon_models::ModelManager::new(daemon_models::ManagerConfig {
        cache_dir: Some(dir.join("hub")),
        fallback_cache_dir: None,
        registry_path: Some(registry_path),
        endpoint: None,
        quantize_worker_bin: None,
    })
    .await
    .expect("manager");

    let AssembledNode { node, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: daemon_common::ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x68; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: Some(Arc::new(manager)),
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
        fs: Default::default(),
        processes: Default::default(),
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
    });

    // The local chat offer carries the text model but never the projector.
    let offered = node
        .provider_models("llama_cpp".into(), None, None, None)
        .await;
    let ids: Vec<&str> = offered.items.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains(&text_id.as_str()),
        "text model offered: {ids:?}"
    );
    assert!(
        !ids.contains(&proj_id.as_str()),
        "projector must not be offered: {ids:?}"
    );

    // The merged models page excludes it too.
    let page = node.models(None).await;
    assert!(page.items.iter().all(|m| m.id != proj_id.as_str()));

    // Activating the projector fails with the actionable error; the text model activates.
    let err = node
        .model_activate(proj_id.clone(), None)
        .await
        .expect_err("projector activation must be rejected");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("vision projector") && msg.contains("text weights"),
        "actionable message expected, got: {msg}"
    );
    node.model_activate(text_id, None)
        .await
        .expect("text model activates");

    // The projector stays in the raw catalog (inventory / uninstall surface).
    let catalog = node.model_catalog().await;
    assert!(catalog.iter().any(|m| m.id == proj_id));

    let _ = std::fs::remove_dir_all(&dir);
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
        fs: Default::default(),
        processes: Default::default(),
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
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

/// Persisted custom providers overlay onto the merged `provider_catalog` read: a new id appears as a
/// Daemon-Cloud-kind row carrying its base URL, and a custom id that collides with a builtin wins
/// (mirrors `agent_catalog`'s manual-over-builtin precedence).
#[tokio::test]
async fn custom_providers_merge_into_catalog_and_win_dedup() {
    let store = Arc::new(InMemoryStore::new());
    seed_custom(
        &store,
        &custom("custom/gw", "My Gateway", "https://gw.example/v1/"),
    )
    .await;
    // Collides with the builtin genai "anthropic" row → custom must win.
    seed_custom(
        &store,
        &custom("anthropic", "My Anthropic Proxy", "https://ap.example/v1/"),
    )
    .await;
    let catalog = Arc::new(RecordingCatalog {
        last_key: Arc::new(Mutex::new(None)),
    });
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let node = assemble_with_catalog_store(catalog, creds, store);

    let providers = node.provider_catalog().await;
    let by_id = |id: &str| providers.iter().find(|p| p.id == id).cloned();

    let gw = by_id("custom/gw").expect("new custom row present");
    assert_eq!(gw.kind, ProviderKindWire::DaemonCloud);
    assert_eq!(gw.wire_selector, ProviderSelector::DaemonApi);
    assert_eq!(
        gw.default_base_url.as_deref(),
        Some("https://gw.example/v1/")
    );

    let anth = by_id("anthropic").expect("anthropic row present");
    assert_eq!(
        anth.kind,
        ProviderKindWire::DaemonCloud,
        "custom wins the id collision"
    );
    assert_eq!(
        anth.default_base_url.as_deref(),
        Some("https://ap.example/v1/")
    );
    assert_eq!(
        providers.iter().filter(|p| p.id == "anthropic").count(),
        1,
        "exactly one anthropic row after dedup"
    );
}

/// `provider_models` for a custom id routes to `openai_compat_models` with the provider's base URL
/// and a credential-aware bearer: a transient key wins, else the provider's own `credential_ref` is
/// resolved through the credential store.
#[tokio::test]
async fn custom_provider_models_route_to_openai_compat_credential_aware() {
    let store = Arc::new(InMemoryStore::new());
    let mut p = custom("custom/gw", "GW", "https://gw.example/v1/");
    p.credential_ref = Some("cred-gw".into());
    seed_custom(&store, &p).await;

    let last = Arc::new(Mutex::new(None));
    let catalog = Arc::new(OpenAiProbe { last: last.clone() });
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    creds.set("cred-gw", "sk-gw-secret").unwrap();
    let node = assemble_with_catalog_store(catalog, creds, store);

    // No transient key: the provider's default credential_ref is resolved through the store.
    let models = node
        .provider_models("custom/gw".into(), None, None, None)
        .await;
    assert_eq!(models.items.len(), 1);
    assert_eq!(models.items[0].provider, ProviderSelector::DaemonApi);
    let (base, key) = last.lock().unwrap().clone().expect("openai_compat called");
    assert_eq!(base, "https://gw.example/v1/");
    assert_eq!(
        key.as_deref(),
        Some("sk-gw-secret"),
        "resolved the provider's credential_ref"
    );

    // A first-run transient key wins over the stored credential.
    let _ = node
        .provider_models("custom/gw".into(), None, Some("transient-xyz".into()), None)
        .await;
    let (_, key) = last.lock().unwrap().clone().unwrap();
    assert_eq!(key.as_deref(), Some("transient-xyz"), "transient key wins");
}

/// Wire `custom_provider_set` forces `source = User` + the `DaemonApi` selector and validates the
/// base URL; `list` reflects the write; `remove` deletes a user entry but refuses a config-seeded
/// one (config is authoritative for its ids, re-seeded each boot).
#[tokio::test]
async fn custom_provider_wire_crud_forces_user_and_guards_config() {
    let store = Arc::new(InMemoryStore::new());
    // A config-seeded entry pre-exists in the store.
    let mut seeded = custom("custom/seeded", "Seeded", "https://seed.example/v1/");
    seeded.source = CustomProviderSource::Config;
    seed_custom(&store, &seeded).await;

    let catalog = Arc::new(RecordingCatalog {
        last_key: Arc::new(Mutex::new(None)),
    });
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let node = assemble_with_catalog_store(catalog, creds, store);

    // A wire set with a bogus selector/source is normalized to User + DaemonApi.
    let mut incoming = custom("custom/gw", "GW", "https://gw.example/v1/");
    incoming.wire_selector = ProviderSelector::Mock;
    incoming.source = CustomProviderSource::Config;
    node.custom_provider_set(incoming).await.unwrap();
    let list = node.custom_provider_list().await;
    let got = list.iter().find(|p| p.id == "custom/gw").expect("listed");
    assert_eq!(got.wire_selector, ProviderSelector::DaemonApi);
    assert!(matches!(got.source, CustomProviderSource::User));

    // Base-URL validation rejects empty + non-http(s).
    assert!(node
        .custom_provider_set(custom("bad", "B", ""))
        .await
        .is_err());
    assert!(node
        .custom_provider_set(custom("bad2", "B", "ftp://x"))
        .await
        .is_err());

    // Remove a user entry, but refuse the config-seeded one.
    node.custom_provider_remove("custom/gw".into())
        .await
        .unwrap();
    assert!(node
        .custom_provider_list()
        .await
        .iter()
        .all(|p| p.id != "custom/gw"));
    assert!(
        node.custom_provider_remove("custom/seeded".into())
            .await
            .is_err(),
        "config-seeded entry is not wire-removable"
    );
    assert!(
        node.custom_provider_list()
            .await
            .iter()
            .any(|p| p.id == "custom/seeded"),
        "config-seeded entry survives the refused remove"
    );
}
