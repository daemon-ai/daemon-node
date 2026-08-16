// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;
use daemon_common::ModelSource;

#[async_trait]
impl ModelApi for NodeApiImpl {
    async fn model_search(&self, query: SearchQuery) -> Result<SearchPage, ApiError> {
        let m = self.require_models()?;
        let engine = self.resolve_available_engine(&query.provider).await?;
        m.search(engine, query).await.map_err(map_model_err)
    }

    async fn model_files(
        &self,
        provider: String,
        repo: String,
        revision: Option<String>,
        after: Option<String>,
    ) -> Result<daemon_api::WirePage<ModelFile>, ApiError> {
        let m = self.require_models()?;
        let engine = self.resolve_available_engine(&provider).await?;
        // The manager returns the full listing already sorted by `path` (the cursor key).
        let files = m
            .model_files(&repo, revision.as_deref(), engine)
            .await
            .map_err(map_model_err)?;
        Ok(daemon_api::paginate(
            files,
            after.as_deref(),
            daemon_api::WIRE_PAGE_MAX,
            |f| f.path.clone(),
        ))
    }

    async fn model_download(
        &self,
        provider: String,
        source: ModelSource,
    ) -> Result<DownloadId, ApiError> {
        let m = self.require_models()?;
        let engine = self.resolve_available_engine(&provider).await?;
        m.download(ModelRef::new(engine, source))
            .await
            .map_err(map_model_err)
    }

    async fn model_install_from_url(
        &self,
        provider: String,
        url: String,
    ) -> Result<daemon_api::InstallFromUrlOutcome, ApiError> {
        let m = self.require_models()?;
        let engine = self.resolve_available_engine(&provider).await?;
        let parsed = daemon_models::hf::url::parse_hf_url(&url).map_err(map_model_err)?;
        // An artifact-strategy catalog (llama.cpp) needs a single file: a bare repo/tree URL is a
        // "pick a file" answer, not a job. Repository-strategy (mistral.rs) installs the whole
        // repo, so the bare form starts immediately; a file-pinning URL always starts.
        if parsed.file.is_none() && matches!(engine, ModelEngine::Llama) {
            return Ok(daemon_api::InstallFromUrlOutcome::NeedsFileChoice {
                repo: parsed.repo,
                revision: parsed.revision,
            });
        }
        let source = ModelSource::Hf {
            repo: parsed.repo,
            file: parsed.file,
            revision: parsed.revision,
        };
        let id = m
            .download(ModelRef::new(engine, source))
            .await
            .map_err(map_model_err)?;
        Ok(daemon_api::InstallFromUrlOutcome::Started { id })
    }

    async fn model_downloads(&self) -> Vec<DownloadStatus> {
        match &self.models {
            Some(m) => m.downloads().await,
            None => Vec::new(),
        }
    }

    async fn model_cancel(&self, id: DownloadId) -> Result<(), ApiError> {
        let m = self.require_models()?;
        m.cancel(id).await.map_err(map_model_err)
    }

    async fn model_pause(&self, id: DownloadId) -> Result<(), ApiError> {
        let m = self.require_models()?;
        m.pause(id).await.map_err(map_model_err)
    }

    async fn model_resume(&self, id: DownloadId) -> Result<(), ApiError> {
        let m = self.require_models()?;
        m.resume(id).await.map_err(map_model_err)
    }

    async fn model_catalog(&self) -> Vec<InstalledModel> {
        match &self.models {
            Some(m) => m.catalog().await,
            None => Vec::new(),
        }
    }

    async fn model_delete(&self, id: ModelId) -> Result<(), ApiError> {
        let m = self.require_models()?;
        m.delete(&id).await.map_err(map_model_err)
    }

    async fn model_activate(&self, id: ModelId, profile: Option<String>) -> Result<(), ApiError> {
        let m = self.require_models()?;
        let profile = profile.unwrap_or_else(|| self.default_local_profile.clone());
        m.activate(&id, &profile)
            .await
            .map(|_| ())
            .map_err(map_model_err)
    }

    async fn model_recommend(
        &self,
        args: ModelRecommendArgs,
    ) -> Result<QuantRecommendation, ApiError> {
        let ModelRecommendArgs {
            repo,
            revision,
            provider,
            budget_bytes,
        } = args;
        let m = self.require_models()?;
        let engine = self.resolve_available_engine(&provider).await?;
        m.recommend(&repo, revision.as_deref(), engine, budget_bytes)
            .await
            .map_err(map_model_err)
    }

    async fn model_quantize(&self, args: ModelQuantizeArgs) -> Result<QuantizeId, ApiError> {
        let ModelQuantizeArgs {
            repo,
            revision,
            target_quant,
            source_file,
        } = args;
        let m = self.require_models()?;
        m.quantize(&repo, revision.as_deref(), &target_quant, source_file)
            .await
            .map_err(map_model_err)
    }

    async fn model_quantizes(&self) -> Vec<QuantizeStatus> {
        match &self.models {
            Some(m) => m.quantizes().await,
            None => Vec::new(),
        }
    }

    async fn model_inspect(&self, id: ModelId) -> Result<GgufInfo, ApiError> {
        let m = self.require_models()?;
        m.inspect(&id).await.map_err(map_model_err)
    }

    async fn models(&self, after: Option<String>) -> daemon_api::WirePage<ModelDescriptor> {
        // Cursor order: descriptor id ascending (the merged cloud+local catalog has no stable
        // order of its own). Internal full-catalog consumers use `models_all` instead.
        let mut catalog = self.models_all().await;
        catalog.sort_by(|a, b| a.id.cmp(&b.id));
        daemon_api::paginate(catalog, after.as_deref(), daemon_api::WIRE_PAGE_MAX, |m| {
            m.id.clone()
        })
    }

    async fn model_current(
        &self,
        profile: Option<String>,
    ) -> Result<Option<ModelDescriptor>, ApiError> {
        let spec = if self.profiles.is_some() {
            self.resolve_profile(profile)?
        } else {
            None
        };
        let Some(spec) = spec else { return Ok(None) };
        // Prefer a catalog entry (carries context/pricing); else synthesize from the profile spec.
        // The FULL catalog (not one wire page): the lookup is by id across everything discoverable.
        if let Some(found) = self
            .models_all()
            .await
            .into_iter()
            .find(|m| m.id == spec.model)
        {
            return Ok(Some(found));
        }
        Ok(Some(ModelDescriptor {
            id: spec.model.clone(),
            provider: spec.provider,
            display_name: None,
            context_length: ModelDescriptor::known_context_length(&spec.model),
            input_price_micros_per_mtok: None,
            output_price_micros_per_mtok: None,
            local: matches!(
                spec.provider,
                ProviderSelector::LlamaCpp | ProviderSelector::MistralRs
            ),
        }))
    }

    async fn provider_catalog(&self) -> Vec<ProviderDescriptor> {
        // The binary wires the genai-backed catalog (local engines + every genai vendor + Daemon
        // Cloud). Independent of the launch default, so an unconfigured node still lists providers.
        let base = match &self.cloud_catalog {
            Some(catalog) => catalog.providers().await,
            // Fallback for a catalog-less node (test stubs / remote-only): the local engines + Daemon
            // Cloud (genai vendors need the binary's genai hook). The base URL is the public gateway.
            None => Self::static_provider_catalog(),
        };
        // Overlay the durable user-defined custom providers (the single merged read model).
        self.merge_custom_providers(base).await
    }

    async fn provider_models(
        &self,
        provider: String,
        credential_ref: Option<String>,
        transient_key: Option<String>,
        after: Option<String>,
    ) -> daemon_api::ProviderModelsResult {
        // Local engines: the node is the single source of truth — return the installed models from
        // the ModelManager catalog. An empty list here genuinely means "nothing installed yet".
        let listed = if daemon_common::ModelEngine::from_provider_id(&provider).is_some() {
            Ok(self.installed_models_for(&provider).await)
        } else if let Some(custom) = self.custom_provider_by_id(&provider).await {
            // A user-defined custom provider: list its OpenAI-compatible endpoint, credential-aware.
            // A first-run transient key wins; else the stored credential the request (or the
            // provider's own default `credential_ref`) points at. A turn always uses the stored
            // profile credential regardless.
            let key = transient_key.or_else(|| {
                credential_ref
                    .as_deref()
                    .or(custom.credential_ref.as_deref())
                    .and_then(|r| self.credentials.as_ref().and_then(|c| c.get(r)))
            });
            match &self.cloud_catalog {
                Some(catalog) => catalog.openai_compat_models(&custom.base_url, key).await,
                None => Err(no_discovery_hook()),
            }
        } else {
            // Resolve the LIST credential: a first-run transient key wins, else the stored
            // credential the `credential_ref` points at. A turn always uses the stored profile
            // credential regardless.
            let key = transient_key.or_else(|| {
                credential_ref
                    .as_deref()
                    .and_then(|r| self.credentials.as_ref().and_then(|c| c.get(r)))
            });
            match &self.cloud_catalog {
                Some(catalog) => catalog.provider_models(&provider, key).await,
                None => Err(no_discovery_hook()),
            }
        };
        let mut models = match listed {
            Ok(models) => models,
            Err(error) => {
                return daemon_api::ProviderModelsResult {
                    models: Vec::new(),
                    next: None,
                    error: Some(error),
                }
            }
        };
        // Cursor order: descriptor id ascending (vendor listings arrive in vendor order).
        models.sort_by(|a, b| a.id.cmp(&b.id));
        let page = daemon_api::paginate(models, after.as_deref(), daemon_api::WIRE_PAGE_MAX, |m| {
            m.id.clone()
        });
        daemon_api::ProviderModelsResult {
            models: page.items,
            next: page.next,
            error: None,
        }
    }

    async fn custom_provider_list(&self) -> Vec<CustomProvider> {
        self.custom_providers_decoded().await
    }

    async fn custom_provider_set(&self, mut provider: CustomProvider) -> Result<(), ApiError> {
        // The node owns provenance + the wire binding: a wire-set is always a `User` entry bound to
        // the OpenAI-compatible `DaemonApi` selector, regardless of what the client supplied.
        provider.source = daemon_api::CustomProviderSource::User;
        provider.wire_selector = ProviderSelector::DaemonApi;
        if provider.id.trim().is_empty() {
            return Err(ApiError::Unsupported("custom provider id is empty".into()));
        }
        validate_base_url(&provider.base_url)?;
        self.store
            .custom_provider_set(daemon_store::CustomProviderRecord {
                id: provider.id.clone(),
                entry: to_cbor(&provider),
            })
            .await
            .map_err(|e| ApiError::Other(format!("custom provider set: {e}")))
    }

    async fn custom_provider_remove(&self, id: String) -> Result<(), ApiError> {
        // Config-seeded entries are owned by node config (re-seeded each boot), so they are not
        // user-removable over the wire; only `User` entries can be deleted.
        if let Some(existing) = self.custom_provider_by_id(&id).await {
            if matches!(existing.source, daemon_api::CustomProviderSource::Config) {
                return Err(ApiError::Unsupported(format!(
                    "custom provider {id:?} is config-seeded and cannot be removed over the wire"
                )));
            }
        }
        self.store
            .custom_provider_remove(&id)
            .await
            .map_err(|e| ApiError::Other(format!("custom provider remove: {e}")))
    }
}

/// Validate a custom-provider base URL is a well-formed http(s) URL. A server-side UX check; the
/// egress client re-validates and enforces SSRF policy on the actual listing/turn call.
fn validate_base_url(base: &str) -> Result<(), ApiError> {
    let trimmed = base.trim();
    if trimmed.is_empty() {
        return Err(ApiError::Unsupported(
            "custom provider base_url is empty".into(),
        ));
    }
    if !(trimmed.starts_with("http://") || trimmed.starts_with("https://")) {
        return Err(ApiError::Unsupported(
            "custom provider base_url must be an http(s) URL".into(),
        ));
    }
    Ok(())
}

impl NodeApiImpl {
    /// The persisted custom providers, decoded from the durable store (an undecodable row is
    /// skipped). Backs `custom_provider_list` and the `provider_catalog`/`provider_models` overlays.
    pub(crate) async fn custom_providers_decoded(&self) -> Vec<CustomProvider> {
        self.store
            .custom_provider_list()
            .await
            .into_iter()
            .filter_map(|rec| from_cbor::<CustomProvider>(&rec.entry).ok())
            .collect()
    }

    /// One persisted custom provider by id, if present.
    async fn custom_provider_by_id(&self, id: &str) -> Option<CustomProvider> {
        self.custom_providers_decoded()
            .await
            .into_iter()
            .find(|p| p.id == id)
    }

    /// Overlay the durable custom providers onto a builtin provider list, custom winning on an id
    /// collision (mirrors `agent_catalog`'s manual-over-builtin precedence).
    async fn merge_custom_providers(
        &self,
        mut base: Vec<ProviderDescriptor>,
    ) -> Vec<ProviderDescriptor> {
        for custom in self.custom_providers_decoded().await {
            let desc = custom.to_descriptor();
            if let Some(slot) = base.iter_mut().find(|d| d.id == desc.id) {
                *slot = desc;
            } else {
                base.push(desc);
            }
        }
        base
    }

    /// The model-management facade, or [`ApiError::Unsupported`] when this node has none.
    fn require_models(&self) -> Result<&Arc<ModelManager>, ApiError> {
        self.models
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("model management is not enabled".into()))
    }

    /// The FULL discoverable catalog (cloud + local), unpaged — the internal backing of the wire
    /// `models` page and of by-id lookups (`model_current`) that must search everything.
    /// Networked models: a live `genai` listing (per adapter with a resolvable key, namespaced,
    /// pricing/context overlaid) when the discovery hook is wired, else the static catalog
    /// (incl. claude-opus-4-8). Then merge any locally-installed (GGUF) models.
    pub(crate) async fn models_all(&self) -> Vec<ModelDescriptor> {
        let mut out = match &self.cloud_catalog {
            Some(catalog) => catalog.list().await,
            None => ModelDescriptor::builtin_cloud_catalog(),
        };
        if let Some(m) = &self.models {
            for im in m.catalog().await {
                // Vision-projector (mmproj) companions are inventory, never chat models.
                if daemon_models::mmproj::is_projector_record(&im) {
                    continue;
                }
                let provider = match im.model.engine {
                    ModelEngine::MistralRs => ProviderSelector::MistralRs,
                    ModelEngine::Llama => ProviderSelector::LlamaCpp,
                };
                out.push(ModelDescriptor {
                    id: im.id.as_str().to_string(),
                    provider,
                    // The record's human-friendly name (repo id / file stem) — without it a
                    // client renders the opaque catalog-id hash.
                    display_name: (!im.display_name.is_empty()).then(|| im.display_name.clone()),
                    context_length: im.context_length,
                    input_price_micros_per_mtok: None,
                    output_price_micros_per_mtok: None,
                    local: true,
                });
            }
        }
        out
    }

    /// The catalog-less fallback provider list: local engines + Daemon Cloud (the genai cloud vendors
    /// require the binary's genai hook). Used by test stubs / remote-only nodes.
    fn static_provider_catalog() -> Vec<ProviderDescriptor> {
        // Catalog order IS the wire order clients render: Daemon Cloud (the product default)
        // first, then the local engines. Local-ness is carried by `kind` (the client renders its
        // own indicator), never baked into the display name. The canonical rows live on
        // `ProviderDescriptor` so binary + host + test stubs share one definition.
        vec![
            ProviderDescriptor::daemon_cloud("https://api.daemon.ai/api/v1/"),
            ProviderDescriptor::llama_cpp(),
            ProviderDescriptor::mistral_rs(),
        ]
    }

    /// The installed local models for one engine id (`"llama_cpp"` / `"mistral_rs"`), read from the
    /// ModelManager catalog. Empty when model management is not enabled. Vision-projector (mmproj)
    /// records are excluded — offering one as a chat model is exactly the `arch == 'clip'` fatal.
    async fn installed_models_for(&self, engine_id: &str) -> Vec<ModelDescriptor> {
        let Some(m) = &self.models else {
            return Vec::new();
        };
        let want = match engine_id {
            "llama_cpp" => ProviderSelector::LlamaCpp,
            "mistral_rs" => ProviderSelector::MistralRs,
            _ => return Vec::new(),
        };
        m.catalog()
            .await
            .into_iter()
            .filter(|im| !daemon_models::mmproj::is_projector_record(im))
            .filter_map(|im| {
                let provider = match im.model.engine {
                    ModelEngine::MistralRs => ProviderSelector::MistralRs,
                    ModelEngine::Llama => ProviderSelector::LlamaCpp,
                };
                (provider == want).then(|| ModelDescriptor {
                    id: im.id.as_str().to_string(),
                    provider,
                    // Same rule as models_all: the record's display name rides along so the
                    // wizard's ProviderModels list never shows the catalog-id hash.
                    display_name: (!im.display_name.is_empty()).then(|| im.display_name.clone()),
                    context_length: im.context_length,
                    input_price_micros_per_mtok: None,
                    output_price_micros_per_mtok: None,
                    local: true,
                })
            })
            .collect()
    }
}

/// Resolve a managed provider id (`"llama_cpp"` / `"mistral_rs"`) to its engine, or refuse: the
/// acquisition surface (search/files/download/install/recommend) only exists for managed catalogs.
fn resolve_managed_provider(provider: &str) -> Result<ModelEngine, ApiError> {
    ModelEngine::from_provider_id(provider).ok_or_else(|| {
        ApiError::Unsupported(format!(
            "provider {provider:?} has no managed catalog (only llama_cpp / mistral_rs install models)"
        ))
    })
}

impl NodeApiImpl {
    /// [`resolve_managed_provider`] PLUS the availability gate: the provider-catalog row carries
    /// the boot-time worker engine probe's verdict, and acquisition (search/files/download/
    /// install/recommend) into an engine the deployed worker cannot run is refused with the
    /// row's reason — no more downloading models that can't run.
    async fn resolve_available_engine(&self, provider: &str) -> Result<ModelEngine, ApiError> {
        let engine = resolve_managed_provider(provider)?;
        let rows = match &self.cloud_catalog {
            Some(catalog) => catalog.providers().await,
            None => Self::static_provider_catalog(),
        };
        if let Some(row) = rows.iter().find(|p| p.id == provider) {
            if !row.available {
                return Err(ApiError::Unsupported(format!(
                    "provider {provider:?} is unavailable on this node: {}",
                    row.unavailable_reason
                        .as_deref()
                        .unwrap_or("its inference engine is not available")
                )));
            }
        }
        Ok(engine)
    }
}

/// The structured listing failure for a node with no discovery hook wired.
fn no_discovery_hook() -> daemon_api::ProviderListError {
    daemon_api::ProviderListError {
        kind: daemon_api::ProviderListErrorKind::Unsupported,
        message: "this node has no provider discovery hook wired".into(),
    }
}

/// Map a `daemon-models` error onto the transport-stable [`ApiError`].
fn map_model_err(e: ModelError) -> ApiError {
    match e {
        ModelError::NotFound(m) => ApiError::Other(format!("not found: {m}")),
        ModelError::AccessDenied(m) => ApiError::Other(format!("access denied: {m}")),
        ModelError::Invalid(m) => ApiError::Unsupported(m),
        ModelError::Unknown(m) => ApiError::Other(format!("unknown id: {m}")),
        other => ApiError::Other(other.to_string()),
    }
}
