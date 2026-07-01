// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;

#[async_trait]
impl ModelApi for NodeApiImpl {
    async fn model_search(&self, query: SearchQuery) -> Result<SearchPage, ApiError> {
        let m = self.require_models()?;
        m.search(query).await.map_err(map_model_err)
    }

    async fn model_files(
        &self,
        repo: String,
        revision: Option<String>,
        engine: ModelEngine,
    ) -> Result<Vec<ModelFile>, ApiError> {
        let m = self.require_models()?;
        m.model_files(&repo, revision.as_deref(), engine)
            .await
            .map_err(map_model_err)
    }

    async fn model_download(&self, model: ModelRef) -> Result<DownloadId, ApiError> {
        let m = self.require_models()?;
        m.download(model).await.map_err(map_model_err)
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
            engine,
            budget_bytes,
        } = args;
        let m = self.require_models()?;
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

    async fn models(&self) -> Vec<ModelDescriptor> {
        // Networked models: a live `genai` listing (per adapter with a resolvable key, namespaced,
        // pricing/context overlaid) when the discovery hook is wired, else the static catalog
        // (incl. claude-opus-4-8). Then merge any locally-installed (GGUF) models.
        let mut out = match &self.cloud_catalog {
            Some(catalog) => catalog.list().await,
            None => ModelDescriptor::builtin_cloud_catalog(),
        };
        if let Some(m) = &self.models {
            for im in m.catalog().await {
                let provider = match im.model.engine {
                    ModelEngine::MistralRs => ProviderSelector::MistralRs,
                    ModelEngine::Llama => ProviderSelector::LlamaCpp,
                };
                out.push(ModelDescriptor {
                    id: im.id.as_str().to_string(),
                    provider,
                    display_name: None,
                    context_length: im.context_length,
                    input_price_micros_per_mtok: None,
                    output_price_micros_per_mtok: None,
                    local: true,
                });
            }
        }
        out
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
        if let Some(found) = self.models().await.into_iter().find(|m| m.id == spec.model) {
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
        match &self.cloud_catalog {
            Some(catalog) => catalog.providers().await,
            // Fallback for a catalog-less node (test stubs / remote-only): the local engines + Daemon
            // Cloud (genai vendors need the binary's genai hook). The base URL is the public gateway.
            None => Self::static_provider_catalog(),
        }
    }

    async fn provider_models(
        &self,
        provider: String,
        credential_ref: Option<String>,
        transient_key: Option<String>,
    ) -> Vec<ModelDescriptor> {
        // Local engines: the node is the single source of truth — return the installed models from
        // the ModelManager catalog (the client appends its own "Discover More" affordance).
        if provider == "llama_cpp" || provider == "mistral_rs" {
            return self.installed_models_for(&provider).await;
        }
        // Resolve the LIST credential: a first-run transient key wins, else the stored credential the
        // `credential_ref` points at. A turn always uses the stored profile credential regardless.
        let key = transient_key.or_else(|| {
            credential_ref
                .as_deref()
                .and_then(|r| self.credentials.as_ref().and_then(|c| c.get(r)))
        });
        match &self.cloud_catalog {
            Some(catalog) => catalog.provider_models(&provider, key).await,
            None => Vec::new(),
        }
    }
}

impl NodeApiImpl {
    /// The model-management facade, or [`ApiError::Unsupported`] when this node has none.
    fn require_models(&self) -> Result<&Arc<ModelManager>, ApiError> {
        self.models
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("model management is not enabled".into()))
    }

    /// The catalog-less fallback provider list: local engines + Daemon Cloud (the genai cloud vendors
    /// require the binary's genai hook). Used by test stubs / remote-only nodes.
    fn static_provider_catalog() -> Vec<ProviderDescriptor> {
        vec![
            ProviderDescriptor {
                id: "llama_cpp".into(),
                display_name: "llama.cpp (local)".into(),
                kind: ProviderKindWire::Local,
                wire_selector: ProviderSelector::LlamaCpp,
                requires_key: false,
                supports_model_discovery: true,
                default_base_url: None,
            },
            ProviderDescriptor {
                id: "mistral_rs".into(),
                display_name: "mistral.rs (local)".into(),
                kind: ProviderKindWire::Local,
                wire_selector: ProviderSelector::MistralRs,
                requires_key: false,
                supports_model_discovery: true,
                default_base_url: None,
            },
            ProviderDescriptor {
                id: "daemon_cloud".into(),
                display_name: "Daemon Cloud".into(),
                kind: ProviderKindWire::DaemonCloud,
                wire_selector: ProviderSelector::DaemonApi,
                // Needs a key to RUN TURNS (bearer-authed inference); model LISTING stays keyless
                // (the public gateway `/models` is unauth; `provider_models` never gates on this).
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: Some("https://api.daemon.ai/api/v1/".into()),
            },
        ]
    }

    /// The installed local models for one engine id (`"llama_cpp"` / `"mistral_rs"`), read from the
    /// ModelManager catalog. Empty when model management is not enabled.
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
            .filter_map(|im| {
                let provider = match im.model.engine {
                    ModelEngine::MistralRs => ProviderSelector::MistralRs,
                    ModelEngine::Llama => ProviderSelector::LlamaCpp,
                };
                (provider == want).then(|| ModelDescriptor {
                    id: im.id.as_str().to_string(),
                    provider,
                    display_name: None,
                    context_length: im.context_length,
                    input_price_micros_per_mtok: None,
                    output_price_micros_per_mtok: None,
                    local: true,
                })
            })
            .collect()
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
