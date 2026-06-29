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
            context_length: ModelDescriptor::known_context_length(&spec.model),
            input_price_micros_per_mtok: None,
            output_price_micros_per_mtok: None,
            local: matches!(
                spec.provider,
                ProviderSelector::LlamaCpp | ProviderSelector::MistralRs
            ),
        }))
    }
}
