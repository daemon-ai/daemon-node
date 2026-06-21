//! [`ModelManager`] — the one facade the node surface (`ModelApi`) and the local provider wiring
//! call. It owns the HF client, the acquisition engine, the installed-model catalog, the shared
//! cache config, and the per-profile *active model* selection used for runtime model switching.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use daemon_common::{
    DownloadId, DownloadState, DownloadStatus, InstalledModel, ModelEngine, ModelFile, ModelId,
    ModelRef, ModelSource, QuantRecommendation, SearchPage, SearchQuery,
};
use tokio::sync::RwLock;

use crate::acquire::{Downloader, ResolvedArtifact};
use crate::cache::CacheConfig;
use crate::error::{ModelError, Result};
use crate::hardware::HardwareProbe;
use crate::hf::{files, search, HfClient};
use crate::quantize::Quantizer;
use crate::registry::{model_id, Registry};
use crate::{gguf, recommend, resolve};

/// Construction inputs for a [`ModelManager`].
#[derive(Clone, Debug, Default)]
pub struct ManagerConfig {
    /// The hub cache directory; `None` follows the `HF_*` / XDG precedence.
    pub cache_dir: Option<PathBuf>,
    /// The catalog manifest path; `None` places it next to the cache (`<hub>/daemon-catalog.json`).
    pub registry_path: Option<PathBuf>,
    /// The Hub endpoint; `None` uses the default (`https://huggingface.co`).
    pub endpoint: Option<String>,
    /// The `daemon-infer` worker binary (built with `--features llama`) used for offline
    /// quantization. `None` disables the quantize surface with a clear error.
    pub quantize_worker_bin: Option<PathBuf>,
}

/// The per-profile active-model selection, shared with the local provider wiring so a runtime
/// `model_activate` swaps which model new worker spawns load.
#[derive(Clone, Default)]
pub struct ActiveModels {
    inner: Arc<RwLock<HashMap<String, ModelRef>>>,
}

impl ActiveModels {
    /// The active model for `profile`, if one was activated.
    pub async fn get(&self, profile: &str) -> Option<ModelRef> {
        self.inner.read().await.get(profile).cloned()
    }

    /// Set the active model for `profile`.
    pub async fn set(&self, profile: impl Into<String>, model: ModelRef) {
        self.inner.write().await.insert(profile.into(), model);
    }
}

/// The model-management facade.
#[derive(Clone)]
pub struct ModelManager {
    client: HfClient,
    downloader: Downloader,
    registry: Registry,
    cache: CacheConfig,
    active: ActiveModels,
    quantizer: Quantizer,
}

impl ModelManager {
    /// Build a manager over the shared cache + catalog.
    pub async fn new(config: ManagerConfig) -> Result<Self> {
        let cache = CacheConfig::resolve(config.cache_dir);
        let client = match config.endpoint {
            Some(ep) => HfClient::with_endpoint(ep, cache.token.clone()),
            None => HfClient::new(cache.token.clone()),
        };
        let downloader = Downloader::new(&cache)?;
        let registry_path = config
            .registry_path
            .unwrap_or_else(|| cache.hub_dir.join("daemon-catalog.json"));
        let registry = Registry::open(registry_path).await?;
        let quantize_output_dir = cache.hub_dir.join("daemon-quantized");
        let quantizer = Quantizer::new(
            config.quantize_worker_bin,
            quantize_output_dir,
            registry.clone(),
        );
        Ok(Self {
            client,
            downloader,
            registry,
            cache,
            active: ActiveModels::default(),
            quantizer,
        })
    }

    /// The shared cache config (the local provider reads `sidecar_env()` from it).
    pub fn cache(&self) -> &CacheConfig {
        &self.cache
    }

    /// The shared active-model handle (the switchable local provider reads it).
    pub fn active_handle(&self) -> ActiveModels {
        self.active.clone()
    }

    // --- Discovery (Hugging Face) -------------------------------------------------------------

    /// Search repos (step 1).
    pub async fn search(&self, query: SearchQuery) -> Result<SearchPage> {
        search::search(&self.client, &query).await
    }

    /// List a repo's loadable files for `engine` (step 2).
    pub async fn model_files(
        &self,
        repo: &str,
        revision: Option<&str>,
        engine: ModelEngine,
    ) -> Result<Vec<ModelFile>> {
        let revision = revision.unwrap_or("main");
        files::list_files(&self.client, repo, revision, engine).await
    }

    /// Recommend a quantization for `repo` on the detected hardware (the "tune"-like pick). For
    /// llama it names the GGUF file to download; for mistral.rs it names an in-engine ISQ level.
    /// `budget_override` (bytes) replaces the auto-detected VRAM/RAM budget when set.
    pub async fn recommend(
        &self,
        repo: &str,
        revision: Option<&str>,
        engine: ModelEngine,
        budget_override: Option<u64>,
    ) -> Result<QuantRecommendation> {
        let revision = revision.unwrap_or("main");
        let budget = budget_override.unwrap_or_else(|| HardwareProbe::detect().budget_bytes());
        match engine {
            ModelEngine::Llama => {
                let files = files::list_files(&self.client, repo, revision, engine).await?;
                Ok(recommend::recommend_llama(repo, &files, budget))
            }
            ModelEngine::MistralRs => {
                let params = files::repo_param_count(&self.client, repo, revision).await;
                Ok(recommend::recommend_mistralrs(repo, params, budget))
            }
        }
    }

    /// Read GGUF metadata for a cataloged model (architecture, context length, file-type, …).
    pub async fn inspect(&self, id: &ModelId) -> Result<daemon_common::GgufInfo> {
        let record = self
            .registry
            .get(id)
            .await
            .ok_or_else(|| ModelError::Unknown(id.to_string()))?;
        let path = record.local_path.clone();
        tokio::task::spawn_blocking(move || crate::inspect::inspect(&path))
            .await
            .map_err(|e| ModelError::Other(e.to_string()))?
    }

    /// Quantize a repo's GGUF down to `target_quant` offline (via the llama-enabled worker). When
    /// `source_file` is `None` the highest-precision GGUF in the repo is used as the source; the
    /// source is downloaded if not already cached. Returns a job handle; poll [`quantizes`].
    ///
    /// Errors if the repo has no GGUF source — safetensors→GGUF conversion is out of scope.
    pub async fn quantize(
        &self,
        repo: &str,
        revision: Option<&str>,
        target_quant: &str,
        source_file: Option<String>,
    ) -> Result<daemon_common::QuantizeId> {
        let revision = revision.unwrap_or("main");
        let source_file = match source_file {
            Some(f) => f,
            None => {
                let files =
                    files::list_files(&self.client, repo, revision, ModelEngine::Llama).await?;
                recommend::highest_precision_gguf(&files)
                    .map(|f| f.path.clone())
                    .ok_or_else(|| {
                        ModelError::Invalid(format!(
                            "repo {repo} has no GGUF to quantize from; \
                             safetensors→GGUF conversion is out of scope"
                        ))
                    })?
            }
        };
        let source_ref = ModelRef::new(
            ModelEngine::Llama,
            ModelSource::Hf {
                repo: repo.to_string(),
                file: Some(source_file.clone()),
                revision: revision.to_string(),
            },
        );
        let artifact = self.resolve(&source_ref).await?;
        self.quantizer
            .start(crate::quantize::QuantizeRequest {
                repo: repo.to_string(),
                source_file,
                source_path: artifact.local_path,
                target_quant: target_quant.to_string(),
                nthread: 0,
            })
            .await
    }

    /// A snapshot of every quantization job.
    pub async fn quantizes(&self) -> Vec<daemon_common::QuantizeStatus> {
        self.quantizer.statuses().await
    }

    // --- Acquisition --------------------------------------------------------------------------

    /// Start a background download for `model`, returning its job handle. On completion the model is
    /// cataloged automatically (a watcher task upserts the registry).
    pub async fn download(&self, model: ModelRef) -> Result<DownloadId> {
        let plan = resolve::plan(&self.client, &model).await?;
        let id = self.downloader.start(model.clone(), plan).await?;
        self.spawn_catalog_watcher(id, model);
        Ok(id)
    }

    /// All download job statuses.
    pub async fn downloads(&self) -> Vec<DownloadStatus> {
        self.downloader.statuses().await
    }

    /// Cancel a download (abandon partial bytes).
    pub async fn cancel(&self, id: DownloadId) -> Result<()> {
        self.downloader.cancel(id).await
    }

    /// Pause a download (keep partial bytes for resume).
    pub async fn pause(&self, id: DownloadId) -> Result<()> {
        self.downloader.pause(id).await
    }

    /// Resume a paused/failed download.
    pub async fn resume(&self, id: DownloadId) -> Result<()> {
        self.downloader.resume(id).await
    }

    // --- Catalog + lifecycle ------------------------------------------------------------------

    /// Every installed model.
    pub async fn catalog(&self) -> Vec<InstalledModel> {
        self.registry.list().await
    }

    /// Remove a model from the catalog and best-effort delete its cached artifact.
    pub async fn delete(&self, id: &ModelId) -> Result<()> {
        let removed = self.registry.remove(id).await?;
        if let Some(record) = removed {
            best_effort_delete(&record.local_path);
        }
        Ok(())
    }

    /// Activate a cataloged model for `profile`: mark it the active selection (new worker spawns
    /// load it) and ensure it is resolvable on disk.
    pub async fn activate(&self, id: &ModelId, profile: &str) -> Result<InstalledModel> {
        let record = self
            .registry
            .get(id)
            .await
            .ok_or_else(|| ModelError::Unknown(id.to_string()))?;
        self.active.set(profile, record.model.clone()).await;
        Ok(record)
    }

    /// Resolve a model to a ready on-disk artifact, downloading + cataloging it if necessary. This
    /// is the "resolve-before-load" entry the local provider calls before spawning a worker.
    pub async fn resolve(&self, model: &ModelRef) -> Result<ResolvedArtifact> {
        // Already-present local models resolve directly.
        if let ModelSource::Local { path } = &model.source {
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            return Ok(ResolvedArtifact {
                local_path: path.clone(),
                size_bytes: size,
            });
        }
        // A cataloged model whose artifact still exists resolves without any network.
        if let Some(record) = self.registry.find(model).await {
            if record.local_path.exists() {
                return Ok(ResolvedArtifact {
                    local_path: record.local_path,
                    size_bytes: record.size_bytes,
                });
            }
        }
        // Otherwise download to completion, catalog, and return.
        let plan = resolve::plan(&self.client, model).await?;
        let id = self.downloader.start(model.clone(), plan).await?;
        self.await_completion(id).await?;
        let record = self.catalog_completed(id, model.clone()).await?;
        Ok(ResolvedArtifact {
            local_path: record.local_path,
            size_bytes: record.size_bytes,
        })
    }

    // --- internals ----------------------------------------------------------------------------

    /// Block until job `id` reaches a terminal state, mapping a non-success terminal to an error.
    async fn await_completion(&self, id: DownloadId) -> Result<()> {
        loop {
            let status = self
                .downloader
                .status(id)
                .await
                .ok_or_else(|| ModelError::Unknown(id.to_string()))?;
            match status.state {
                DownloadState::Completed => return Ok(()),
                DownloadState::Failed => {
                    return Err(ModelError::Download(
                        status.error.unwrap_or_else(|| "download failed".into()),
                    ))
                }
                DownloadState::Cancelled => {
                    return Err(ModelError::Download("download cancelled".into()))
                }
                DownloadState::Paused => {
                    return Err(ModelError::Download("download paused".into()))
                }
                DownloadState::Queued | DownloadState::Downloading => {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    }

    /// Build + persist the catalog record for a completed job.
    async fn catalog_completed(&self, id: DownloadId, model: ModelRef) -> Result<InstalledModel> {
        let artifact = self
            .downloader
            .artifact(id)
            .await
            .ok_or_else(|| ModelError::Download("completed job has no artifact".into()))?;
        let record = build_record(model, artifact.local_path, artifact.size_bytes);
        self.registry.upsert(record.clone()).await?;
        Ok(record)
    }

    /// Spawn a background watcher that catalogs job `id` once it completes.
    fn spawn_catalog_watcher(&self, id: DownloadId, model: ModelRef) {
        let downloader = self.downloader.clone();
        let registry = self.registry.clone();
        tokio::spawn(async move {
            loop {
                let Some(status) = downloader.status(id).await else {
                    return;
                };
                match status.state {
                    DownloadState::Completed => {
                        if let Some(artifact) = downloader.artifact(id).await {
                            let record = build_record(
                                model.clone(),
                                artifact.local_path,
                                artifact.size_bytes,
                            );
                            if let Err(e) = registry.upsert(record).await {
                                tracing::warn!(error = %e, "failed to catalog completed download");
                            }
                        }
                        return;
                    }
                    DownloadState::Failed | DownloadState::Cancelled => return,
                    _ => tokio::time::sleep(Duration::from_millis(300)).await,
                }
            }
        });
    }
}

/// Build a catalog record for a freshly acquired model, enriching it with GGUF metadata (arch,
/// context length, authoritative file-type) when the local artifact is a single GGUF file.
fn build_record(model: ModelRef, local_path: PathBuf, size_bytes: u64) -> InstalledModel {
    let mut record = InstalledModel {
        id: model_id(&model),
        display_name: display_name(&model),
        quant: model_quant(&model),
        local_path,
        size_bytes,
        installed_at_ms: now_ms(),
        arch: None,
        context_length: None,
        file_type: None,
        model,
    };
    crate::inspect::enrich_installed(&mut record);
    record
}

/// A human-friendly display name for a model reference.
fn display_name(model: &ModelRef) -> String {
    match &model.source {
        ModelSource::Hf { repo, file, .. } => match file {
            Some(f) => format!("{repo}/{f}"),
            None => repo.clone(),
        },
        ModelSource::Local { path } => path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned()),
    }
}

/// The quantization label for a model reference (from a named GGUF file).
fn model_quant(model: &ModelRef) -> Option<String> {
    match &model.source {
        ModelSource::Hf {
            file: Some(f), ..
        } => gguf::quant_label(f),
        ModelSource::Local { path } => path
            .file_name()
            .and_then(|n| gguf::quant_label(&n.to_string_lossy())),
        _ => None,
    }
}

/// Best-effort delete of a cached artifact (a file or a directory). Errors are ignored — a stale
/// cache entry is harmless and the user can clear the cache manually.
fn best_effort_delete(path: &std::path::Path) {
    if path.is_dir() {
        let _ = std::fs::remove_dir_all(path);
    } else {
        let _ = std::fs::remove_file(path);
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
