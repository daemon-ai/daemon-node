// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: fs here touches the model cache/registry under the node data root (paths derived from
// sanitized model refs, not attacker-influenced); raw fs allowed file-wide. No process spawns here.
#![allow(clippy::disallowed_methods)]

//! [`ModelManager`] — the one facade the node surface (`ModelApi`) and the local provider wiring
//! call. It owns the HF client, the acquisition engine, the installed-model catalog, the shared
//! cache config, and the per-profile *active model* selection used for runtime model switching.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;
use daemon_common::{
    DownloadId, DownloadState, DownloadStatus, InstalledModel, ModelEngine, ModelFile, ModelId,
    ModelRef, ModelSource, QuantRecommendation, SearchPage, SearchQuery,
};

use crate::acquire::{Downloader, ResolvedArtifact};
use crate::cache::CacheConfig;
use crate::error::{ModelError, Result};
use crate::hardware::HardwareProbe;
use crate::hf::{files, search, HfClient};
use crate::quantize::Quantizer;
use crate::registry::{model_id, Registry};
use crate::{gguf, mmproj, recommend, resolve};

/// Construction inputs for a [`ModelManager`].
#[derive(Clone, Debug, Default)]
pub struct ManagerConfig {
    /// The hub cache directory; `None` follows the `HF_*` / XDG precedence.
    pub cache_dir: Option<PathBuf>,
    /// The last-resort hub cache directory when `cache_dir` is unset AND the `HF_*`/XDG/`HOME`
    /// precedence resolves nothing (HOME-less containers/microvms). The daemon passes a directory
    /// under its own data dir so boot never depends on a home directory existing.
    pub fallback_cache_dir: Option<PathBuf>,
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
#[derive(Clone)]
pub struct ActiveModels {
    inner: Arc<ArcSwap<HashMap<String, ModelRef>>>,
}

impl Default for ActiveModels {
    fn default() -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(HashMap::new())),
        }
    }
}

impl ActiveModels {
    /// The active model for `profile`, if one was activated.
    pub fn get(&self, profile: &str) -> Option<ModelRef> {
        self.inner.load().get(profile).cloned()
    }

    /// Set the active model for `profile`.
    pub fn set(&self, profile: impl Into<String>, model: ModelRef) {
        let profile = profile.into();
        self.inner.rcu(|current| {
            let mut next = (**current).clone();
            next.insert(profile.clone(), model.clone());
            Arc::new(next)
        });
    }
}

/// A callback the host wires to announce installed-model registry changes (the L3
/// `CatalogChanged` node event): invoked after a completed download is cataloged and after a
/// model is deleted, so clients refetch `ModelCatalog` instead of polling.
pub type CatalogChangedCb = Arc<dyn Fn() + Send + Sync>;

/// The shared, late-wireable slot a [`CatalogChangedCb`] lives in (set after assembly, read by
/// the background catalog watchers spawned per download).
type CatalogChangedSlot = Arc<std::sync::Mutex<Option<CatalogChangedCb>>>;

/// The model-management facade.
#[derive(Clone)]
pub struct ModelManager {
    client: HfClient,
    downloader: Downloader,
    registry: Registry,
    cache: CacheConfig,
    active: ActiveModels,
    quantizer: Quantizer,
    catalog_changed: CatalogChangedSlot,
}

impl ModelManager {
    /// Wire the node-wide download-progress callback (L3 `DownloadProgress`): the host sets this
    /// after assembly so a job's progress fans onto the event feed instead of the client polling.
    pub fn set_download_progress(&self, cb: crate::acquire::DownloadProgressCb) {
        self.downloader.set_progress(cb);
    }

    /// Wire the node-wide catalog-changed callback (L3 `CatalogChanged`): the host sets this after
    /// assembly so registry changes (a cataloged download / a delete) fan onto the event feed.
    pub fn set_catalog_changed(&self, cb: CatalogChangedCb) {
        *self.catalog_changed.lock().unwrap() = Some(cb);
    }

    /// Build a manager over the shared cache + catalog.
    pub async fn new(config: ManagerConfig) -> Result<Self> {
        let cache = CacheConfig::resolve_with_fallback(config.cache_dir, config.fallback_cache_dir);
        let client = match &config.endpoint {
            Some(ep) => HfClient::with_endpoint(ep.clone(), cache.token.clone()),
            None => HfClient::new(cache.token.clone()),
        };
        let downloader = Downloader::new(&cache, config.endpoint.as_deref())?;
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
            catalog_changed: Arc::new(std::sync::Mutex::new(None)),
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
            notify_catalog_changed(&self.catalog_changed);
        }
        Ok(())
    }

    /// Activate a cataloged model for `profile`: mark it the active selection (new worker spawns
    /// load it) and ensure it is resolvable on disk. Vision-projector (mmproj) artifacts are
    /// rejected with an actionable error — activating one would hand the llama worker a CLIP
    /// projector as the chat model (`unsupported model architecture: 'clip'`).
    pub async fn activate(&self, id: &ModelId, profile: &str) -> Result<InstalledModel> {
        let record = self
            .registry
            .get(id)
            .await
            .ok_or_else(|| ModelError::Unknown(id.to_string()))?;
        if mmproj::is_projector_record(&record) {
            return Err(projector_rejection(&record.display_name));
        }
        self.active.set(profile, record.model.clone());
        Ok(record)
    }

    /// Resolve a model to a ready on-disk artifact, downloading + cataloging it if necessary. This
    /// is the "resolve-before-load" entry the local provider calls before spawning a worker.
    ///
    /// Guards: a reference naming a vision-projector (mmproj) artifact — or resolving to a record
    /// classified as one (`arch == "clip"`) — is rejected with an actionable error instead of
    /// letting the worker die on `unsupported model architecture: 'clip'`. A cataloged artifact is
    /// re-verified (exists + size matches the record + GGUF magic) before it is trusted; a failed
    /// check falls through to a fresh acquisition instead of loading a corrupt file.
    pub async fn resolve(&self, model: &ModelRef) -> Result<ResolvedArtifact> {
        // Already-present local models resolve directly.
        if let ModelSource::Local { path } = &model.source {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            if name.as_deref().is_some_and(mmproj::is_mmproj_path) {
                return Err(projector_rejection(&path.to_string_lossy()));
            }
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            return Ok(ResolvedArtifact {
                local_path: path.clone(),
                size_bytes: size,
                mmproj_path: None,
                sha256: None,
            });
        }
        if let ModelSource::Hf { file: Some(f), .. } = &model.source {
            if mmproj::is_mmproj_path(f) {
                return Err(projector_rejection(f));
            }
        }
        // A cataloged model whose artifact is still intact resolves without any network.
        if let Some(record) = self.registry.find(model).await {
            if mmproj::is_projector_record(&record) {
                return Err(projector_rejection(&record.display_name));
            }
            // Provenance: verify the on-disk artifact against its node-local pin BEFORE load, and
            // refuse (never load) on mismatch — a content swap that preserves size + GGUF magic
            // slips past `artifact_intact` but not this hash check (Phase 3 / Cluster E, L2).
            verify_pin_before_load(&record.local_path).await?;
            if artifact_intact(&record) {
                let mmproj_path = record
                    .mmproj_path
                    .as_deref()
                    .map(PathBuf::from)
                    .filter(|p| p.exists());
                return Ok(ResolvedArtifact {
                    local_path: record.local_path,
                    size_bytes: record.size_bytes,
                    mmproj_path,
                    sha256: None,
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
            mmproj_path: record.mmproj_path.as_deref().map(PathBuf::from),
            sha256: None,
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
        catalog_artifact(&self.registry, &self.catalog_changed, model, artifact).await
    }

    /// Spawn a background watcher that catalogs job `id` once it completes.
    fn spawn_catalog_watcher(&self, id: DownloadId, model: ModelRef) {
        let downloader = self.downloader.clone();
        let registry = self.registry.clone();
        let catalog_changed = self.catalog_changed.clone();
        tokio::spawn(async move {
            loop {
                let Some(status) = downloader.status(id).await else {
                    return;
                };
                match status.state {
                    DownloadState::Completed => {
                        if let Some(artifact) = downloader.artifact(id).await {
                            if let Err(e) =
                                catalog_artifact(&registry, &catalog_changed, model, artifact).await
                            {
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

/// Catalog a completed download's artifact (registry upsert) and announce the catalog change.
/// Shared by the background watcher and the blocking resolve path so every install emits exactly
/// one `CatalogChanged`. After the upsert, a best-effort pairing scan links vision-projector
/// companions across the whole catalog (a new text model picks up a pre-existing projector; a
/// new projector attaches to existing text records), so the returned record carries any
/// `mmproj_path` the scan assigned.
async fn catalog_artifact(
    registry: &Registry,
    catalog_changed: &CatalogChangedSlot,
    model: ModelRef,
    artifact: ResolvedArtifact,
) -> Result<InstalledModel> {
    let ResolvedArtifact {
        local_path,
        size_bytes,
        mmproj_path,
        sha256,
    } = artifact;
    // Record the node-local provenance pin beside the primary single-file artifact (Phase 3 /
    // Cluster E, L2). Verified before every subsequent load; a failed write is non-fatal.
    if let Some(hash) = &sha256 {
        write_pin(&local_path, hash);
    }
    let mut record = build_record(model, local_path, size_bytes);
    record.mmproj_path = mmproj_path.map(|p| p.to_string_lossy().into_owned());
    registry.upsert(record.clone()).await?;
    pair_projector_companions(registry).await;
    notify_catalog_changed(catalog_changed);
    Ok(registry.get(&record.id).await.unwrap_or(record))
}

/// Best-effort catalog-wide projector pairing (the local-scan rule: hard quant-compatibility
/// gate + recall ≥ 0.8): every text record whose `mmproj_path` is unset — or points at a file
/// that no longer exists — gets the best-matching cataloged projector. Failures are logged, never
/// fatal (pairing is an enrichment, not a correctness gate).
async fn pair_projector_companions(registry: &Registry) {
    let all = registry.list().await;
    let candidates: Vec<mmproj::LocalCandidate> = all
        .iter()
        .filter(|r| mmproj::is_projector_record(r))
        .map(|r| mmproj::LocalCandidate {
            path: r.local_path.to_string_lossy().into_owned(),
            file_type: r.file_type.clone(),
            quant: r.quant.clone(),
        })
        .collect();
    if candidates.is_empty() {
        return;
    }
    for record in all {
        if mmproj::is_projector_record(&record) {
            continue;
        }
        let stale = record
            .mmproj_path
            .as_deref()
            .is_some_and(|p| !std::path::Path::new(p).exists());
        if record.mmproj_path.is_some() && !stale {
            continue; // an intact pairing is never overridden
        }
        let stem = record
            .local_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Some(best) =
            mmproj::best_registry_companion(&stem, record.file_type.as_deref(), &candidates)
        else {
            continue;
        };
        let mut updated = record;
        updated.mmproj_path = Some(best);
        if let Err(e) = registry.upsert(updated).await {
            tracing::warn!(error = %e, "projector pairing upsert failed");
        }
    }
}

/// The actionable rejection for a vision-projector artifact selected as a chat model.
fn projector_rejection(name: &str) -> ModelError {
    ModelError::Invalid(format!(
        "{name} is a vision projector (mmproj), not a chat model — select the model's text \
         weights instead"
    ))
}

/// Whether a cataloged artifact is still trustworthy on disk: it exists, and — for single-file
/// (GGUF) artifacts — the size matches the record and the GGUF magic is intact. Directory
/// artifacts (mistral.rs snapshots) are checked for existence only.
fn artifact_intact(record: &InstalledModel) -> bool {
    let path = &record.local_path;
    if !path.exists() {
        return false;
    }
    if path.is_dir() {
        return true;
    }
    let size_ok = std::fs::metadata(path).is_ok_and(|m| m.len() == record.size_bytes);
    if !size_ok {
        return false;
    }
    if gguf::is_gguf(&path.to_string_lossy()) {
        return gguf::verify_gguf_magic(path).unwrap_or(false);
    }
    true
}

/// Invoke the wired catalog-changed callback, if any (the lock is held only to clone the `Arc`).
fn notify_catalog_changed(slot: &CatalogChangedSlot) {
    let cb = slot.lock().unwrap().clone();
    if let Some(cb) = cb {
        cb();
    }
}

/// Build a catalog record for a freshly acquired model, enriching it with GGUF metadata (arch,
/// context length, authoritative file-type) when the local artifact is a single GGUF file.
fn build_record(model: ModelRef, local_path: PathBuf, size_bytes: u64) -> InstalledModel {
    // wire v28: surface the node-local provenance pin (`<local_path>.sha256`) on the wire record for
    // display. Read before `local_path` is moved into the struct; `None` for directory models / no
    // sidecar. Node-side verify-before-load stays authoritative.
    let sha256 = read_pin(&local_path);
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
        mmproj_path: None,
        sha256,
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
        ModelSource::Hf { file: Some(f), .. } => gguf::quant_label(f),
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
        // Drop the provenance pin sidecar alongside the artifact so it can't linger and refuse a
        // legitimately re-downloaded replacement.
        let _ = std::fs::remove_file(pin_sidecar(path));
    }
}

/// The node-local provenance-pin sidecar path for an artifact (`<artifact>.sha256`).
fn pin_sidecar(artifact: &std::path::Path) -> PathBuf {
    let mut p = artifact.as_os_str().to_os_string();
    p.push(".sha256");
    PathBuf::from(p)
}

/// Read the recorded pin (lowercase-hex sha256) for `artifact`, if a non-empty sidecar exists.
fn read_pin(artifact: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(pin_sidecar(artifact))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Best-effort write of the provenance pin beside `artifact`. A failure logs a warning and leaves
/// the model loadable-but-unpinned (matching a legacy install) rather than blocking the install.
fn write_pin(artifact: &std::path::Path, sha256: &str) {
    let sidecar = pin_sidecar(artifact);
    if let Err(e) = std::fs::write(&sidecar, sha256) {
        tracing::warn!(path = %sidecar.display(), error = %e, "failed to write artifact pin sidecar");
    }
}

/// Verify a cataloged artifact against its node-local provenance pin **before load**. When a
/// `<artifact>.sha256` sidecar exists and the artifact is a present file, recompute its sha256 and
/// refuse (never load) on mismatch. No sidecar (a legacy install) or a missing file → `Ok` (the
/// existing existence/size check then decides fast-path vs benign re-acquire).
async fn verify_pin_before_load(local_path: &std::path::Path) -> Result<()> {
    let Some(expected) = read_pin(local_path) else {
        return Ok(());
    };
    if !local_path.is_file() {
        return Ok(());
    }
    let path = local_path.to_path_buf();
    let actual = tokio::task::spawn_blocking(move || crate::hash::sha256_file(&path))
        .await
        .map_err(|e| ModelError::Other(format!("hashing task failed: {e}")))?
        .map_err(|e| ModelError::io(local_path, e))?;
    if !actual.eq_ignore_ascii_case(&expected) {
        return Err(ModelError::Integrity(format!(
            "{}: on-disk artifact sha256 {actual} does not match the pinned {expected} — refusing \
             to load a tampered or corrupted model",
            local_path.display(),
        )));
    }
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::ModelEngine;

    fn model(repo: &str) -> ModelRef {
        ModelRef::new(ModelEngine::Llama, ModelSource::hf_file(repo, "m.gguf"))
    }

    #[tokio::test]
    async fn active_models_copy_on_write_reads_complete_snapshots() {
        let active = ActiveModels::default();
        active.set("default", model("org/old"));
        let readers: Vec<_> = (0..8)
            .map(|_| {
                let active = active.clone();
                tokio::spawn(async move {
                    for _ in 0..100 {
                        let seen = active.get("default");
                        assert!(seen.is_some());
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();

        active.set("default", model("org/new"));
        for reader in readers {
            reader.await.unwrap();
        }
        assert_eq!(active.get("default"), Some(model("org/new")));
    }

    /// Cataloging a completed artifact upserts the registry AND fires the wired catalog-changed
    /// callback (the L3 `CatalogChanged` hook) — exactly once per install.
    #[tokio::test]
    async fn catalog_artifact_upserts_and_notifies() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let path = std::env::temp_dir().join(format!(
            "daemon-models-catalog-notify-{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let registry = Registry::open(&path).await.unwrap();

        let fired = Arc::new(AtomicUsize::new(0));
        let slot: CatalogChangedSlot = Arc::new(std::sync::Mutex::new(None));
        {
            let fired = fired.clone();
            *slot.lock().unwrap() = Some(Arc::new(move || {
                fired.fetch_add(1, Ordering::SeqCst);
            }) as CatalogChangedCb);
        }

        let m = model("org/notify");
        let artifact = ResolvedArtifact {
            local_path: PathBuf::from("/tmp/notify.gguf"),
            size_bytes: 42,
            mmproj_path: None,
            sha256: None,
        };
        let record = catalog_artifact(&registry, &slot, m.clone(), artifact)
            .await
            .expect("catalog");
        assert_eq!(fired.load(Ordering::SeqCst), 1, "one notify per install");
        assert!(registry.get(&record.id).await.is_some(), "record upserted");

        // An unwired slot is a no-op (the host may not have assembled the feed yet).
        let empty: CatalogChangedSlot = Arc::new(std::sync::Mutex::new(None));
        notify_catalog_changed(&empty);

        let _ = std::fs::remove_file(&path);
    }

    /// Deleting a cataloged model fires the callback; deleting an unknown id does not.
    #[tokio::test]
    async fn delete_notifies_catalog_changed_only_when_a_record_was_removed() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let dir = std::env::temp_dir().join(format!(
            "daemon-models-delete-notify-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let registry_path = dir.join("catalog.json");
        // Pre-seed the registry file, then open the manager over it.
        {
            let registry = Registry::open(&registry_path).await.unwrap();
            let m = model("org/delete-me");
            let record = InstalledModel {
                id: crate::registry::model_id(&m),
                model: m,
                display_name: "org/delete-me".into(),
                local_path: dir.join("delete-me.gguf"),
                size_bytes: 1,
                quant: None,
                installed_at_ms: 1,
                arch: None,
                context_length: None,
                file_type: None,
                mmproj_path: None,
                sha256: None,
            };
            registry.upsert(record).await.unwrap();
        }
        let manager = ModelManager::new(ManagerConfig {
            cache_dir: Some(dir.clone()),
            fallback_cache_dir: None,
            registry_path: Some(registry_path),
            endpoint: None,
            quantize_worker_bin: None,
        })
        .await
        .expect("manager");

        let fired = Arc::new(AtomicUsize::new(0));
        {
            let fired = fired.clone();
            manager.set_catalog_changed(Arc::new(move || {
                fired.fetch_add(1, Ordering::SeqCst);
            }));
        }

        let id = crate::registry::model_id(&model("org/delete-me"));
        manager.delete(&id).await.expect("delete");
        assert_eq!(fired.load(Ordering::SeqCst), 1, "delete announces");

        // A second delete removes nothing — no spurious event.
        manager.delete(&id).await.expect("idempotent delete");
        assert_eq!(fired.load(Ordering::SeqCst), 1, "no-op delete stays quiet");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- artifact provenance pinning (Phase 3 / Cluster E) ------------------------------------

    /// The node-local pin sidecar path for an artifact (`<artifact>.sha256`) — computed here
    /// independently of the impl so the test pins down the on-disk convention.
    fn sidecar_of(artifact: &std::path::Path) -> PathBuf {
        let mut p = artifact.as_os_str().to_os_string();
        p.push(".sha256");
        PathBuf::from(p)
    }

    /// Lowercase-hex sha256 of `bytes` (the pin format).
    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Seed a manager over `dir` with one cataloged single-file GGUF artifact + its pin sidecar.
    /// Returns `(manager, model, artifact_path, bytes)`.
    async fn seed_pinned(dir: &std::path::Path) -> (ModelManager, ModelRef, PathBuf, Vec<u8>) {
        std::fs::create_dir_all(dir).unwrap();
        let registry_path = dir.join("catalog.json");
        let artifact = dir.join("model-Q4_K_M.gguf");
        let mut bytes = b"GGUF".to_vec();
        bytes.resize(256, 0xCD);
        std::fs::write(&artifact, &bytes).unwrap();
        std::fs::write(sidecar_of(&artifact), sha256_hex(&bytes)).unwrap();

        let m = ModelRef::new(
            ModelEngine::Llama,
            ModelSource::hf_file("org/pinned", "model-Q4_K_M.gguf"),
        );
        {
            let registry = Registry::open(&registry_path).await.unwrap();
            registry
                .upsert(InstalledModel {
                    id: crate::registry::model_id(&m),
                    model: m.clone(),
                    display_name: "org/pinned".into(),
                    local_path: artifact.clone(),
                    size_bytes: bytes.len() as u64,
                    quant: Some("Q4_K_M".into()),
                    installed_at_ms: 1,
                    arch: None,
                    context_length: None,
                    file_type: None,
                    mmproj_path: None,
                    sha256: None,
                })
                .await
                .unwrap();
        }
        let manager = ModelManager::new(ManagerConfig {
            cache_dir: Some(dir.join("hub")),
            fallback_cache_dir: None,
            registry_path: Some(registry_path),
            endpoint: None,
            quantize_worker_bin: None,
        })
        .await
        .expect("manager");
        (manager, m, artifact, bytes)
    }

    /// wire v28: `build_record` surfaces the node-local provenance pin (`<path>.sha256`) on the wire
    /// `InstalledModel.sha256` for display; an artifact with no sidecar reports `None`.
    #[test]
    fn build_record_surfaces_pin_sha256() {
        let dir =
            std::env::temp_dir().join(format!("daemon-models-pin-record-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let m = ModelRef::new(
            ModelEngine::Llama,
            ModelSource::hf_file("org/pinned", "m.gguf"),
        );

        // Pinned artifact: the sidecar hash surfaces on the record.
        let pinned = dir.join("m.gguf");
        std::fs::write(&pinned, b"gguf-bytes").unwrap();
        write_pin(&pinned, "abc123def456");
        let rec = build_record(m.clone(), pinned, 10);
        assert_eq!(rec.sha256.as_deref(), Some("abc123def456"));

        // No sidecar: sha256 is None (a legacy / unpinned install).
        let unpinned = dir.join("legacy.gguf");
        std::fs::write(&unpinned, b"gguf-bytes").unwrap();
        let rec2 = build_record(m, unpinned, 10);
        assert_eq!(rec2.sha256, None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cataloged artifact whose on-disk bytes still match its pin resolves for load.
    #[tokio::test]
    async fn valid_pinned_artifact_loads() {
        let dir =
            std::env::temp_dir().join(format!("daemon-models-pin-valid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (manager, model, artifact, _bytes) = seed_pinned(&dir).await;

        let resolved = manager.resolve(&model).await.expect("valid pin loads");
        assert_eq!(resolved.local_path, artifact);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cataloged artifact tampered after install — content swapped while size + `GGUF` magic stay
    /// intact (so the size/magic `artifact_intact` check still trusts it) — is REFUSED before load
    /// with `ModelError::Integrity`, never resolved.
    #[tokio::test]
    async fn tampered_pinned_artifact_is_refused_before_load() {
        let dir =
            std::env::temp_dir().join(format!("daemon-models-pin-tampered-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (manager, model, artifact, mut bytes) = seed_pinned(&dir).await;

        // Tamper: same length, `GGUF` magic preserved, one interior byte flipped.
        bytes[100] ^= 0xFF;
        std::fs::write(&artifact, &bytes).unwrap();

        let err = manager
            .resolve(&model)
            .await
            .expect_err("tampered artifact must be refused before load");
        assert!(
            matches!(err, ModelError::Integrity(_)),
            "expected Integrity refusal, got: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
