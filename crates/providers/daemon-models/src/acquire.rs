// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Acquisition: download a [`ModelRef`]'s files into the shared HF cache via `hf-hub`, tracking
//! per-job progress and supporting pause / resume / cancel.
//!
//! `hf-hub` owns the byte transfer (chunked, with native resume via its `*.sync.part` temp file and
//! HTTP `Range` requests), so we do **not** hand-roll a downloader. We wrap it with a job table, a
//! custom [`Progress`] sink that feeds a shared byte counter, and a `CancellationToken` per job so a
//! pause keeps the partial file (resume re-invokes `hf-hub`, which continues from the part) while a
//! cancel abandons it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use daemon_common::{DownloadId, DownloadState, DownloadStatus, ModelRef, ModelSource};
use hf_hub::api::tokio::{Api, ApiBuilder, Progress};
use hf_hub::{Repo, RepoType};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::cache::CacheConfig;
use crate::error::{from_hf, ModelError, Result};
use crate::gguf;

/// A progress callback the host wires to fan a job's [`DownloadStatus`] onto the node-wide event
/// feed (L3 `DownloadProgress`), so the client renders live progress without polling. Invoked on
/// every state transition and per-file completion.
pub type DownloadProgressCb = Arc<dyn Fn(DownloadStatus) + Send + Sync>;

/// One file to fetch for a job, with its expected size (for the progress total).
#[derive(Clone, Debug)]
pub struct PlanFile {
    /// The repo-relative path.
    pub path: String,
    /// The expected size in bytes (0 when the Hub didn't report one).
    pub size: u64,
}

/// What a job downloads: the repo coordinates plus the ordered file list.
#[derive(Clone, Debug)]
pub struct DownloadPlan {
    /// The `org/name` repo id.
    pub repo: String,
    /// The git revision to pin.
    pub revision: String,
    /// The files to fetch.
    pub files: Vec<PlanFile>,
    /// Whether the artifact is a single GGUF file (llama) vs. a repo directory (mistral.rs).
    pub single_file: bool,
}

impl DownloadPlan {
    /// The total bytes across all planned files (0s where unknown).
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// The resolved artifact a completed job yields.
#[derive(Clone, Debug)]
pub struct ResolvedArtifact {
    /// The path to load: the GGUF file (llama) or the snapshot directory (mistral.rs).
    pub local_path: PathBuf,
    /// Total bytes on disk for the cataloged files.
    pub size_bytes: u64,
}

/// Why a running download was stopped early.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StopKind {
    Pause,
    Cancel,
}

/// Shared, lock-free byte counters a [`Sink`] feeds and a status read folds.
#[derive(Default)]
struct ByteCounters {
    /// Bytes from fully-completed files.
    base: AtomicU64,
    /// Bytes downloaded in the in-flight file (reset at each file boundary).
    current: AtomicU64,
}

/// One tracked job.
struct Job {
    status: Arc<Mutex<DownloadStatus>>,
    counters: Arc<ByteCounters>,
    plan: DownloadPlan,
    model: ModelRef,
    cancel: CancellationToken,
    stop: Arc<Mutex<Option<StopKind>>>,
    /// The resolved artifact once the job completes.
    artifact: Arc<Mutex<Option<ResolvedArtifact>>>,
    /// The shared progress callback (L3): cloned from the [`Downloader`] at job creation, so a
    /// callback wired after a job started still fires (the `Arc<Mutex<Option>>` is shared).
    progress: Arc<std::sync::Mutex<Option<DownloadProgressCb>>>,
}

/// The acquisition engine: an `hf-hub` API over the shared cache plus a job table.
#[derive(Clone)]
pub struct Downloader {
    api: Api,
    jobs: Arc<Mutex<HashMap<DownloadId, Arc<Job>>>>,
    by_model: Arc<Mutex<HashMap<ModelRef, DownloadId>>>,
    next_id: Arc<AtomicU64>,
    /// The optional node-wide progress callback (L3). Interior-mutable + shared across clones/jobs so
    /// the host can wire it after the manager is built (the feed is assembled later).
    progress: Arc<std::sync::Mutex<Option<DownloadProgressCb>>>,
}

impl Downloader {
    /// Build a downloader over the shared cache (`cache.hub_dir`, `cache.token`).
    pub fn new(cache: &CacheConfig) -> Result<Self> {
        let api = ApiBuilder::new()
            .with_cache_dir(cache.hub_dir.clone())
            .with_token(cache.token.clone())
            // We render progress ourselves via the custom sink; suppress hf-hub's progress bar.
            .with_progress(false)
            .build()
            .map_err(from_hf)?;
        Ok(Self {
            api,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            by_model: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(AtomicU64::new(1)),
            progress: Arc::new(std::sync::Mutex::new(None)),
        })
    }

    /// Wire the node-wide progress callback (L3 `DownloadProgress`). Shared across clones + in-flight
    /// jobs, so it can be set after the manager is constructed (once the event feed exists).
    pub fn set_progress(&self, cb: DownloadProgressCb) {
        *self.progress.lock().unwrap() = Some(cb);
    }

    /// Start (or rejoin) a download for `model` following `plan`. Dedupes on the model reference:
    /// an existing job for the same model is returned rather than duplicated.
    pub async fn start(&self, model: ModelRef, plan: DownloadPlan) -> Result<DownloadId> {
        if let Some(existing) = self.by_model.lock().await.get(&model).copied() {
            // Rejoin an in-flight/finished job; re-kick a paused/failed one.
            let state = {
                let jobs = self.jobs.lock().await;
                jobs.get(&existing).cloned()
            };
            if let Some(job) = state {
                let s = job.status.lock().await.state.clone();
                if matches!(s, DownloadState::Paused | DownloadState::Failed) {
                    drop(job);
                    self.resume(existing).await?;
                }
                return Ok(existing);
            }
        }

        let id = DownloadId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let status = DownloadStatus {
            id,
            model: model.clone(),
            state: DownloadState::Queued,
            downloaded_bytes: 0,
            total_bytes: plan.total_bytes(),
            files_done: 0,
            files_total: plan.files.len() as u32,
            error: None,
        };
        let job = Arc::new(Job {
            status: Arc::new(Mutex::new(status)),
            counters: Arc::new(ByteCounters::default()),
            plan,
            model: model.clone(),
            cancel: CancellationToken::new(),
            stop: Arc::new(Mutex::new(None)),
            artifact: Arc::new(Mutex::new(None)),
            progress: self.progress.clone(),
        });
        self.jobs.lock().await.insert(id, job.clone());
        self.by_model.lock().await.insert(model, id);
        self.spawn_run(id, job);
        Ok(id)
    }

    /// Spawn the background transfer task for `job`.
    fn spawn_run(&self, id: DownloadId, job: Arc<Job>) {
        let api = self.api.clone();
        tokio::spawn(async move {
            run_job(api, id, job).await;
        });
    }

    /// A snapshot of one job's status.
    pub async fn status(&self, id: DownloadId) -> Option<DownloadStatus> {
        let job = self.jobs.lock().await.get(&id).cloned()?;
        Some(self.read_status(&job).await)
    }

    /// A snapshot of every job's status.
    pub async fn statuses(&self) -> Vec<DownloadStatus> {
        let jobs: Vec<Arc<Job>> = self.jobs.lock().await.values().cloned().collect();
        let mut out = Vec::with_capacity(jobs.len());
        for job in jobs {
            out.push(self.read_status(&job).await);
        }
        out
    }

    /// The resolved artifact for a completed job, if any.
    pub async fn artifact(&self, id: DownloadId) -> Option<ResolvedArtifact> {
        let job = self.jobs.lock().await.get(&id).cloned()?;
        let artifact = job.artifact.lock().await.clone();
        artifact
    }

    /// Fold the live byte counters into the stored status.
    async fn read_status(&self, job: &Arc<Job>) -> DownloadStatus {
        let mut status = job.status.lock().await.clone();
        if matches!(
            status.state,
            DownloadState::Downloading | DownloadState::Queued
        ) {
            let base = job.counters.base.load(Ordering::Relaxed);
            let current = job.counters.current.load(Ordering::Relaxed);
            status.downloaded_bytes = base + current;
        }
        status
    }

    /// Pause a running job (keep the partial file for resume).
    pub async fn pause(&self, id: DownloadId) -> Result<()> {
        let job = self.require(id).await?;
        *job.stop.lock().await = Some(StopKind::Pause);
        job.cancel.cancel();
        Ok(())
    }

    /// Cancel a job (abandon it; the partial file is left for `hf-hub` to revalidate or discard).
    pub async fn cancel(&self, id: DownloadId) -> Result<()> {
        let job = self.require(id).await?;
        *job.stop.lock().await = Some(StopKind::Cancel);
        job.cancel.cancel();
        Ok(())
    }

    /// Resume a paused/failed job by re-spawning its transfer (hf-hub continues from the part file).
    pub async fn resume(&self, id: DownloadId) -> Result<()> {
        let old = self.require(id).await?;
        // A fresh cancellation token + cleared stop reason for the new run.
        let job = Arc::new(Job {
            status: old.status.clone(),
            counters: old.counters.clone(),
            plan: old.plan.clone(),
            model: old.model.clone(),
            cancel: CancellationToken::new(),
            stop: Arc::new(Mutex::new(None)),
            artifact: old.artifact.clone(),
            progress: self.progress.clone(),
        });
        {
            let mut s = job.status.lock().await;
            s.state = DownloadState::Queued;
            s.error = None;
        }
        self.jobs.lock().await.insert(id, job.clone());
        self.spawn_run(id, job);
        Ok(())
    }

    async fn require(&self, id: DownloadId) -> Result<Arc<Job>> {
        self.jobs
            .lock()
            .await
            .get(&id)
            .cloned()
            .ok_or_else(|| ModelError::Unknown(id.to_string()))
    }
}

/// The custom progress sink: each `update` adds to the in-flight file's byte counter.
#[derive(Clone)]
struct Sink {
    counters: Arc<ByteCounters>,
}

impl Progress for Sink {
    async fn init(&mut self, _size: usize, _filename: &str) {
        self.counters.current.store(0, Ordering::Relaxed);
    }
    async fn update(&mut self, size: usize) {
        self.counters
            .current
            .fetch_add(size as u64, Ordering::Relaxed);
    }
    async fn finish(&mut self) {}
}

/// Run one job to completion (or early stop).
async fn run_job(api: Api, _id: DownloadId, job: Arc<Job>) {
    set_state(&job, DownloadState::Downloading).await;
    let repo = api.repo(Repo::with_revision(
        job.plan.repo.clone(),
        RepoType::Model,
        job.plan.revision.clone(),
    ));

    let mut last_path: Option<PathBuf> = None;
    let mut total_on_disk: u64 = 0;

    for (i, file) in job.plan.files.iter().enumerate() {
        job.counters.current.store(0, Ordering::Relaxed);
        let sink = Sink {
            counters: job.counters.clone(),
        };
        let download = repo.download_with_progress(&file.path, sink);
        tokio::pin!(download);

        let result = tokio::select! {
            biased;
            _ = job.cancel.cancelled() => {
                let kind = job.stop.lock().await.unwrap_or(StopKind::Cancel);
                let state = match kind {
                    StopKind::Pause => DownloadState::Paused,
                    StopKind::Cancel => DownloadState::Cancelled,
                };
                set_state(&job, state).await;
                return;
            }
            r = &mut download => r,
        };

        let path = match result {
            Ok(p) => p,
            Err(e) => {
                let err = from_hf(e);
                fail(&job, err.to_string()).await;
                return;
            }
        };

        // Integrity preflight on GGUF artifacts: a truncated / HTML-error download fails the magic.
        if gguf::is_gguf(&file.path) {
            match gguf::verify_gguf_magic(&path) {
                Ok(true) => {}
                Ok(false) => {
                    fail(
                        &job,
                        format!("{}: not a valid GGUF file (bad magic)", file.path),
                    )
                    .await;
                    return;
                }
                Err(e) => {
                    fail(&job, format!("{}: {e}", file.path)).await;
                    return;
                }
            }
        }

        let actual = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or(file.size);
        total_on_disk += actual;
        job.counters.base.fetch_add(actual, Ordering::Relaxed);
        job.counters.current.store(0, Ordering::Relaxed);
        {
            let mut s = job.status.lock().await;
            s.files_done = (i + 1) as u32;
            s.downloaded_bytes = job.counters.base.load(Ordering::Relaxed);
        }
        notify_progress(&job).await; // L3: per-file completion progress
        last_path = Some(path);
    }

    // Resolve the artifact: the single GGUF file, or the snapshot directory for a repo prewarm.
    let artifact = match last_path {
        Some(path) => {
            let local_path = if job.plan.single_file {
                path
            } else {
                snapshot_dir(&path, &job.plan.files)
            };
            ResolvedArtifact {
                local_path,
                size_bytes: total_on_disk,
            }
        }
        None => {
            fail(&job, "no files to download".into()).await;
            return;
        }
    };
    *job.artifact.lock().await = Some(artifact);
    set_state(&job, DownloadState::Completed).await;
}

/// Derive the snapshot directory from a downloaded pointer path by stripping the file's
/// repo-relative components (`…/snapshots/<commit>/<relpath>` → `…/snapshots/<commit>`).
fn snapshot_dir(file_path: &std::path::Path, files: &[PlanFile]) -> PathBuf {
    let depth = files
        .first()
        .map(|f| std::path::Path::new(&f.path).components().count())
        .unwrap_or(1);
    let mut dir = file_path.to_path_buf();
    for _ in 0..depth {
        dir.pop();
    }
    dir
}

async fn set_state(job: &Arc<Job>, state: DownloadState) {
    job.status.lock().await.state = state;
    notify_progress(job).await;
}

async fn fail(job: &Arc<Job>, message: String) {
    {
        let mut s = job.status.lock().await;
        s.state = DownloadState::Failed;
        s.error = Some(message);
    }
    notify_progress(job).await;
}

/// Fan the job's current status onto the wired node-wide progress callback (L3), folding the live
/// byte counters into the snapshot (mirrors [`Downloader::read_status`]). No-op when no callback is
/// wired (the lock is held only to clone the `Arc`, never across the callback).
async fn notify_progress(job: &Arc<Job>) {
    let cb = job.progress.lock().unwrap().clone();
    let Some(cb) = cb else {
        return;
    };
    let mut status = job.status.lock().await.clone();
    if matches!(
        status.state,
        DownloadState::Downloading | DownloadState::Queued
    ) {
        let base = job.counters.base.load(Ordering::Relaxed);
        let current = job.counters.current.load(Ordering::Relaxed);
        status.downloaded_bytes = base + current;
    }
    cb(status);
}

/// Build a [`DownloadPlan`] for a Hugging Face model reference. `files` is the resolved file list
/// (llama: the GGUF + its shards; mistral.rs: every repo file).
pub fn plan_for(model: &ModelRef, files: Vec<PlanFile>, single_file: bool) -> Result<DownloadPlan> {
    match &model.source {
        ModelSource::Hf { repo, revision, .. } => {
            if files.is_empty() {
                return Err(ModelError::Invalid("no files selected for download".into()));
            }
            Ok(DownloadPlan {
                repo: repo.clone(),
                revision: revision.clone(),
                files,
                single_file,
            })
        }
        ModelSource::Local { .. } => Err(ModelError::Invalid(
            "local models are already present; nothing to download".into(),
        )),
    }
}
