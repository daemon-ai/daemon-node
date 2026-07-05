// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Offline local quantization orchestration.
//!
//! The daemon never links `llama-cpp-4` itself; the only binary that does is the `daemon-infer`
//! worker. So quantization runs out-of-process: this [`Quantizer`] resolves a high-precision GGUF
//! source (done by the caller), spawns `worker_bin quantize --in … --out … --ftype …`, verifies the
//! produced GGUF, and catalogs it as a local `InstalledModel`. Jobs are tracked in a table mirroring
//! [`crate::acquire::Downloader`] so a client can poll progress.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_common::{
    InstalledModel, ModelEngine, ModelRef, ModelSource, QuantizeId, QuantizeState, QuantizeStatus,
};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::error::{ModelError, Result};
use crate::registry::{model_id, Registry};
use crate::{gguf, inspect};

/// One quantization request: a resolved high-precision GGUF source and the target precision.
#[derive(Clone, Debug)]
pub struct QuantizeRequest {
    /// The source `org/name` repo (recorded on the job + the produced record's provenance).
    pub repo: String,
    /// The source GGUF file path within the repo (provenance).
    pub source_file: String,
    /// The resolved local path of the high-precision source GGUF.
    pub source_path: PathBuf,
    /// The target quant label (e.g. `Q4_K_M`).
    pub target_quant: String,
    /// Quantizer thread count (`0` = auto).
    pub nthread: i32,
}

/// The quantization engine: a worker-bin path, an output directory, the catalog, and a job table.
#[derive(Clone)]
pub struct Quantizer {
    worker_bin: Option<PathBuf>,
    output_dir: PathBuf,
    registry: Registry,
    jobs: Arc<Mutex<HashMap<QuantizeId, Arc<Mutex<QuantizeStatus>>>>>,
    next_id: Arc<AtomicU64>,
}

impl Quantizer {
    /// Build a quantizer. `worker_bin` is the `daemon-infer` binary (built with `--features llama`);
    /// `None` makes [`start`](Self::start) error clearly. Outputs land under `output_dir`.
    pub fn new(worker_bin: Option<PathBuf>, output_dir: PathBuf, registry: Registry) -> Self {
        Self {
            worker_bin,
            output_dir,
            registry,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Start a quantization job, returning its handle. The job runs in the background; poll it via
    /// [`statuses`](Self::statuses) / [`status`](Self::status).
    pub async fn start(&self, req: QuantizeRequest) -> Result<QuantizeId> {
        let worker_bin = self.worker_bin.clone().ok_or_else(|| {
            ModelError::Invalid(
                "quantization unavailable: no llama-enabled worker binary is configured \
                 (set local.worker_bin / DAEMON_INFER_WORKER_BIN)"
                    .into(),
            )
        })?;
        if !req.source_path.exists() {
            return Err(ModelError::Invalid(format!(
                "source GGUF does not exist: {}",
                req.source_path.display()
            )));
        }

        let id = QuantizeId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let output_path = self
            .output_dir
            .join(output_filename(&req.repo, &req.target_quant));
        let status = Arc::new(Mutex::new(QuantizeStatus {
            id,
            repo: req.repo.clone(),
            source_file: req.source_file.clone(),
            target_quant: req.target_quant.clone(),
            state: QuantizeState::Queued,
            output_path: None,
            model_id: None,
            error: None,
        }));
        self.jobs.lock().await.insert(id, status.clone());

        let registry = self.registry.clone();
        let output_dir = self.output_dir.clone();
        tokio::spawn(async move {
            run_job(worker_bin, output_dir, output_path, req, status, registry).await;
        });
        Ok(id)
    }

    /// A snapshot of every quantization job.
    pub async fn statuses(&self) -> Vec<QuantizeStatus> {
        let jobs = self.jobs.lock().await;
        let mut out = Vec::with_capacity(jobs.len());
        for job in jobs.values() {
            out.push(job.lock().await.clone());
        }
        out.sort_by_key(|s| s.id.0);
        out
    }

    /// A snapshot of one job, if known.
    pub async fn status(&self, id: QuantizeId) -> Option<QuantizeStatus> {
        let job = self.jobs.lock().await.get(&id).cloned()?;
        let snapshot = job.lock().await.clone();
        Some(snapshot)
    }
}

/// Drive a single quantization: spawn the worker, verify the output, catalog it.
async fn run_job(
    worker_bin: PathBuf,
    output_dir: PathBuf,
    output_path: PathBuf,
    req: QuantizeRequest,
    status: Arc<Mutex<QuantizeStatus>>,
    registry: Registry,
) {
    set_state(&status, QuantizeState::Quantizing).await;

    // Output dir is under the daemon-controlled model cache; not attacker-influenced.
    #[allow(clippy::disallowed_methods)]
    let mk = tokio::fs::create_dir_all(&output_dir).await;
    if let Err(e) = mk {
        return fail(&status, format!("creating output dir: {e}")).await;
    }

    // Spawns the daemon-controlled quantize worker binary (argv-only, no shell).
    #[allow(clippy::disallowed_methods)]
    let spawn = Command::new(&worker_bin)
        .arg("quantize")
        .arg("--in")
        .arg(&req.source_path)
        .arg("--out")
        .arg(&output_path)
        .arg("--ftype")
        .arg(&req.target_quant)
        .arg("--nthread")
        .arg(req.nthread.to_string())
        .output()
        .await;

    let out = match spawn {
        Ok(out) => out,
        Err(e) => {
            return fail(
                &status,
                format!("spawning worker {}: {e}", worker_bin.display()),
            )
            .await
        }
    };
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: String = stderr.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
        return fail(
            &status,
            format!("worker exited with {}: {tail}", out.status),
        )
        .await;
    }

    // Verify the produced file is a real GGUF before cataloging.
    match gguf::verify_gguf_magic(&output_path) {
        Ok(true) => {}
        Ok(false) => return fail(&status, "produced file is not a valid GGUF".into()).await,
        Err(e) => return fail(&status, format!("reading produced file: {e}")).await,
    }

    let model = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::Local {
            path: output_path.clone(),
        },
    );
    let size_bytes = tokio::fs::metadata(&output_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    let mut record = InstalledModel {
        id: model_id(&model),
        display_name: format!("{} ({})", req.repo, req.target_quant),
        local_path: output_path.clone(),
        size_bytes,
        quant: Some(req.target_quant.clone()),
        installed_at_ms: now_ms(),
        arch: None,
        context_length: None,
        file_type: None,
        mmproj_path: None,
        model,
    };
    inspect::enrich_installed(&mut record);
    let record_id = record.id.clone();
    if let Err(e) = registry.upsert(record).await {
        return fail(&status, format!("cataloging quantized model: {e}")).await;
    }

    let mut s = status.lock().await;
    s.state = QuantizeState::Completed;
    s.output_path = Some(output_path);
    s.model_id = Some(record_id);
}

/// Set a job's state.
async fn set_state(status: &Arc<Mutex<QuantizeStatus>>, state: QuantizeState) {
    status.lock().await.state = state;
}

/// Mark a job failed with `reason`.
async fn fail(status: &Arc<Mutex<QuantizeStatus>>, reason: String) {
    tracing::warn!(error = %reason, "quantization failed");
    let mut s = status.lock().await;
    s.state = QuantizeState::Failed;
    s.error = Some(reason);
}

/// The output filename for a quantized model: `<org__name>-<quant>.gguf`.
fn output_filename(repo: &str, quant: &str) -> String {
    let sanitized = repo.replace(['/', ' '], "__");
    format!("{sanitized}-{quant}.gguf")
}

/// Milliseconds since the Unix epoch.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_filename_sanitizes_repo() {
        assert_eq!(
            output_filename("TheBloke/Llama-2-7B-GGUF", "Q4_K_M"),
            "TheBloke__Llama-2-7B-GGUF-Q4_K_M.gguf"
        );
    }

    #[tokio::test]
    async fn start_without_worker_errors() {
        let dir = std::env::temp_dir().join("daemon-models-qz-test");
        let registry = Registry::open(dir.join("cat.json")).await.unwrap();
        let qz = Quantizer::new(None, dir.clone(), registry);
        let src = dir.join("src.gguf");
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(&src, b"GGUF").await.unwrap();
        let err = qz
            .start(QuantizeRequest {
                repo: "org/m".into(),
                source_file: "m-F16.gguf".into(),
                source_path: src,
                target_quant: "Q4_K_M".into(),
                nthread: 0,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ModelError::Invalid(_)));
    }
}
