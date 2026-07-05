// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: fs here is the model-catalog manifest under the node data root (daemon-controlled path),
// not attacker-influenced; raw fs allowed file-wide. No process spawns in this file.
#![allow(clippy::disallowed_methods)]

//! The installed-model catalog: a single atomic JSON manifest of [`InstalledModel`] records.
//!
//! Chosen over SQLite (the old app's store) because the catalog is small, read-mostly, and benefits
//! from being human-inspectable; durability is a temp-file + atomic-rename swap. The catalog id is
//! content-derived from the [`ModelRef`] so the same model dedupes to one record regardless of which
//! profile installed it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use daemon_common::{InstalledModel, ModelId, ModelRef, ModelSource};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::error::{ModelError, Result};

/// The stable, content-derived catalog id for a model reference (engine + source).
pub fn model_id(model: &ModelRef) -> ModelId {
    let mut hasher = Sha256::new();
    hasher.update(model.engine.as_str().as_bytes());
    hasher.update([0u8]);
    match &model.source {
        ModelSource::Hf {
            repo,
            file,
            revision,
        } => {
            hasher.update(b"hf\0");
            hasher.update(repo.as_bytes());
            hasher.update([0u8]);
            hasher.update(file.as_deref().unwrap_or("").as_bytes());
            hasher.update([0u8]);
            hasher.update(revision.as_bytes());
        }
        ModelSource::Local { path } => {
            hasher.update(b"local\0");
            hasher.update(path.to_string_lossy().as_bytes());
        }
    }
    let digest = hasher.finalize();
    // A short, stable hex handle (first 16 bytes is ample for collision-freedom in a local catalog).
    let hex: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
    ModelId::new(hex)
}

/// The installed-model catalog backed by an atomic JSON manifest.
#[derive(Clone)]
pub struct Registry {
    path: PathBuf,
    models: Arc<ArcSwap<BTreeMap<String, InstalledModel>>>,
    persist_lock: Arc<Mutex<()>>,
}

impl Registry {
    /// Open (or create) the catalog at `path`, loading any existing records.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let models = load(&path)?;
        Ok(Self {
            path,
            models: Arc::new(ArcSwap::from_pointee(models)),
            persist_lock: Arc::new(Mutex::new(())),
        })
    }

    /// All installed models, ordered by id.
    pub async fn list(&self) -> Vec<InstalledModel> {
        self.models.load().values().cloned().collect()
    }

    /// Look up an installed model by its catalog id.
    pub async fn get(&self, id: &ModelId) -> Option<InstalledModel> {
        self.models.load().get(id.as_str()).cloned()
    }

    /// Find an installed model by its reference (the dedupe key acquisition checks).
    pub async fn find(&self, model: &ModelRef) -> Option<InstalledModel> {
        let id = model_id(model);
        self.get(&id).await
    }

    /// Insert or replace a record, persisting the manifest atomically.
    pub async fn upsert(&self, record: InstalledModel) -> Result<()> {
        let _guard = self.persist_lock.lock().await;
        let mut next = (**self.models.load()).clone();
        next.insert(record.id.0.clone(), record);
        persist(&self.path, &next)?;
        self.models.store(Arc::new(next));
        Ok(())
    }

    /// Remove a record by id (returns the removed record), persisting the manifest. The on-disk
    /// artifact is *not* deleted here — the manager removes the cached files separately.
    pub async fn remove(&self, id: &ModelId) -> Result<Option<InstalledModel>> {
        let _guard = self.persist_lock.lock().await;
        let mut next = (**self.models.load()).clone();
        let removed = next.remove(id.as_str());
        if removed.is_some() {
            persist(&self.path, &next)?;
            self.models.store(Arc::new(next));
        }
        Ok(removed)
    }
}

/// Load the manifest (an empty catalog when the file is absent).
fn load(path: &Path) -> Result<BTreeMap<String, InstalledModel>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let records: Vec<InstalledModel> = serde_json::from_slice(&bytes)
                .map_err(|e| ModelError::Decode(format!("catalog manifest: {e}")))?;
            Ok(records.into_iter().map(|r| (r.id.0.clone(), r)).collect())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(ModelError::io(path, e)),
    }
}

/// Persist the manifest via a temp-file + atomic rename.
fn persist(path: &Path, models: &BTreeMap<String, InstalledModel>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ModelError::io(parent, e))?;
    }
    let records: Vec<&InstalledModel> = models.values().collect();
    let json = serde_json::to_vec_pretty(&records)
        .map_err(|e| ModelError::Other(format!("encode catalog: {e}")))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &json).map_err(|e| ModelError::io(&tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| ModelError::io(path, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::ModelEngine;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "daemon-models-test-{}-{}.json",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn sample(repo: &str) -> InstalledModel {
        let model = ModelRef::new(ModelEngine::Llama, ModelSource::hf_file(repo, "m.gguf"));
        InstalledModel {
            id: model_id(&model),
            model,
            display_name: repo.to_string(),
            local_path: PathBuf::from("/tmp/m.gguf"),
            size_bytes: 10,
            quant: Some("Q4_K_M".into()),
            installed_at_ms: 1,
            arch: None,
            context_length: None,
            file_type: None,
            mmproj_path: None,
        }
    }

    #[test]
    fn ids_are_stable_and_distinct() {
        let a = ModelRef::new(ModelEngine::Llama, ModelSource::hf_file("o/r", "a.gguf"));
        let b = ModelRef::new(ModelEngine::Llama, ModelSource::hf_file("o/r", "b.gguf"));
        assert_eq!(model_id(&a), model_id(&a));
        assert_ne!(model_id(&a), model_id(&b));
    }

    #[tokio::test]
    async fn upsert_remove_roundtrip_persists() {
        let path = temp_path("roundtrip");
        let reg = Registry::open(&path).await.unwrap();
        let rec = sample("org/repo");
        let id = rec.id.clone();
        reg.upsert(rec.clone()).await.unwrap();
        assert!(reg.find(&rec.model).await.is_some());

        // Reopen from disk: the record survives.
        let reopened = Registry::open(&path).await.unwrap();
        assert_eq!(reopened.list().await.len(), 1);
        assert!(reopened.get(&id).await.is_some());

        reopened.remove(&id).await.unwrap();
        assert!(reopened.get(&id).await.is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn concurrent_reads_see_valid_snapshots_during_upsert() {
        let path = temp_path("concurrent");
        let reg = Registry::open(&path).await.unwrap();
        let first = sample("org/first");
        reg.upsert(first.clone()).await.unwrap();

        let reader_reg = reg.clone();
        let reader = tokio::spawn(async move {
            for _ in 0..100 {
                let listed = reader_reg.list().await;
                assert!(!listed.is_empty());
                assert!(reader_reg.find(&first.model).await.is_some());
                tokio::task::yield_now().await;
            }
        });

        let second = sample("org/second");
        reg.upsert(second.clone()).await.unwrap();
        reader.await.unwrap();
        assert!(reg.find(&second.model).await.is_some());
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn failed_persist_does_not_publish_snapshot() {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "daemon-models-test-{}-persist-failure-dir",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let existing = sample("org/existing");
        let mut models = BTreeMap::new();
        models.insert(existing.id.0.clone(), existing.clone());
        let reg = Registry {
            path: dir.clone(),
            models: Arc::new(ArcSwap::from_pointee(models)),
            persist_lock: Arc::new(Mutex::new(())),
        };

        let new = sample("org/new");
        assert!(reg.upsert(new.clone()).await.is_err());
        assert!(reg.find(&existing.model).await.is_some());
        assert!(reg.find(&new.model).await.is_none());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(dir.with_extension("json.tmp"));
    }
}
