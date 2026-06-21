//! Profile persistence: the durable store backing the node's `ProfileApi` surface.
//!
//! A [`ProfileStore`] holds the set of [`ProfileSpec`] bundles a node knows about plus which one is
//! the active default. The file-backed implementation mirrors hermes' per-home isolation by writing
//! one `<id>.json` per profile under a profiles directory, with an `active` marker file naming the
//! default. An in-memory implementation backs ephemeral nodes (the zero-config default).
//!
//! Resolution of a `ProfileSpec` into an engine-construction `EngineProfile` lives at session open
//! (see `node_api`); this module is only concerned with storage.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use daemon_api::ProfileSpec;

/// Errors a [`ProfileStore`] can surface.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    /// No profile with that id exists.
    #[error("profile not found: {0}")]
    NotFound(String),
    /// A profile with that id already exists (on create).
    #[error("profile already exists: {0}")]
    Exists(String),
    /// Underlying I/O failure.
    #[error("profile store io: {0}")]
    Io(#[from] io::Error),
    /// (De)serialization failure.
    #[error("profile store codec: {0}")]
    Codec(String),
}

/// The durable set of agent profiles plus the active default selection.
pub trait ProfileStore: Send + Sync {
    /// All known profiles (unspecified order; callers sort for display).
    fn list(&self) -> Result<Vec<ProfileSpec>, ProfileError>;
    /// Fetch one profile by id.
    fn get(&self, id: &str) -> Result<Option<ProfileSpec>, ProfileError>;
    /// Create a new profile; errors if the id already exists.
    fn create(&self, spec: ProfileSpec) -> Result<(), ProfileError>;
    /// Replace an existing profile; errors if the id does not exist.
    fn update(&self, spec: ProfileSpec) -> Result<(), ProfileError>;
    /// Delete a profile by id (no-op if absent is treated as success by callers that check first).
    fn delete(&self, id: &str) -> Result<(), ProfileError>;
    /// The active default profile id, if one is selected.
    fn active(&self) -> Result<Option<String>, ProfileError>;
    /// Select the active default profile; errors if the id does not exist.
    fn set_active(&self, id: &str) -> Result<(), ProfileError>;

    /// Insert `spec` only if no profile with its id exists yet (idempotent seeding). Returns whether
    /// it was inserted.
    fn seed(&self, spec: ProfileSpec) -> Result<bool, ProfileError> {
        if self.get(&spec.id)?.is_some() {
            return Ok(false);
        }
        let id = spec.id.clone();
        self.create(spec)?;
        if self.active()?.is_none() {
            self.set_active(&id)?;
        }
        Ok(true)
    }
}

/// An in-memory profile store for ephemeral nodes (no persistence across restarts).
#[derive(Default)]
pub struct MemProfileStore {
    inner: RwLock<MemState>,
}

#[derive(Default)]
struct MemState {
    profiles: BTreeMap<String, ProfileSpec>,
    active: Option<String>,
}

impl MemProfileStore {
    /// An empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ProfileStore for MemProfileStore {
    fn list(&self) -> Result<Vec<ProfileSpec>, ProfileError> {
        let s = self.inner.read().unwrap();
        Ok(s.profiles.values().cloned().collect())
    }

    fn get(&self, id: &str) -> Result<Option<ProfileSpec>, ProfileError> {
        let s = self.inner.read().unwrap();
        Ok(s.profiles.get(id).cloned())
    }

    fn create(&self, spec: ProfileSpec) -> Result<(), ProfileError> {
        let mut s = self.inner.write().unwrap();
        if s.profiles.contains_key(&spec.id) {
            return Err(ProfileError::Exists(spec.id));
        }
        s.profiles.insert(spec.id.clone(), spec);
        Ok(())
    }

    fn update(&self, spec: ProfileSpec) -> Result<(), ProfileError> {
        let mut s = self.inner.write().unwrap();
        if !s.profiles.contains_key(&spec.id) {
            return Err(ProfileError::NotFound(spec.id));
        }
        s.profiles.insert(spec.id.clone(), spec);
        Ok(())
    }

    fn delete(&self, id: &str) -> Result<(), ProfileError> {
        let mut s = self.inner.write().unwrap();
        s.profiles.remove(id);
        if s.active.as_deref() == Some(id) {
            s.active = None;
        }
        Ok(())
    }

    fn active(&self) -> Result<Option<String>, ProfileError> {
        Ok(self.inner.read().unwrap().active.clone())
    }

    fn set_active(&self, id: &str) -> Result<(), ProfileError> {
        let mut s = self.inner.write().unwrap();
        if !s.profiles.contains_key(id) {
            return Err(ProfileError::NotFound(id.to_string()));
        }
        s.active = Some(id.to_string());
        Ok(())
    }
}

/// A file-backed profile store: one `<id>.json` per profile under `dir`, plus an `active` file
/// naming the default. Mirrors hermes' per-home profile isolation.
pub struct FileProfileStore {
    dir: PathBuf,
    /// Serializes mutations so create/update/active stay consistent across threads.
    lock: RwLock<()>,
}

impl FileProfileStore {
    /// Open (creating the directory if needed) a file-backed store rooted at `dir`.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, ProfileError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            lock: RwLock::new(()),
        })
    }

    fn profile_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.json", sanitize(id)))
    }

    fn active_path(&self) -> PathBuf {
        self.dir.join("active")
    }

    fn read_spec(path: &Path) -> Result<ProfileSpec, ProfileError> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| ProfileError::Codec(e.to_string()))
    }

    fn write_spec(&self, spec: &ProfileSpec) -> Result<(), ProfileError> {
        let bytes = serde_json::to_vec_pretty(spec).map_err(|e| ProfileError::Codec(e.to_string()))?;
        std::fs::write(self.profile_path(&spec.id), bytes)?;
        Ok(())
    }
}

impl ProfileStore for FileProfileStore {
    fn list(&self) -> Result<Vec<ProfileSpec>, ProfileError> {
        let _g = self.lock.read().unwrap();
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                match Self::read_spec(&path) {
                    Ok(spec) => out.push(spec),
                    Err(e) => tracing::warn!(path = %path.display(), error = %e, "skipping unreadable profile"),
                }
            }
        }
        Ok(out)
    }

    fn get(&self, id: &str) -> Result<Option<ProfileSpec>, ProfileError> {
        let _g = self.lock.read().unwrap();
        let path = self.profile_path(id);
        if !path.exists() {
            return Ok(None);
        }
        Self::read_spec(&path).map(Some)
    }

    fn create(&self, spec: ProfileSpec) -> Result<(), ProfileError> {
        let _g = self.lock.write().unwrap();
        if self.profile_path(&spec.id).exists() {
            return Err(ProfileError::Exists(spec.id));
        }
        self.write_spec(&spec)
    }

    fn update(&self, spec: ProfileSpec) -> Result<(), ProfileError> {
        let _g = self.lock.write().unwrap();
        if !self.profile_path(&spec.id).exists() {
            return Err(ProfileError::NotFound(spec.id));
        }
        self.write_spec(&spec)
    }

    fn delete(&self, id: &str) -> Result<(), ProfileError> {
        let _g = self.lock.write().unwrap();
        let path = self.profile_path(id);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        if self.active()? .as_deref() == Some(id) {
            let _ = std::fs::remove_file(self.active_path());
        }
        Ok(())
    }

    fn active(&self) -> Result<Option<String>, ProfileError> {
        let path = self.active_path();
        if !path.exists() {
            return Ok(None);
        }
        let id = std::fs::read_to_string(path)?;
        let id = id.trim().to_string();
        Ok(if id.is_empty() { None } else { Some(id) })
    }

    fn set_active(&self, id: &str) -> Result<(), ProfileError> {
        let _g = self.lock.write().unwrap();
        if !self.profile_path(id).exists() {
            return Err(ProfileError::NotFound(id.to_string()));
        }
        std::fs::write(self.active_path(), id)?;
        Ok(())
    }
}

/// Restrict a profile id to a filename-safe slug so it can key an on-disk file.
fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::ProviderSelector;

    fn sample(id: &str) -> ProfileSpec {
        ProfileSpec::new(id, ProviderSelector::Anthropic, "claude-opus-4-8")
    }

    #[test]
    fn mem_store_crud_and_active() {
        let s = MemProfileStore::new();
        assert!(s.list().unwrap().is_empty());
        s.create(sample("opus")).unwrap();
        assert!(s.create(sample("opus")).is_err());
        assert_eq!(s.get("opus").unwrap().unwrap().model, "claude-opus-4-8");
        s.set_active("opus").unwrap();
        assert_eq!(s.active().unwrap().as_deref(), Some("opus"));
        let mut updated = sample("opus");
        updated.model = "claude-3-5-sonnet-latest".into();
        s.update(updated).unwrap();
        assert_eq!(s.get("opus").unwrap().unwrap().model, "claude-3-5-sonnet-latest");
        s.delete("opus").unwrap();
        assert!(s.get("opus").unwrap().is_none());
        assert!(s.active().unwrap().is_none());
    }

    #[test]
    fn file_store_roundtrip_and_seed() {
        let dir = std::env::temp_dir().join(format!("daemon-profiles-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = FileProfileStore::open(&dir).unwrap();
        assert!(s.seed(sample("opus")).unwrap());
        assert!(!s.seed(sample("opus")).unwrap());
        assert_eq!(s.active().unwrap().as_deref(), Some("opus"));
        let reopened = FileProfileStore::open(&dir).unwrap();
        assert_eq!(reopened.get("opus").unwrap().unwrap().model, "claude-opus-4-8");
        assert_eq!(reopened.list().unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
