//! Persisted provider credentials: the durable store backing the node's `CredentialApi` surface,
//! and the [`CredentialSource`] that feeds stored secrets into the credential authority.
//!
//! A GUI sets a provider API key per profile/credential-ref via `CredentialApi`; the key lands in a
//! [`CredentialStore`] (in-memory for an ephemeral node, file-backed for a durable one). The node's
//! owner credential authority provisions secrets through a [`StoreCredentialSource`] over that store,
//! so the lease secret threaded onto each model request (`Request.auth`) is the GUI-set key.
//!
//! Scope note (v1): one authority binds one profile, so a single node serves one credential profile
//! (the launch/active profile). Per-profile distinct authorities + a fallback pool are a later
//! (P2) refinement; this store already keys by profile so that step is additive.

use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
use std::sync::RwLock;

use daemon_api::CredentialInfo;
use daemon_common::{CredError, CredId, CredMode, ProfileRef};
use daemon_credentials::{CredentialSource, Provisioned};

/// A durable map of `profile -> secret` for provider credentials.
pub trait CredentialStore: Send + Sync {
    /// Store (or replace) the secret for `profile`.
    fn set(&self, profile: &str, secret: &str) -> io::Result<()>;
    /// Fetch the secret for `profile` (the source's read path; never exposed on the wire).
    fn get(&self, profile: &str) -> Option<String>;
    /// Remove the secret for `profile`.
    fn remove(&self, profile: &str) -> io::Result<()>;
    /// A redacted listing (profiles + masked hints, never secrets).
    fn list_redacted(&self) -> Vec<CredentialInfo>;
}

/// An in-memory credential store (ephemeral nodes; secrets do not survive a restart).
#[derive(Default)]
pub struct MemCredentialStore {
    inner: RwLock<BTreeMap<String, String>>,
}

impl MemCredentialStore {
    /// An empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl CredentialStore for MemCredentialStore {
    fn set(&self, profile: &str, secret: &str) -> io::Result<()> {
        self.inner
            .write()
            .unwrap()
            .insert(profile.to_string(), secret.to_string());
        Ok(())
    }

    fn get(&self, profile: &str) -> Option<String> {
        self.inner.read().unwrap().get(profile).cloned()
    }

    fn remove(&self, profile: &str) -> io::Result<()> {
        self.inner.write().unwrap().remove(profile);
        Ok(())
    }

    fn list_redacted(&self) -> Vec<CredentialInfo> {
        self.inner
            .read()
            .unwrap()
            .iter()
            .map(|(p, s)| CredentialInfo::redacted(p, Some(s)))
            .collect()
    }
}

/// A file-backed credential store: a single JSON object (`profile -> secret`) at `path`.
///
/// Secrets are stored in plaintext at rest for v1 (an OS-keychain / sealed-secret backend is a later
/// refinement). The file is created with `0600` permissions on unix.
pub struct FileCredentialStore {
    path: PathBuf,
    lock: RwLock<()>,
}

impl FileCredentialStore {
    /// Open (creating the parent dir if needed) a file-backed store at `path`.
    pub fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self {
            path,
            lock: RwLock::new(()),
        })
    }

    fn read_map(&self) -> BTreeMap<String, String> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
            Err(_) => BTreeMap::new(),
        }
    }

    fn write_map(&self, map: &BTreeMap<String, String>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(map)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(&self.path, bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&self.path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }
}

impl CredentialStore for FileCredentialStore {
    fn set(&self, profile: &str, secret: &str) -> io::Result<()> {
        let _g = self.lock.write().unwrap();
        let mut map = self.read_map();
        map.insert(profile.to_string(), secret.to_string());
        self.write_map(&map)
    }

    fn get(&self, profile: &str) -> Option<String> {
        let _g = self.lock.read().unwrap();
        self.read_map().get(profile).cloned()
    }

    fn remove(&self, profile: &str) -> io::Result<()> {
        let _g = self.lock.write().unwrap();
        let mut map = self.read_map();
        map.remove(profile);
        self.write_map(&map)
    }

    fn list_redacted(&self) -> Vec<CredentialInfo> {
        let _g = self.lock.read().unwrap();
        self.read_map()
            .iter()
            .map(|(p, s)| CredentialInfo::redacted(p, Some(s)))
            .collect()
    }
}

/// A [`CredentialSource`] over a [`CredentialStore`]: provisions the stored secret for its bound
/// profile (falling back to a configured key when none is stored). Replaces the demo
/// `StubCredentialSource` so a GUI-set key actually reaches the provider.
pub struct StoreCredentialSource {
    store: std::sync::Arc<dyn CredentialStore>,
    profile: ProfileRef,
    fallback: String,
}

impl StoreCredentialSource {
    /// A source over `store` bound to `profile`, handing over `fallback` when no secret is stored.
    pub fn new(
        store: std::sync::Arc<dyn CredentialStore>,
        profile: impl Into<String>,
        fallback: impl Into<String>,
    ) -> Self {
        Self {
            store,
            profile: ProfileRef::new(profile),
            fallback: fallback.into(),
        }
    }
}

impl CredentialSource for StoreCredentialSource {
    fn profile(&self) -> &ProfileRef {
        &self.profile
    }

    fn provision(&self, _cap_id: &CredId, _mode: CredMode) -> Result<Provisioned, CredError> {
        // The stored key is handed over as-is (a real provider key used as the request bearer); the
        // mode-specific minting/STS dance is the stub's domain and out of scope for the GUI path.
        let secret = self
            .store
            .get(self.profile.as_str())
            .unwrap_or_else(|| self.fallback.clone());
        Ok(Provisioned {
            secret,
            fresh: false,
        })
    }

    fn revoke(&self, _cap_id: &CredId) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_store_set_get_remove_redacts() {
        let s = MemCredentialStore::new();
        s.set("opus", "sk-ant-secret-1234").unwrap();
        assert_eq!(s.get("opus").as_deref(), Some("sk-ant-secret-1234"));
        let listed = s.list_redacted();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].present);
        assert_eq!(listed[0].hint, "…1234");
        assert!(!listed[0].hint.contains("secret"));
        s.remove("opus").unwrap();
        assert!(s.get("opus").is_none());
    }

    #[test]
    fn source_provisions_stored_then_fallback() {
        use std::sync::Arc;
        let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
        let source = StoreCredentialSource::new(store.clone(), "opus", "sk-fallback");
        let cap = CredId::new("c1");
        assert_eq!(
            source.provision(&cap, CredMode::Bearer).unwrap().secret,
            "sk-fallback"
        );
        store.set("opus", "sk-real").unwrap();
        assert_eq!(
            source.provision(&cap, CredMode::Bearer).unwrap().secret,
            "sk-real"
        );
    }
}
