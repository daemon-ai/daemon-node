//! Persisted provider credentials: the durable store backing the node's `CredentialApi` surface,
//! and the [`CredentialSource`] that feeds stored secrets into the credential authority.
//!
//! A GUI sets a provider API key per profile/credential-ref via `CredentialApi`; the key lands in a
//! [`CredentialStore`] (in-memory for an ephemeral node, file-backed for a durable one). The node's
//! owner credential authority provisions secrets through a [`StoreCredentialSource`] over that store,
//! so the lease secret threaded onto each model request (`Request.auth`) is the GUI-set key.
//!
//! Scope note: the store keys by profile and holds a *pool* of keys per profile (multi-key), and
//! [`PooledStoreCredentialSource`] selects/rotates among them with cooldown — the owner-local
//! analogue of hermes-agent's `credential_pool.py`. A cross-credential fallback chain is layered on
//! top by the engine (`Recovery::Fallback`) via the profile's `fallback_credential_ref`.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::PathBuf;
use std::sync::{Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_api::CredentialInfo;
use daemon_common::{CredError, CredId, CredMode, ProfileRef};
use daemon_credentials::{CredentialSource, Provisioned};

/// A durable map of `profile -> [secret]` for provider credentials (a per-profile key pool).
pub trait CredentialStore: Send + Sync {
    /// Store the secret for `profile`, *replacing* any existing pool (the single-key set path).
    fn set(&self, profile: &str, secret: &str) -> io::Result<()>;
    /// Fetch the primary (first) secret for `profile` (the source's read path; never on the wire).
    fn get(&self, profile: &str) -> Option<String>;
    /// Remove all secrets for `profile`.
    fn remove(&self, profile: &str) -> io::Result<()>;
    /// A redacted listing (profiles + masked hints, never secrets).
    fn list_redacted(&self) -> Vec<CredentialInfo>;
    /// Append a key to `profile`'s pool (multi-key). Default: replace via [`CredentialStore::set`].
    fn add_key(&self, profile: &str, secret: &str) -> io::Result<()> {
        self.set(profile, secret)
    }
    /// All keys for `profile`, in insertion order — the pool a source rotates through. Default: the
    /// single stored key, if any.
    fn keys(&self, profile: &str) -> Vec<String> {
        self.get(profile).into_iter().collect()
    }
}

/// Wall-clock milliseconds since the Unix epoch (the cooldown clock).
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Map a long-lived **stored** provider key onto the requested [`CredMode`] (B4 mode-awareness).
///
/// A store-backed source holds a long-lived provider key (the GUI-set secret), not an OAuth/STS
/// session, so the three modes are honored as:
/// - `Bearer` — thread the stored key as the request bearer (compensating control is the audit);
/// - `Proxied` — return the stored key to the authority, which retains it and resolves uses without
///   it ever leaving (the authority puts no secret on the lease);
/// - `Native` — mint a short-lived token via an OAuth/STS exchange, which a store-backed profile
///   cannot do, so it is **refused** with a clear error rather than mislabelling the long-lived key
///   as a short-lived `Native` token (real OAuth/STS provisioning is deferred — ties to G3).
fn stored_secret_for_mode(
    mode: CredMode,
    profile: &str,
    secret: impl FnOnce() -> String,
) -> Result<Provisioned, CredError> {
    match mode {
        CredMode::Bearer | CredMode::Proxied => Ok(Provisioned {
            secret: secret(),
            fresh: false,
        }),
        CredMode::Native => Err(CredError::Other(format!(
            "native (short-lived) credentials require an OAuth/STS-backed source; \
             profile '{profile}' is store-backed — use Bearer or Proxied"
        ))),
    }
}

/// An in-memory credential store (ephemeral nodes; secrets do not survive a restart). Holds a
/// per-profile key pool so multi-key rotation can be exercised without a file backend.
#[derive(Default)]
pub struct MemCredentialStore {
    inner: RwLock<BTreeMap<String, Vec<String>>>,
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
            .insert(profile.to_string(), vec![secret.to_string()]);
        Ok(())
    }

    fn get(&self, profile: &str) -> Option<String> {
        self.inner
            .read()
            .unwrap()
            .get(profile)
            .and_then(|ks| ks.first().cloned())
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
            .map(|(p, ks)| CredentialInfo::redacted(p, ks.first().map(|s| s.as_str())))
            .collect()
    }

    fn add_key(&self, profile: &str, secret: &str) -> io::Result<()> {
        self.inner
            .write()
            .unwrap()
            .entry(profile.to_string())
            .or_default()
            .push(secret.to_string());
        Ok(())
    }

    fn keys(&self, profile: &str) -> Vec<String> {
        self.inner
            .read()
            .unwrap()
            .get(profile)
            .cloned()
            .unwrap_or_default()
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

    fn provision(&self, _cap_id: &CredId, mode: CredMode) -> Result<Provisioned, CredError> {
        // Honor the requested mode: the stored long-lived key backs Bearer/Proxied; Native (a
        // short-lived OAuth/STS token) is refused for a store-backed profile (B4 mode-awareness).
        stored_secret_for_mode(mode, self.profile.as_str(), || {
            self.store
                .get(self.profile.as_str())
                .unwrap_or_else(|| self.fallback.clone())
        })
    }

    fn revoke(&self, _cap_id: &CredId) {}
}

/// The pooled, rotation-aware [`CredentialSource`] over a [`CredentialStore`] — the owner-local
/// analogue of hermes-agent's `credential_pool.py`. For its bound profile it selects a live key
/// from the store's pool (skipping any cooling down), records which key served each capability, and
/// — on a rotatable failure signalled via [`CredentialSource::rotate`] — marks that key exhausted
/// for `cooldown_ms` so the next acquire prefers another. An empty pool falls back to the configured
/// `fallback` key (so a zero-config / launch-env single key still works).
pub struct PooledStoreCredentialSource {
    store: std::sync::Arc<dyn CredentialStore>,
    profile: ProfileRef,
    fallback: String,
    cooldown_ms: u64,
    state: Mutex<PoolState>,
}

#[derive(Default)]
struct PoolState {
    /// key secret -> wall-clock ms the key revives at (absent / past => live).
    cooldown: HashMap<String, u64>,
    /// capability id -> the key secret it was served (so rotation knows which key to penalize).
    served: HashMap<CredId, String>,
}

impl PooledStoreCredentialSource {
    /// A pooled source over `store` bound to `profile`, handing over `fallback` when the pool is
    /// empty. Exhausted keys revive after a 30s cooldown.
    pub fn new(
        store: std::sync::Arc<dyn CredentialStore>,
        profile: impl Into<String>,
        fallback: impl Into<String>,
    ) -> Self {
        Self::with_cooldown(store, profile, fallback, 30_000)
    }

    /// As [`PooledStoreCredentialSource::new`] with an explicit revive `cooldown_ms`.
    pub fn with_cooldown(
        store: std::sync::Arc<dyn CredentialStore>,
        profile: impl Into<String>,
        fallback: impl Into<String>,
        cooldown_ms: u64,
    ) -> Self {
        Self {
            store,
            profile: ProfileRef::new(profile),
            fallback: fallback.into(),
            cooldown_ms,
            state: Mutex::new(PoolState::default()),
        }
    }

    /// The currently-live key count for this profile (test observability).
    pub fn live_count(&self) -> usize {
        let now = now_ms();
        let st = self.state.lock().unwrap();
        let keys = self.pool();
        keys.iter()
            .filter(|k| st.cooldown.get(*k).copied().unwrap_or(0) <= now)
            .count()
    }

    fn pool(&self) -> Vec<String> {
        let keys = self.store.keys(self.profile.as_str());
        if keys.is_empty() {
            vec![self.fallback.clone()]
        } else {
            keys
        }
    }
}

impl CredentialSource for PooledStoreCredentialSource {
    fn profile(&self) -> &ProfileRef {
        &self.profile
    }

    fn provision(&self, cap_id: &CredId, mode: CredMode) -> Result<Provisioned, CredError> {
        // Native (short-lived OAuth/STS) is not provisionable from a store-backed key pool; Bearer
        // and Proxied both draw a live key from the pool (B4 mode-awareness).
        if matches!(mode, CredMode::Native) {
            return stored_secret_for_mode(mode, self.profile.as_str(), String::new);
        }
        let now = now_ms();
        let keys = self.pool();
        let mut st = self.state.lock().unwrap();
        // Prefer the first live key; if every key is cooling down, serve the soonest-reviving one
        // (best effort) rather than failing — the engine's retry/backoff still applies.
        let chosen = keys
            .iter()
            .find(|k| st.cooldown.get(*k).copied().unwrap_or(0) <= now)
            .cloned()
            .or_else(|| {
                keys.iter()
                    .min_by_key(|k| st.cooldown.get(*k).copied().unwrap_or(0))
                    .cloned()
            })
            .ok_or_else(|| CredError::Unavailable(self.profile.to_string()))?;
        st.served.insert(cap_id.clone(), chosen.clone());
        Ok(Provisioned {
            secret: chosen,
            fresh: false,
        })
    }

    fn revoke(&self, cap_id: &CredId) {
        self.state.lock().unwrap().served.remove(cap_id);
    }

    fn rotate(&self, cap_id: &CredId) {
        let until = now_ms() + self.cooldown_ms;
        let mut st = self.state.lock().unwrap();
        if let Some(key) = st.served.get(cap_id).cloned() {
            st.cooldown.insert(key, until);
        }
    }
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

    #[test]
    fn mem_store_multi_key_pool() {
        let s = MemCredentialStore::new();
        s.set("grok", "key-a").unwrap();
        s.add_key("grok", "key-b").unwrap();
        assert_eq!(s.keys("grok"), vec!["key-a".to_string(), "key-b".to_string()]);
        // The redacted listing collapses to one entry per profile (the primary key's hint).
        assert_eq!(s.list_redacted().len(), 1);
        // `set` replaces the whole pool.
        s.set("grok", "key-c").unwrap();
        assert_eq!(s.keys("grok"), vec!["key-c".to_string()]);
    }

    #[test]
    fn pooled_source_rotates_to_another_key() {
        use std::sync::Arc;
        let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
        store.set("grok", "key-a").unwrap();
        store.add_key("grok", "key-b").unwrap();
        let source = PooledStoreCredentialSource::new(store, "grok", "sk-fallback");
        assert_eq!(source.live_count(), 2);

        // First acquire serves the primary key.
        let c1 = CredId::new("c1");
        let first = source.provision(&c1, CredMode::Bearer).unwrap().secret;
        assert_eq!(first, "key-a");

        // A rotatable failure on c1 cools the primary down; the next acquire prefers the other key.
        source.rotate(&c1);
        assert_eq!(source.live_count(), 1, "the primary is cooling down");
        let c2 = CredId::new("c2");
        let second = source.provision(&c2, CredMode::Bearer).unwrap().secret;
        assert_eq!(second, "key-b");
        assert_ne!(first, second, "rotation must select a different key");
    }

    #[test]
    fn pooled_source_falls_back_when_pool_empty() {
        use std::sync::Arc;
        let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
        let source = PooledStoreCredentialSource::new(store, "grok", "sk-fallback");
        let cap = CredId::new("c1");
        assert_eq!(
            source.provision(&cap, CredMode::Bearer).unwrap().secret,
            "sk-fallback"
        );
    }

    #[test]
    fn store_source_is_mode_aware() {
        use std::sync::Arc;
        let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
        store.set("opus", "sk-real").unwrap();
        let source = StoreCredentialSource::new(store, "opus", "sk-fallback");
        let cap = CredId::new("c1");
        // Bearer + Proxied both hand the authority the stored long-lived key.
        assert_eq!(
            source.provision(&cap, CredMode::Bearer).unwrap().secret,
            "sk-real"
        );
        assert_eq!(
            source.provision(&cap, CredMode::Proxied).unwrap().secret,
            "sk-real"
        );
        // Native (short-lived OAuth/STS) is refused for a store-backed profile.
        let err = source.provision(&cap, CredMode::Native).unwrap_err();
        assert!(
            matches!(err, CredError::Other(ref m) if m.contains("OAuth/STS")),
            "Native must be refused with a clear error, got {err:?}"
        );
    }

    #[test]
    fn pooled_source_is_mode_aware() {
        use std::sync::Arc;
        let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
        store.set("grok", "key-a").unwrap();
        let source = PooledStoreCredentialSource::new(store, "grok", "sk-fallback");
        let cap = CredId::new("c1");
        assert_eq!(
            source.provision(&cap, CredMode::Proxied).unwrap().secret,
            "key-a"
        );
        let err = source.provision(&cap, CredMode::Native).unwrap_err();
        assert!(matches!(err, CredError::Other(_)), "Native must be refused");
    }
}
