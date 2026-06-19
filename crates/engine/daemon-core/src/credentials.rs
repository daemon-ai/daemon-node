//! The credential provider port (§7) and the default embedded multi-key pool.
//!
//! The engine never holds raw secret material as an owned pool — it holds a [`CredentialProvider`]
//! **handle**. Standalone (L1) it is the in-tree [`EmbeddedCredentialPool`] (modeled on
//! hermes-agent's `credential_pool.py`: multi-key select, cooldown, rotation); under a host the
//! authority-backed impl is injected, and across a placement cut a brokered client round-trips to
//! the owner (host-spec §6). The unit of issuance is a [`CapabilityLease`] — a scoped, TTL-bounded
//! capability, never the long-lived key (the `Bearer` mode is the one that hands a key over, and it
//! is compensated by the audit trail, not the TTL).

use async_trait::async_trait;
use daemon_common::{
    CapabilityLease, CredError, CredId, CredMode, CredScope, LeaseSecret, ProfileRef,
};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// The engine's §7 credential port: acquire a scoped capability around a model call, release it,
/// and signal a rotatable failure so the backing pool prefers a different credential next time.
#[async_trait]
pub trait CredentialProvider: Send + Sync {
    /// Acquire a capability for `profile`, attenuated to (at most) `scope`.
    async fn acquire(
        &self,
        profile: &ProfileRef,
        scope: &CredScope,
    ) -> Result<CapabilityLease, CredError>;

    /// Release a capability (best-effort; leases also expire on their own).
    async fn release(&self, lease: &CapabilityLease);

    /// Signal that the credential behind `cap_id` failed in a rotatable way (quota/auth); the pool
    /// should mark it and prefer another credential on the next [`CredentialProvider::acquire`].
    async fn rotate(&self, profile: &ProfileRef, cap_id: &CredId);
}

/// Wall-clock milliseconds since the Unix epoch (the lease clock).
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The health of one key in the pool (mirrors `credential_pool.py` `CredentialStatus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyStatus {
    /// Usable now.
    Live,
    /// Temporarily out (quota/rate); revives after `cooldown_until_ms`.
    Exhausted,
    /// Permanently unusable (revoked/invalid).
    Dead,
}

/// One credential in the embedded pool.
#[derive(Clone, Debug)]
struct Key {
    id: CredId,
    secret: String,
    status: KeyStatus,
    cooldown_until_ms: u64,
}

/// The default embedded credential pool (L1 standalone) — multi-key selection with cooldown and
/// rotation, mirroring `credential_pool.py`. Mints `Native`-mode leases embedding the selected key
/// (the standalone engine is its own authority, so leases are self-trusted and unsigned).
pub struct EmbeddedCredentialPool {
    profile: ProfileRef,
    grant: CredScope,
    ttl_ms: u64,
    cooldown_ms: u64,
    keys: Mutex<Vec<Key>>,
}

impl EmbeddedCredentialPool {
    /// A pool over the named profile with the given keys (`(id, secret)` pairs) and a grantable
    /// scope. Leases live `ttl_ms`; an exhausted key revives after `cooldown_ms`.
    pub fn new(
        profile: impl Into<String>,
        grant: CredScope,
        keys: impl IntoIterator<Item = (String, String)>,
        ttl_ms: u64,
        cooldown_ms: u64,
    ) -> Self {
        let keys = keys
            .into_iter()
            .map(|(id, secret)| Key {
                id: CredId::new(id),
                secret,
                status: KeyStatus::Live,
                cooldown_until_ms: 0,
            })
            .collect();
        Self {
            profile: ProfileRef::new(profile),
            grant,
            ttl_ms,
            cooldown_ms,
            keys: Mutex::new(keys),
        }
    }

    /// A single-key pool over a `default` profile granting the `chat` action — the zero-config L1
    /// default so a standalone engine works without any credential wiring.
    pub fn single_key() -> Self {
        Self::new(
            "default",
            CredScope::new(["default"], ["chat"], None),
            [("default".to_string(), "embedded-key".to_string())],
            60_000,
            30_000,
        )
    }

    /// How many keys are currently live (test observability).
    pub fn live_count(&self) -> usize {
        let now = now_ms();
        self.keys
            .lock()
            .unwrap()
            .iter()
            .filter(|k| matches!(k.status, KeyStatus::Live) || revivable(k, now))
            .count()
    }

    /// Select a live key (reviving any whose cooldown has elapsed), returning its `(id, secret)`.
    fn select(&self) -> Option<(CredId, String)> {
        let now = now_ms();
        let mut keys = self.keys.lock().unwrap();
        for k in keys.iter_mut() {
            if revivable(k, now) {
                k.status = KeyStatus::Live;
            }
        }
        keys.iter()
            .find(|k| matches!(k.status, KeyStatus::Live))
            .map(|k| (k.id.clone(), k.secret.clone()))
    }

    /// Mark a key exhausted (sets a cooldown); the pool prefers others until it revives.
    fn mark_exhausted(&self, cap_id: &CredId) {
        let until = now_ms() + self.cooldown_ms;
        let mut keys = self.keys.lock().unwrap();
        if let Some(k) = keys.iter_mut().find(|k| &k.id == cap_id) {
            k.status = KeyStatus::Exhausted;
            k.cooldown_until_ms = until;
        }
    }

    /// Mark a key permanently dead (revoked/invalid).
    pub fn mark_dead(&self, cap_id: &CredId) {
        let mut keys = self.keys.lock().unwrap();
        if let Some(k) = keys.iter_mut().find(|k| &k.id == cap_id) {
            k.status = KeyStatus::Dead;
        }
    }
}

fn revivable(k: &Key, now_ms: u64) -> bool {
    matches!(k.status, KeyStatus::Exhausted) && now_ms >= k.cooldown_until_ms
}

#[async_trait]
impl CredentialProvider for EmbeddedCredentialPool {
    async fn acquire(
        &self,
        profile: &ProfileRef,
        scope: &CredScope,
    ) -> Result<CapabilityLease, CredError> {
        if profile != &self.profile {
            return Err(CredError::Unavailable(profile.to_string()));
        }
        let (cap_id, secret) = self
            .select()
            .ok_or_else(|| CredError::Unavailable(profile.to_string()))?;
        // The issued scope is the attenuation of the request against what the pool may grant.
        let issued = self.grant.intersect(scope);
        if issued.is_empty() {
            return Err(CredError::ScopeDenied);
        }
        Ok(CapabilityLease {
            cap_id,
            profile: profile.clone(),
            scope: issued,
            mode: CredMode::Native,
            expires_at_ms: now_ms() + self.ttl_ms,
            secret: Some(LeaseSecret::new(secret)),
            signature: Vec::new(),
        })
    }

    async fn release(&self, _lease: &CapabilityLease) {}

    async fn rotate(&self, _profile: &ProfileRef, cap_id: &CredId) {
        self.mark_exhausted(cap_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> EmbeddedCredentialPool {
        EmbeddedCredentialPool::new(
            "openai",
            CredScope::new(["openai"], ["chat", "embed"], Some(1_000)),
            [
                ("key-a".to_string(), "secret-a".to_string()),
                ("key-b".to_string(), "secret-b".to_string()),
            ],
            60_000,
            30_000,
        )
    }

    #[tokio::test]
    async fn acquire_attenuates_scope() {
        let p = pool();
        let req = CredScope::new(["openai", "anthropic"], ["chat"], Some(5_000));
        let lease = p.acquire(&ProfileRef::new("openai"), &req).await.unwrap();
        // Intersection: profile narrows to {openai}, actions to {chat}, ceiling to min(1000, 5000).
        assert!(lease.scope.profiles.contains("openai"));
        assert!(!lease.scope.profiles.contains("anthropic"));
        assert_eq!(lease.scope.actions.iter().cloned().collect::<Vec<_>>(), ["chat"]);
        assert_eq!(lease.scope.max_tokens, Some(1_000));
        assert!(lease.secret.is_some());
    }

    #[tokio::test]
    async fn unknown_profile_is_unavailable() {
        let p = pool();
        let req = CredScope::new(["ghost"], ["chat"], None);
        let err = p.acquire(&ProfileRef::new("ghost"), &req).await.unwrap_err();
        assert_eq!(err, CredError::Unavailable("ghost".into()));
    }

    #[tokio::test]
    async fn rotate_prefers_another_key() {
        let p = pool();
        let profile = ProfileRef::new("openai");
        let scope = CredScope::new(["openai"], ["chat"], None);

        let first = p.acquire(&profile, &scope).await.unwrap();
        // Rotate the first key out (quota); the next acquire must pick a different credential.
        p.rotate(&profile, &first.cap_id).await;
        assert_eq!(p.live_count(), 1, "one key is now cooling down");
        let second = p.acquire(&profile, &scope).await.unwrap();
        assert_ne!(
            first.cap_id, second.cap_id,
            "rotation must select a different credential"
        );
    }

    #[tokio::test]
    async fn all_dead_is_unavailable() {
        let p = pool();
        let profile = ProfileRef::new("openai");
        p.mark_dead(&CredId::new("key-a"));
        p.mark_dead(&CredId::new("key-b"));
        let err = p
            .acquire(&profile, &CredScope::new(["openai"], ["chat"], None))
            .await
            .unwrap_err();
        assert_eq!(err, CredError::Unavailable("openai".into()));
    }
}
