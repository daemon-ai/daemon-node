//! The `CredentialSource` seam: where secret material comes from, and the stub used in this phase.
//!
//! The authority owns *governance* (scoping, attenuation, minting capabilities, rotation, cost);
//! the source owns *secret provisioning* — the boundary behind which real OAuth/STS refresh and
//! real provider key-generation APIs sit (both deferred). The stub source proves the three modes:
//! a simulated short-lived `Native` token, a `Bearer` key (fresh per-grant when minting is enabled,
//! else the configured key), and the configured key for `Proxied` use.

use daemon_common::{CredError, CredId, CredMode, ProfileRef};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// The secret material a source provisions for one grant.
#[derive(Clone, Debug)]
pub struct Provisioned {
    /// The usable secret (a short-lived token for `Native`, a usable key for `Bearer`/`Proxied`).
    pub secret: String,
    /// Whether this is a fresh per-grant credential (so the source can later revoke it).
    pub fresh: bool,
}

/// Secret provisioning for one profile. Synchronous: a stub returns immediately; a real source
/// (OAuth refresh / provider key-gen) would do its I/O behind this boundary.
pub trait CredentialSource: Send + Sync {
    /// The profile this source serves.
    fn profile(&self) -> &ProfileRef;
    /// Provision secret material for `cap_id` in `mode`.
    fn provision(&self, cap_id: &CredId, mode: CredMode) -> Result<Provisioned, CredError>;
    /// Revoke a previously-provisioned fresh credential (no-op where unsupported).
    fn revoke(&self, cap_id: &CredId);
    /// Signal that the credential behind `cap_id` failed in a rotatable way (quota/auth): a
    /// multi-key (pooled) source marks the underlying key exhausted so the next `provision` prefers
    /// another. Single-key sources cannot rotate — default no-op.
    fn rotate(&self, _cap_id: &CredId) {}
}

/// A stub source over a single configured key, optionally able to "mint" fresh per-grant keys.
pub struct StubCredentialSource {
    profile: ProfileRef,
    key: String,
    can_mint: bool,
    minted: Mutex<HashMap<CredId, String>>,
    revoked: Mutex<HashSet<CredId>>,
}

impl StubCredentialSource {
    /// A source over `profile` whose configured long-lived key is `key`. Cannot mint fresh keys
    /// (so `Bearer` hands over the configured key, audit-only revocation).
    pub fn new(profile: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            profile: ProfileRef::new(profile),
            key: key.into(),
            can_mint: false,
            minted: Mutex::new(HashMap::new()),
            revoked: Mutex::new(HashSet::new()),
        }
    }

    /// As [`StubCredentialSource::new`], but able to provision a fresh per-grant key for `Bearer`
    /// (genuinely revocable at the source), simulating a provider key-generation API.
    pub fn minting(profile: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            can_mint: true,
            ..Self::new(profile, key)
        }
    }

    /// Whether `cap_id`'s fresh credential has been revoked at the source (test observability).
    pub fn is_revoked(&self, cap_id: &CredId) -> bool {
        self.revoked.lock().unwrap().contains(cap_id)
    }
}

impl CredentialSource for StubCredentialSource {
    fn profile(&self) -> &ProfileRef {
        &self.profile
    }

    fn provision(&self, cap_id: &CredId, mode: CredMode) -> Result<Provisioned, CredError> {
        match mode {
            // Simulated STS/OAuth: a fresh short-lived token bound to this grant.
            CredMode::Native => Ok(Provisioned {
                secret: format!("native-token:{cap_id}"),
                fresh: true,
            }),
            CredMode::Bearer => {
                if self.can_mint {
                    let key = format!("sk-fresh-{cap_id}");
                    self.minted
                        .lock()
                        .unwrap()
                        .insert(cap_id.clone(), key.clone());
                    Ok(Provisioned {
                        secret: key,
                        fresh: true,
                    })
                } else {
                    // Hand over the configured long-lived key (audit-only revocation).
                    Ok(Provisioned {
                        secret: self.key.clone(),
                        fresh: false,
                    })
                }
            }
            // The key never leaves the authority in proxied mode, but the authority still needs it.
            CredMode::Proxied => Ok(Provisioned {
                secret: self.key.clone(),
                fresh: false,
            }),
        }
    }

    fn revoke(&self, cap_id: &CredId) {
        self.minted.lock().unwrap().remove(cap_id);
        self.revoked.lock().unwrap().insert(cap_id.clone());
    }
}
