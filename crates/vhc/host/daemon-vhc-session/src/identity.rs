// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The certified per-run identity subsystem (architecture §4.3; ABI §12.1).
//!
//! Three separate keys, three separate lifecycles — never derived from one another and never
//! from any public value:
//!
//! - the **base identity**: one durable ed25519 key per node installation (persisted by
//!   [`crate::keystore`]), touched once per binding to issue certificates — never per frame;
//! - the **per-run key**: a fresh CSPRNG ed25519 keypair per `(run, role, incarnation)`,
//!   certified by the base identity to the full execution identity
//!   `(run_id, epoch, role, incarnation, module_hash)`; an epoch change REBINDS the same key
//!   with a new certificate, only an incarnation change rotates the key itself;
//! - the **transport identity** (iroh): its own CSPRNG secret with its own store entry
//!   (architecture §7.2 keeps transport identity distinct from signing identity).
//!
//! The old scheme — key seeds as `blake3("…/{run_id}")`, pure functions of the public run label —
//! is dead: anyone knowing a run id could impersonate any role. The negative suites in
//! `daemon-vhc-worker` and this crate pin that a reconstructed derivation of that shape can no
//! longer authenticate anywhere.

use daemon_vhc_proto::{
    peer_id, CertScope, GenesisEnvelope, PeerId, RunKeyCertificate, SigningKey, VhcProtoError,
};
use zeroize::Zeroize;

/// A 32-byte secret seed that zeroes itself on drop. The transient carrier between the CSPRNG /
/// keystore record and an [`SigningKey`] (which zeroizes its own copy on drop via
/// `ed25519-dalek`'s `zeroize` feature).
pub struct SecretSeed(pub(crate) [u8; 32]);

impl SecretSeed {
    /// A fresh CSPRNG seed from the operating system's entropy source.
    ///
    /// # Errors
    /// The OS entropy source failed (surfaced, never silently degraded).
    pub fn fresh() -> Result<Self, IdentityError> {
        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).map_err(|e| IdentityError::Entropy(e.to_string()))?;
        Ok(Self(seed))
    }

    /// Wrap existing seed bytes (e.g. a keystore record's payload) in the zeroizing carrier.
    #[must_use]
    pub fn from_bytes(seed: [u8; 32]) -> Self {
        Self(seed)
    }

    /// The ed25519 signing key this seed expands to.
    #[must_use]
    pub fn signing_key(&self) -> SigningKey {
        SigningKey::from_bytes(&self.0)
    }

    /// Read access for persistence (the keystore serializes the seed into its 0600 record).
    #[must_use]
    pub fn bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Drop for SecretSeed {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl std::fmt::Debug for SecretSeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Redaction by construction: seed material never lands in logs or debug dumps.
        f.write_str("SecretSeed(..)")
    }
}

/// An identity-subsystem failure.
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    /// The OS entropy source failed.
    #[error("entropy source: {0}")]
    Entropy(String),
    /// Certificate issuance failed (canonical encode / signing).
    #[error("certificate issue: {0}")]
    Issue(#[from] VhcProtoError),
}

/// A per-run signing key together with the certificate that authenticates it: the pair every
/// production signer carries. There is deliberately no way to construct one on a production path
/// without a certificate.
pub struct CertifiedRunKey {
    /// The per-run signing key (zeroized on drop by `ed25519-dalek`).
    pub key: SigningKey,
    /// The base-identity certificate binding [`CertifiedRunKey::key`]'s public half to the full
    /// execution identity.
    pub cert: RunKeyCertificate,
}

impl CertifiedRunKey {
    /// The certified public identity (the §12.1 frame envelope `sender`).
    #[must_use]
    pub fn sender(&self) -> PeerId {
        peer_id(&self.key)
    }
}

/// Generate a fresh CSPRNG per-run key for `scope` and certify it with `base`: the one
/// production construction path for a run signer (incarnation change = a new call here).
///
/// # Errors
/// Entropy failure or certificate issuance failure.
pub fn issue_run_key(
    base: &SigningKey,
    scope: CertScope,
) -> Result<CertifiedRunKey, IdentityError> {
    let seed = SecretSeed::fresh()?;
    let key = seed.signing_key();
    certify_existing_key(base, scope, key)
}

/// Certify an ALREADY-HELD per-run key (e.g. one recovered from the keystore after a crash, or
/// an epoch rebind) for `scope`. The key itself is not rotated — that is exactly the epoch-rebind
/// policy: same key, new binding.
///
/// # Errors
/// Certificate issuance failure.
pub fn certify_existing_key(
    base: &SigningKey,
    scope: CertScope,
    key: SigningKey,
) -> Result<CertifiedRunKey, IdentityError> {
    let cert = RunKeyCertificate::issue(base, scope, peer_id(&key))?;
    Ok(CertifiedRunKey { key, cert })
}

/// Rebind a certified run key to a new epoch (and the epoch's pinned module): the SAME key gets a
/// fresh certificate — journal identity stays stable across the fence; rotation happens only on
/// incarnation change.
///
/// # Errors
/// Certificate issuance failure.
pub fn rebind_epoch(
    base: &SigningKey,
    certified: &CertifiedRunKey,
    epoch: u64,
    module_hash: daemon_vhc_proto::Hash,
) -> Result<RunKeyCertificate, IdentityError> {
    let scope = CertScope {
        epoch,
        module_hash,
        ..certified.cert.body.scope.clone()
    };
    Ok(RunKeyCertificate::issue(base, scope, certified.sender())?)
}

/// The base identities a run participant trusts to certify per-run keys — sourced from the
/// run's genesis/Authority configuration, NEVER from ambient config. Today's genesis names the
/// coordinator base identity (`identities.coordinator` / `coordinator_set`); a fuller roster rule
/// arrives with the Authority-governed seat work.
#[derive(Debug, Clone, Default)]
pub struct TrustedBases {
    bases: Vec<PeerId>,
}

impl TrustedBases {
    /// The trusted certificate issuers named by a genesis envelope's `[identities]` section.
    #[must_use]
    pub fn from_genesis(env: &GenesisEnvelope) -> Self {
        let mut bases = Vec::new();
        if let Some(coord) = env.identities.coordinator {
            bases.push(coord);
        }
        for id in &env.identities.coordinator_set {
            if !bases.contains(id) {
                bases.push(*id);
            }
        }
        Self { bases }
    }

    /// An explicit base set (tests / node-authored trust for its own workers).
    #[must_use]
    pub fn from_bases(bases: Vec<PeerId>) -> Self {
        Self { bases }
    }

    /// Whether `base` may issue per-run certificates for this run.
    #[must_use]
    pub fn contains(&self, base: &PeerId) -> bool {
        self.bases.contains(base)
    }

    /// The trusted issuers, in genesis order.
    #[must_use]
    pub fn bases(&self) -> &[PeerId] {
        &self.bases
    }

    /// Whether the set is empty (a genesis with no named identities — nothing can certify).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_proto::{verify_certified_sender, Hash};

    fn scope(run: u8, incarnation: u64) -> CertScope {
        CertScope {
            run_id: Hash([run; 32]),
            epoch: 0,
            role: "trainer".into(),
            instance: incarnation,
            module_hash: Hash([0xAB; 32]),
        }
    }

    #[test]
    fn fresh_run_keys_are_unique_even_for_the_same_run_scope() {
        // The load-bearing property the old blake3(run label) derivation violated: joining the
        // SAME run twice must produce DIFFERENT keys — a key is never a function of run identity.
        let base = SecretSeed::fresh().unwrap().signing_key();
        let a = issue_run_key(&base, scope(1, 1)).unwrap();
        let b = issue_run_key(&base, scope(1, 2)).unwrap();
        assert_ne!(a.sender(), b.sender());
        // Even for an identical scope (two racing generations), the keys differ.
        let c = issue_run_key(&base, scope(1, 1)).unwrap();
        assert_ne!(a.sender(), c.sender());
    }

    #[test]
    fn an_issued_key_authenticates_through_its_certificate_chain() {
        let base = SecretSeed::fresh().unwrap().signing_key();
        let certified = issue_run_key(&base, scope(2, 7)).unwrap();
        let store = [certified.cert.clone()];
        verify_certified_sender(&scope(2, 7), &certified.sender(), &peer_id(&base), &store)
            .expect("the issued key is certified for exactly its scope");
        // ...and for no other scope.
        assert!(verify_certified_sender(
            &scope(2, 8),
            &certified.sender(),
            &peer_id(&base),
            &store
        )
        .is_err());
    }

    #[test]
    fn an_epoch_rebind_keeps_the_key_and_reissues_the_certificate() {
        let base = SecretSeed::fresh().unwrap().signing_key();
        let certified = issue_run_key(&base, scope(3, 1)).unwrap();
        let new_module = Hash([0xCD; 32]);
        let rebound = rebind_epoch(&base, &certified, 1, new_module).unwrap();
        // Same key...
        assert_eq!(rebound.body.run_key, certified.sender());
        // ...new binding.
        assert_eq!(rebound.body.scope.epoch, 1);
        assert_eq!(rebound.body.scope.module_hash, new_module);
        assert!(rebound.verify_chain().is_ok());
    }

    #[test]
    fn trusted_bases_come_from_the_genesis_identities_section() {
        let base_a = peer_id(&SecretSeed::fresh().unwrap().signing_key());
        let base_b = peer_id(&SecretSeed::fresh().unwrap().signing_key());
        let stranger = peer_id(&SecretSeed::fresh().unwrap().signing_key());

        let mut env = minimal_genesis();
        env.identities.coordinator = Some(base_a);
        env.identities.coordinator_set = vec![base_a, base_b];
        let trusted = TrustedBases::from_genesis(&env);
        assert!(trusted.contains(&base_a));
        assert!(trusted.contains(&base_b));
        assert!(!trusted.contains(&stranger), "ambient keys never trusted");
        assert_eq!(trusted.bases().len(), 2, "deduplicated");

        let empty = TrustedBases::from_genesis(&minimal_genesis());
        assert!(empty.is_empty());
    }

    /// A structurally-minimal genesis envelope (identity fields only are read here).
    fn minimal_genesis() -> GenesisEnvelope {
        GenesisEnvelope {
            run: daemon_vhc_proto::genesis::RunSection {
                schema: daemon_vhc_proto::genesis::GENESIS_SCHEMA_MAJOR,
                run_label: "identity-test".into(),
                min_peers: 1,
                max_peers: 2,
                access: daemon_vhc_proto::envelope::Access::Org,
            },
            roles: std::collections::BTreeMap::new(),
            artifacts: std::collections::BTreeMap::new(),
            authority: ciborium::value::Value::Null,
            transport: daemon_vhc_proto::genesis::TransportSelection::default(),
            identities: daemon_vhc_proto::genesis::Identities::default(),
        }
    }
}
