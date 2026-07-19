// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Per-run identity provisioning (architecture §6.3.1; D-P8) — the ONE place a run instance's
//! signing identity is minted and certified.
//!
//! The NODE calls this at join authorship: it mints the CSPRNG per-run key for
//! `(run, role, incarnation)`, issues its [`RunKeyCertificate`] under the node's durable base
//! identity, and persists both in the keystore. The base identity never leaves the node process;
//! a worker subprocess resolves the key + certificate READ-ONLY by reference and never mints
//! (closing the base-key-custody gap: nothing in the sandboxing target ever reads `base.key`).
//!
//! Idempotent within an incarnation: the keystore recovers the same key on a re-provision (a
//! crash-safe resume mints nothing new), and the certificate is re-issued over it.

use daemon_vhc_proto::{peer_id, CertScope, Hash, RunKeyCertificate, SigningKey};

use crate::identity::certify_existing_key;
use crate::keystore::{KeystoreError, VhcKeystore};

/// A provisioning failure.
#[derive(Debug, thiserror::Error)]
pub enum ProvisionError {
    /// A keystore filesystem / record failure.
    #[error("keystore: {0}")]
    Keystore(#[from] KeystoreError),
    /// Certificate issuance failed (entropy / canonical encode / signing).
    #[error("certificate issue: {0}")]
    Issue(#[from] crate::identity::IdentityError),
}

/// The execution identity to provision.
pub struct ProvisionScope<'a> {
    /// The run label (keystore namespace).
    pub run_label: &'a str,
    /// The run's cryptographic identity (genesis hash).
    pub genesis_hash: [u8; 32],
    /// The transition-chain epoch.
    pub epoch: u64,
    /// The envelope role label.
    pub role: &'a str,
    /// The node-minted, never-reused incarnation.
    pub incarnation: u64,
    /// The pinned module hash.
    pub module_hash: [u8; 32],
}

/// Mint (or recover) the per-run key for `scope`, issue its certificate under the keystore's base
/// identity, persist both, and return the certificate. The signing key stays in the keystore —
/// callers that need to sign resolve it there (a worker does so read-only).
///
/// # Errors
/// A keystore or certificate-issuance failure.
pub fn provision_run_identity(
    keystore: &VhcKeystore,
    scope: &ProvisionScope<'_>,
) -> Result<RunKeyCertificate, ProvisionError> {
    let run_key = keystore.run_signing_key(scope.run_label, scope.role, scope.incarnation)?;
    let base = keystore.base_identity()?;
    let certified = certify_existing_key(
        &base,
        CertScope {
            run_id: Hash(scope.genesis_hash),
            epoch: scope.epoch,
            role: scope.role.to_string(),
            instance: scope.incarnation,
            module_hash: Hash(scope.module_hash),
        },
        SigningKey::from_bytes(&run_key.to_bytes()),
    )?;
    debug_assert_eq!(certified.cert.body.run_key, peer_id(&run_key));
    keystore.store_run_certificate(
        scope.run_label,
        scope.role,
        scope.incarnation,
        &certified.cert,
    )?;
    Ok(certified.cert)
}
