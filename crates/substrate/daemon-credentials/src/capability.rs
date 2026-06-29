// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Capability signing + verification (the same ed25519 / Gordian Envelope stack as the phase-6
//! trace journal).
//!
//! A [`CapabilityLease`]'s signature is an ed25519 signature over the **digest of a deterministic
//! Gordian Envelope** of its fields (`cap_id`, `profile`, `mode`, `expires_at`, a scope fingerprint,
//! and the embedded secret). Because the digest covers the secret and the scope, editing either —
//! or the expiry — invalidates the signature; a holder several cuts down can therefore verify a
//! capability minted by the owner without trusting the intermediates that relayed it.

use bc_components::{
    PrivateKeyBase, Signature, Signer, SigningPrivateKey, SigningPublicKey, Verifier,
};
use bc_envelope::prelude::*;
use daemon_common::{CapabilityLease, CredError, CredMode, CredScope};

fn digest_to_32(d: &Digest) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(d.data());
    out
}

fn mode_tag(mode: CredMode) -> &'static str {
    match mode {
        CredMode::Native => "native",
        CredMode::Bearer => "bearer",
        CredMode::Proxied => "proxied",
    }
}

/// A deterministic string fingerprint of a scope (profiles + actions are sorted `BTreeSet`s, so
/// this is reproducible across processes — the signable form of the scope).
fn scope_fingerprint(scope: &CredScope) -> String {
    let profiles = scope.profiles.iter().cloned().collect::<Vec<_>>().join(",");
    let actions = scope.actions.iter().cloned().collect::<Vec<_>>().join(",");
    let tokens = scope
        .max_tokens
        .map(|t| t.to_string())
        .unwrap_or_else(|| "inf".into());
    format!("p[{profiles}]a[{actions}]t[{tokens}]")
}

/// The 32-byte digest the authority signs (and a verifier recomputes) for a capability. The
/// `signature` field is excluded by construction — only the capability's authority-set fields.
fn capability_digest(lease: &CapabilityLease) -> [u8; 32] {
    let secret = lease.secret.as_ref().map(|s| s.expose()).unwrap_or("");
    let env = Envelope::new(lease.cap_id.as_str())
        .add_assertion("profile", lease.profile.as_str())
        .add_assertion("mode", mode_tag(lease.mode))
        .add_assertion("expires", lease.expires_at_ms)
        .add_assertion("scope", scope_fingerprint(&lease.scope))
        .add_assertion("secret", secret);
    digest_to_32(&env.digest())
}

/// The authority's signing key — mints the detached signature carried in a `CapabilityLease`.
pub struct CapabilitySigner {
    private: SigningPrivateKey,
    public: SigningPublicKey,
}

impl Default for CapabilitySigner {
    fn default() -> Self {
        Self::generate()
    }
}

impl CapabilitySigner {
    /// Generate a fresh ed25519 capability-signing key.
    pub fn generate() -> Self {
        let private = PrivateKeyBase::new().ed25519_signing_private_key();
        let public = private
            .public_key()
            .expect("ed25519 private key yields a public key");
        Self { private, public }
    }

    /// The verifying half, distributed to capability holders.
    pub fn verifying_key(&self) -> CapabilityVerifyingKey {
        CapabilityVerifyingKey(self.public.clone())
    }

    /// Sign a fully-populated lease (its `signature` field is ignored), returning the detached
    /// signature bytes to store in `CapabilityLease::signature`.
    pub fn sign(&self, lease: &CapabilityLease) -> Vec<u8> {
        let digest = capability_digest(lease);
        let msg: &[u8] = &digest;
        let sig: Signature = self
            .private
            .sign(&msg)
            .expect("ed25519 signing is infallible for valid keys");
        sig.to_cbor_data()
    }
}

/// The public half of a [`CapabilitySigner`]: lets any holder verify a capability minted by the
/// authority, several cuts away, without trusting the relays in between.
#[derive(Clone)]
pub struct CapabilityVerifyingKey(SigningPublicKey);

impl CapabilityVerifyingKey {
    /// Verify a capability end-to-end against `now_ms`: signature over the recomputed digest, then
    /// expiry. Returns the precise [`CredError`] so the gate can assert tamper vs expiry.
    pub fn verify(&self, lease: &CapabilityLease, now_ms: u64) -> Result<(), CredError> {
        let digest = capability_digest(lease);
        let Ok(cbor) = CBOR::try_from_data(&lease.signature) else {
            return Err(CredError::BadSignature);
        };
        let Ok(sig) = Signature::try_from(cbor) else {
            return Err(CredError::BadSignature);
        };
        let msg: &[u8] = &digest;
        if !self.0.verify(&sig, &msg) {
            return Err(CredError::BadSignature);
        }
        if lease.is_expired(now_ms) {
            return Err(CredError::Expired);
        }
        Ok(())
    }
}
