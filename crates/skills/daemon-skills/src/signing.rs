// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Optional skill-bundle signing (wire v28) — the verify-at-import gate.
//!
//! A [`SkillBundle`]'s signature is an ed25519 signature over the **digest of a deterministic Gordian
//! Envelope** of its content (name + category + every file), reusing the exact `bc-components` /
//! `bc-envelope` stack as the `daemon-credentials` capability leases and the `daemon-telemetry` trace
//! journal. Because the digest covers the identity and every file, editing any of them invalidates
//! the signature.
//!
//! This is **opt-in and default-off**: a [`SkillStore`](crate::SkillStore) enforces it only when an
//! operator has attached a [`SkillBundleVerifier`] (a trusted public key). With no verifier
//! configured, `import_bundle` behaves exactly as before and unsigned bundles import normally.

use bc_components::{
    PrivateKeyBase, Signature, Signer, SigningPrivateKey, SigningPublicKey, Verifier,
};
use bc_envelope::prelude::*;
use daemon_common::SkillBundle;

use crate::SkillError;

/// The 32-byte digest a signer signs and a verifier recomputes for a bundle. Covers the identity
/// (name + category) and every file (bundle-relative path -> content); the `signature` field is
/// excluded by construction. Gordian Envelope digests are set-based, so file order is irrelevant.
fn bundle_digest(bundle: &SkillBundle) -> [u8; 32] {
    let mut env = Envelope::new(bundle.name.as_str())
        .add_assertion("category", bundle.category.as_deref().unwrap_or(""));
    for (path, content) in &bundle.files {
        env = env.add_assertion(format!("file:{path}"), content.as_str());
    }
    let digest = env.digest();
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.data());
    out
}

/// Lowercase-hex encode (mirrors the `daemon-telemetry` / `daemon-credentials` convention).
fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Decode lowercase/uppercase hex; `None` on any non-hex or odd-length input.
fn from_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

/// Signs skill bundles (tooling / tests / an operator's out-of-band signing step). Produces the
/// hex-encoded detached signature stored in [`SkillBundle::signature`].
pub struct SkillBundleSigner {
    private: SigningPrivateKey,
    public: SigningPublicKey,
}

impl Default for SkillBundleSigner {
    fn default() -> Self {
        Self::generate()
    }
}

impl SkillBundleSigner {
    /// Generate a fresh ed25519 skill-signing key.
    pub fn generate() -> Self {
        let private = PrivateKeyBase::new().ed25519_signing_private_key();
        let public = private
            .public_key()
            .expect("ed25519 private key yields a public key");
        Self { private, public }
    }

    /// Derive the key deterministically from a 32-byte seed (so an operator can reproduce the same
    /// signing key across machines / a re-sign).
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let private = PrivateKeyBase::from_data(*seed).ed25519_signing_private_key();
        let public = private
            .public_key()
            .expect("ed25519 private key yields a public key");
        Self { private, public }
    }

    /// The verifying half — attach to a [`SkillStore`](crate::SkillStore) to enforce the gate.
    pub fn verifier(&self) -> SkillBundleVerifier {
        SkillBundleVerifier(self.public.clone())
    }

    /// Sign `bundle` (its own `signature` field is ignored), returning the hex signature to store in
    /// [`SkillBundle::signature`].
    pub fn sign(&self, bundle: &SkillBundle) -> String {
        let digest = bundle_digest(bundle);
        let msg: &[u8] = &digest;
        let sig: Signature = self
            .private
            .sign(&msg)
            .expect("ed25519 signing is infallible for valid keys");
        to_hex(&sig.to_cbor_data())
    }
}

/// The trusted public half configured on a [`SkillStore`](crate::SkillStore): when present, an import
/// must carry a signature that verifies against it, else it is refused.
#[derive(Clone)]
pub struct SkillBundleVerifier(SigningPublicKey);

impl SkillBundleVerifier {
    /// Reconstruct a verifier from the hex-encoded dCBOR of a [`SkillBundleSigner`]'s public key (the
    /// operator-configured trust anchor). `None` if the hex/CBOR/key is malformed.
    pub fn from_public_hex(hex: &str) -> Option<Self> {
        let bytes = from_hex(hex)?;
        let cbor = CBOR::try_from_data(&bytes).ok()?;
        let key = SigningPublicKey::try_from(cbor).ok()?;
        Some(Self(key))
    }

    /// The trust-anchor public key as hex-encoded dCBOR (for display / config round-trip).
    pub fn to_public_hex(&self) -> String {
        to_hex(&self.0.to_cbor_data())
    }

    /// Verify a bundle end-to-end: require a signature, decode it, and check it over the recomputed
    /// digest. Returns the precise [`SkillError::Signature`] on any failure (fail-closed).
    pub fn verify(&self, bundle: &SkillBundle) -> Result<(), SkillError> {
        let Some(sig_hex) = bundle.signature.as_deref() else {
            return Err(SkillError::Signature(format!(
                "bundle `{}` is unsigned but a trusted signing key is configured",
                bundle.name
            )));
        };
        let Some(sig_bytes) = from_hex(sig_hex) else {
            return Err(SkillError::Signature(format!(
                "bundle `{}` signature is not valid hex",
                bundle.name
            )));
        };
        let Ok(cbor) = CBOR::try_from_data(&sig_bytes) else {
            return Err(SkillError::Signature(format!(
                "bundle `{}` signature is not valid CBOR",
                bundle.name
            )));
        };
        let Ok(sig) = Signature::try_from(cbor) else {
            return Err(SkillError::Signature(format!(
                "bundle `{}` signature is not an ed25519 signature",
                bundle.name
            )));
        };
        let digest = bundle_digest(bundle);
        let msg: &[u8] = &digest;
        if !self.0.verify(&sig, &msg) {
            return Err(SkillError::Signature(format!(
                "bundle `{}` signature does not verify against the configured key",
                bundle.name
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn bundle(name: &str, body: &str) -> SkillBundle {
        let mut files = BTreeMap::new();
        files.insert("SKILL.md".to_string(), body.to_string());
        SkillBundle {
            name: name.to_string(),
            category: Some("cat".to_string()),
            files,
            signature: None,
        }
    }

    #[test]
    fn valid_signature_verifies() {
        let signer = SkillBundleSigner::generate();
        let mut b = bundle("demo", "hello");
        b.signature = Some(signer.sign(&b));
        signer
            .verifier()
            .verify(&b)
            .expect("valid signature verifies");
    }

    #[test]
    fn tampered_content_is_refused() {
        let signer = SkillBundleSigner::generate();
        let mut b = bundle("demo", "hello");
        b.signature = Some(signer.sign(&b));
        // Mutate a file after signing: the recomputed digest no longer matches.
        b.files.insert("SKILL.md".into(), "tampered".into());
        let err = signer.verifier().verify(&b).expect_err("must refuse");
        assert!(matches!(err, SkillError::Signature(_)));
    }

    #[test]
    fn absent_signature_is_refused() {
        let signer = SkillBundleSigner::generate();
        let b = bundle("demo", "hello"); // signature: None
        let err = signer
            .verifier()
            .verify(&b)
            .expect_err("must refuse unsigned");
        assert!(matches!(err, SkillError::Signature(_)));
    }

    #[test]
    fn wrong_key_is_refused() {
        let signer = SkillBundleSigner::generate();
        let attacker = SkillBundleSigner::generate();
        let mut b = bundle("demo", "hello");
        b.signature = Some(signer.sign(&b));
        // A different trust anchor rejects a signature it did not produce.
        let err = attacker
            .verifier()
            .verify(&b)
            .expect_err("must refuse foreign sig");
        assert!(matches!(err, SkillError::Signature(_)));
    }

    #[test]
    fn public_key_hex_round_trips() {
        let signer = SkillBundleSigner::generate();
        let v = signer.verifier();
        let hex = v.to_public_hex();
        let restored = SkillBundleVerifier::from_public_hex(&hex).expect("valid hex key");
        let mut b = bundle("demo", "hello");
        b.signature = Some(signer.sign(&b));
        restored
            .verify(&b)
            .expect("restored-from-hex key verifies a genuine signature");
    }
}
