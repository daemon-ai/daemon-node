// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! ed25519 signing over canonical CBOR.
//!
//! Every swarm control message and the run envelope are signed by the node identity (spec §7.2,
//! §7.3). Signatures are always taken over the [`crate::canonical`] encoding of the value, so a
//! signature is a commitment to a *value*, independent of any transport re-encoding. ed25519
//! signing is deterministic (RFC 8032), which is what makes envelope freeze idempotent (PROTO-11)
//! and keeps the crate free of any RNG dependency — it stays `wasm32`-clean.

use serde::{Deserialize, Serialize};

use ed25519_dalek::Signer;

use crate::bytes::{PeerId, Signature};
use crate::canonical::to_canonical_vec;
use crate::error::SwarmProtoError;

pub use ed25519_dalek::{SigningKey, VerifyingKey};

/// The 32-byte [`PeerId`] (ed25519 public key) for a signing key.
#[must_use]
pub fn peer_id(key: &SigningKey) -> PeerId {
    PeerId(key.verifying_key().to_bytes())
}

/// Sign the canonical CBOR encoding of `value` with `key`.
pub fn sign_canonical<T: Serialize + ?Sized>(
    key: &SigningKey,
    value: &T,
) -> Result<Signature, SwarmProtoError> {
    let bytes = to_canonical_vec(value)?;
    Ok(Signature(key.sign(&bytes).to_bytes()))
}

/// Verify `sig` over the canonical CBOR encoding of `value`, against `signer`.
pub fn verify_canonical<T: Serialize + ?Sized>(
    signer: &PeerId,
    sig: &Signature,
    value: &T,
) -> Result<(), SwarmProtoError> {
    let bytes = to_canonical_vec(value)?;
    verify_bytes(signer, sig, &bytes)
}

/// Verify `sig` over raw `bytes`, against `signer`. Uses `verify_strict` (rejects the small-order /
/// malleability edge cases), matching the consensus intent that a signature be unambiguous.
pub fn verify_bytes(signer: &PeerId, sig: &Signature, bytes: &[u8]) -> Result<(), SwarmProtoError> {
    let vk = VerifyingKey::from_bytes(&signer.0)
        .map_err(|e| SwarmProtoError::Signature(format!("malformed public key: {e}")))?;
    let dsig = ed25519_dalek::Signature::from_bytes(&sig.0);
    vk.verify_strict(bytes, &dsig)
        .map_err(|_| SwarmProtoError::Signature("signature does not verify".into()))
}

/// A value bundled with the node identity that signed its canonical CBOR encoding.
///
/// The generic wrapper used for envelope/message signing; concrete wire frames (e.g.
/// [`crate::messages::SignedMessage`]) build on the same [`sign_canonical`]/[`verify_canonical`]
/// pair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signed<T> {
    /// The signed value.
    pub body: T,
    /// The node identity (ed25519 public key) that produced [`Signed::sig`].
    pub signer: PeerId,
    /// ed25519 signature over the canonical CBOR encoding of [`Signed::body`].
    pub sig: Signature,
}

impl<T: Serialize> Signed<T> {
    /// Seal `body` under `key`.
    pub fn seal(key: &SigningKey, body: T) -> Result<Self, SwarmProtoError> {
        let sig = sign_canonical(key, &body)?;
        Ok(Self {
            body,
            signer: peer_id(key),
            sig,
        })
    }

    /// Verify the signature over the body.
    pub fn verify(&self) -> Result<(), SwarmProtoError> {
        verify_canonical(&self.signer, &self.sig, &self.body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    #[test]
    fn sign_is_deterministic() {
        let k = key(7);
        let a = sign_canonical(&k, &"round-42".to_string()).unwrap();
        let b = sign_canonical(&k, &"round-42".to_string()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn seal_verifies_and_rejects_tamper() {
        let k = key(9);
        let mut sealed = Signed::seal(&k, vec![1u32, 2, 3]).unwrap();
        assert!(sealed.verify().is_ok());

        // Tampered body no longer matches the signature.
        sealed.body.push(4);
        assert!(sealed.verify().is_err());
    }

    #[test]
    fn wrong_signer_rejected() {
        let k = key(1);
        let sealed = Signed::seal(&k, "hello".to_string()).unwrap();
        let impostor = peer_id(&key(2));
        assert!(verify_canonical(&impostor, &sealed.sig, &sealed.body).is_err());
    }

    #[test]
    fn corrupted_signature_rejected() {
        let k = key(3);
        let mut sealed = Signed::seal(&k, 12345u64).unwrap();
        sealed.sig.0[0] ^= 0xff;
        assert!(sealed.verify().is_err());
    }
}
