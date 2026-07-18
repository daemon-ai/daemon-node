// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Certified per-run keys — the D1 signing-oracle evolution **around** the A2-frozen frame
//! envelope (architecture §4.3; refactor §8/D1; ABI §12.1).
//!
//! At admission the host generates a fresh software keypair for the role-instance and signs a
//! **certificate** with the base machine identity, binding `(genesis hash, role-instance, epoch
//! validity, per-run public key)`. All run traffic is then signed with the per-run key — that key
//! is exactly the `sender` field of the A2-frozen §12.1 frame envelope — and peers authenticate it
//! by verifying this certificate chain back to the base identity.
//!
//! **This is additive and lands strictly around the frozen envelope (ABI §12.1):** the certificate
//! is a *separate distribution record*, never a `frame-envelope` field. D1 MUST NOT add, remove, or
//! change any frame-envelope field (the fields that give a Phase-A sequence its evidentiary meaning
//! are frozen at A2). The **old verifier is retained**: a receiver still verifies the frame
//! signature over `sender` exactly as A2 did ([`crate::sign::verify_bytes`] /
//! `daemon-vhc-session::v2_attach`); the certificate check is an *additional* layer that
//! authenticates the `sender` per-run key to the base identity. A receiver that holds no cert store
//! keeps the A2 behavior unchanged (the transition path).
//!
//! Why this shape (architecture §4.3): it scopes every signature to one run — meaningless in any
//! other run, role, daemon service, or protocol version (the domain tag + `run_id` bind it);
//! it works with hardware-backed, non-exportable base identity keys (the base key is touched
//! **once per run** to issue the cert, not per frame); and it yields rotation and expiry for free
//! via the certificate's epoch validity window.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Signature};
use crate::error::VhcProtoError;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};

/// The domain-separation tag every run-key certificate body carries at ABI major 2. Distinct from
/// the frame-envelope domain (`daemon-vhc/frame/2`, ABI §12.1) so a certificate signature can never
/// be replayed as a frame signature or vice versa.
pub const CERT_DOMAIN_V2: &str = "daemon-vhc/cert/2";

/// The signed body of a run-key certificate: the binding the base identity attests to. Every field
/// is part of the signed preimage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunKeyCertBody {
    /// Domain-separation tag — MUST be [`CERT_DOMAIN_V2`].
    pub domain: String,
    /// The run's cryptographic identity: the genesis-envelope hash (ABI §8.1 `run_id`).
    pub run_id: Hash,
    /// The envelope-level role label this per-run key acts as.
    pub role: String,
    /// The never-reused monotonic role-instance incarnation id (ABI §8.1 `instance`).
    pub instance: u64,
    /// First epoch (inclusive) the certificate is valid for.
    pub epoch_from: u64,
    /// Last epoch (inclusive) the certificate is valid for — expiry for free (architecture §4.3).
    pub epoch_to: u64,
    /// The certified **per-run public key** — this is exactly the §12.1 frame envelope `sender`.
    pub run_key: PeerId,
}

/// A run-key certificate: the [`RunKeyCertBody`] binding, plus the base machine identity that
/// signed it and its ed25519 signature over the canonical CBOR of the body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunKeyCertificate {
    /// The signed binding.
    pub body: RunKeyCertBody,
    /// The base machine identity (the certificate issuer / root of the per-run chain).
    pub base_identity: PeerId,
    /// ed25519 signature by `base_identity` over the canonical CBOR of `body`.
    pub sig: Signature,
}

/// Why a certificate (or a certified-sender check) was rejected. Distinguished so the
/// signature-downgrade matrix can assert *which* guard fired (a certified-sender acceptance vs a
/// scope/expiry/chain refusal).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CertError {
    /// The body's domain tag is not [`CERT_DOMAIN_V2`] (wrong protocol/era certificate).
    WrongDomain {
        /// The tag actually carried.
        got: String,
    },
    /// The base identity's signature over the body does not verify (a forged / tampered cert).
    BadChain,
    /// The certificate's `run_id`/`role`/`instance` scope does not match the frame's scope.
    ScopeMismatch,
    /// The frame's epoch is outside the certificate's `[epoch_from, epoch_to]` validity window.
    Expired {
        /// The frame epoch checked.
        epoch: u64,
        /// The certificate's inclusive lower bound.
        from: u64,
        /// The certificate's inclusive upper bound.
        to: u64,
    },
    /// The certified `run_key` is not the frame's `sender` — a per-run key with no matching cert
    /// (the downgrade case: an uncertified signer presenting under a cert that is not its own).
    SenderNotCertified,
    /// No certificate in the store authenticates the frame's `sender` for its scope+epoch.
    NoCertifiedChain,
}

impl core::fmt::Display for CertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongDomain { got } => {
                write!(f, "certificate domain `{got}` is not `{CERT_DOMAIN_V2}`")
            }
            Self::BadChain => write!(
                f,
                "certificate signature does not verify to the base identity"
            ),
            Self::ScopeMismatch => write!(f, "certificate scope does not match the frame scope"),
            Self::Expired { epoch, from, to } => {
                write!(
                    f,
                    "frame epoch {epoch} outside certificate validity [{from}, {to}]"
                )
            }
            Self::SenderNotCertified => {
                write!(f, "frame sender is not the certificate's certified run key")
            }
            Self::NoCertifiedChain => {
                write!(
                    f,
                    "no certificate authenticates the frame sender for its scope/epoch"
                )
            }
        }
    }
}

impl std::error::Error for CertError {}

impl From<CertError> for VhcProtoError {
    fn from(e: CertError) -> Self {
        VhcProtoError::Signature(e.to_string())
    }
}

impl RunKeyCertificate {
    /// Issue a certificate: the base machine identity `base_key` attests that `run_key` is the
    /// per-run key for `(run_id, role, instance)` valid over epochs `[epoch_from, epoch_to]`. The
    /// base key is touched exactly here (once per run), never per frame (architecture §4.3).
    ///
    /// # Errors
    /// A signing failure (canonical-CBOR encode / ed25519), surfaced as [`VhcProtoError`].
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        base_key: &SigningKey,
        run_id: Hash,
        role: impl Into<String>,
        instance: u64,
        epoch_from: u64,
        epoch_to: u64,
        run_key: PeerId,
    ) -> Result<Self, VhcProtoError> {
        let body = RunKeyCertBody {
            domain: CERT_DOMAIN_V2.to_string(),
            run_id,
            role: role.into(),
            instance,
            epoch_from,
            epoch_to,
            run_key,
        };
        let sig = sign_canonical(base_key, &body)?;
        Ok(Self {
            body,
            base_identity: peer_id(base_key),
            sig,
        })
    }

    /// Verify the certificate **chain only**: the domain tag is [`CERT_DOMAIN_V2`] and the base
    /// identity's signature over the body verifies (`verify_strict`). This does not check scope or
    /// expiry — see [`RunKeyCertificate::authorizes_sender`].
    ///
    /// # Errors
    /// [`CertError::WrongDomain`] or [`CertError::BadChain`].
    pub fn verify_chain(&self) -> Result<(), CertError> {
        if self.body.domain != CERT_DOMAIN_V2 {
            return Err(CertError::WrongDomain {
                got: self.body.domain.clone(),
            });
        }
        verify_canonical(&self.base_identity, &self.sig, &self.body)
            .map_err(|_| CertError::BadChain)
    }

    /// Whether this certificate's scope matches `(run_id, role, instance)` and its validity window
    /// includes `epoch`. Chain-independent (call [`RunKeyCertificate::verify_chain`] too).
    #[must_use]
    pub fn covers(&self, run_id: &Hash, role: &str, instance: u64, epoch: u64) -> bool {
        self.body.run_id == *run_id
            && self.body.role == role
            && self.body.instance == instance
            && epoch >= self.body.epoch_from
            && epoch <= self.body.epoch_to
    }

    /// The full certified-sender check for one frame: the chain verifies, the scope matches, the
    /// epoch is in-window, and the certified `run_key` **is** the frame's `sender`. This is what
    /// authenticates a "v2 signer" (a certified per-run key) — and what refuses a downgraded /
    /// uncertified sender.
    ///
    /// # Errors
    /// The applicable [`CertError`] (wrong domain, bad chain, scope mismatch, expired, or
    /// sender-not-certified).
    pub fn authorizes_sender(
        &self,
        run_id: &Hash,
        role: &str,
        instance: u64,
        epoch: u64,
        sender: &PeerId,
    ) -> Result<(), CertError> {
        self.verify_chain()?;
        if self.body.run_id != *run_id || self.body.role != role || self.body.instance != instance {
            return Err(CertError::ScopeMismatch);
        }
        if epoch < self.body.epoch_from || epoch > self.body.epoch_to {
            return Err(CertError::Expired {
                epoch,
                from: self.body.epoch_from,
                to: self.body.epoch_to,
            });
        }
        if self.body.run_key != *sender {
            return Err(CertError::SenderNotCertified);
        }
        Ok(())
    }
}

/// Authenticate a frame's `sender` per-run key against a set of certificates, trusting only
/// certificates chained to `trusted_base` (the base machine identity a peer is willing to accept a
/// per-run key from — e.g. the coordinator's or a rostered worker's base identity, distributed via
/// the genesis / a join record). Returns `Ok` on the first certificate that
/// [`RunKeyCertificate::authorizes_sender`] accepts.
///
/// This is the D1 cert-aware acceptance that layers **around** the A2 frame-signature check (the
/// old verifier): the caller must have already verified the frame signature over `sender`; this
/// establishes that `sender` is a legitimately certified per-run key for the frame's scope/epoch.
///
/// # Errors
/// [`CertError::NoCertifiedChain`] if no trusted certificate authenticates the sender.
pub fn verify_certified_sender(
    run_id: &Hash,
    role: &str,
    instance: u64,
    epoch: u64,
    sender: &PeerId,
    trusted_base: &PeerId,
    certs: &[RunKeyCertificate],
) -> Result<(), CertError> {
    for cert in certs {
        if cert.base_identity != *trusted_base {
            continue;
        }
        if cert
            .authorizes_sender(run_id, role, instance, epoch, sender)
            .is_ok()
        {
            return Ok(());
        }
    }
    Err(CertError::NoCertifiedChain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sign::SigningKey;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn run_id(n: u8) -> Hash {
        Hash([n; 32])
    }

    fn issue(
        base: &SigningKey,
        run: Hash,
        role: &str,
        inst: u64,
        from: u64,
        to: u64,
        run_key: PeerId,
    ) -> RunKeyCertificate {
        RunKeyCertificate::issue(base, run, role, inst, from, to, run_key).unwrap()
    }

    #[test]
    fn a_certified_per_run_key_authenticates_to_its_base_identity() {
        let base = key(1);
        let run_key = peer_id(&key(2));
        let cert = issue(&base, run_id(9), "trainer", 7, 0, 5, run_key);
        assert!(cert.verify_chain().is_ok());
        // The certified sender at an in-window epoch is authenticated.
        assert!(cert
            .authorizes_sender(&run_id(9), "trainer", 7, 3, &run_key)
            .is_ok());
    }

    #[test]
    fn a_tampered_certificate_breaks_the_chain() {
        let base = key(1);
        let run_key = peer_id(&key(2));
        let mut cert = issue(&base, run_id(9), "trainer", 7, 0, 5, run_key);
        cert.body.instance = 8; // re-scope without re-signing
        assert_eq!(cert.verify_chain(), Err(CertError::BadChain));
    }

    #[test]
    fn wrong_domain_is_rejected() {
        let base = key(1);
        let run_key = peer_id(&key(2));
        let mut cert = issue(&base, run_id(9), "trainer", 7, 0, 5, run_key);
        cert.body.domain = "daemon-vhc/frame/2".into(); // a frame tag, not a cert tag
        assert!(matches!(
            cert.verify_chain(),
            Err(CertError::WrongDomain { .. })
        ));
    }

    // -- the signature-downgrade matrix (tier-1) ---------------------------------------------------
    //
    // "a v2 signer is refused by no cell that should accept it, and vice versa": a certified per-run
    // key is accepted for exactly its (run, role, instance, epoch-window) and refused everywhere
    // else; an uncertified / downgraded sender is refused.

    #[test]
    fn downgrade_uncertified_sender_is_refused_certified_sender_is_accepted() {
        let coord_base = key(10);
        let coord_run_key = peer_id(&key(11));
        let cert = issue(
            &coord_base,
            run_id(1),
            "coordinator",
            1,
            0,
            10,
            coord_run_key,
        );
        let store = [cert];
        let base_id = peer_id(&coord_base);

        // ACCEPT: the certified per-run key, in-scope and in-window.
        assert!(verify_certified_sender(
            &run_id(1),
            "coordinator",
            1,
            4,
            &coord_run_key,
            &base_id,
            &store
        )
        .is_ok());

        // REFUSE (downgrade): a different, uncertified key presenting on the same scope.
        let impostor = peer_id(&key(99));
        assert_eq!(
            verify_certified_sender(&run_id(1), "coordinator", 1, 4, &impostor, &base_id, &store),
            Err(CertError::NoCertifiedChain)
        );
    }

    #[test]
    fn certified_sender_is_refused_out_of_scope_and_out_of_window() {
        let base = key(10);
        let run_key = peer_id(&key(11));
        let cert = issue(&base, run_id(1), "coordinator", 1, 2, 6, run_key);
        let base_id = peer_id(&base);
        let store = [cert.clone()];

        // Wrong run.
        assert_eq!(
            verify_certified_sender(&run_id(2), "coordinator", 1, 4, &run_key, &base_id, &store),
            Err(CertError::NoCertifiedChain)
        );
        // Wrong role.
        assert_eq!(
            verify_certified_sender(&run_id(1), "trainer", 1, 4, &run_key, &base_id, &store),
            Err(CertError::NoCertifiedChain)
        );
        // Wrong incarnation.
        assert_eq!(
            verify_certified_sender(&run_id(1), "coordinator", 2, 4, &run_key, &base_id, &store),
            Err(CertError::NoCertifiedChain)
        );
        // Below and above the validity window.
        assert_eq!(
            cert.authorizes_sender(&run_id(1), "coordinator", 1, 1, &run_key),
            Err(CertError::Expired {
                epoch: 1,
                from: 2,
                to: 6
            })
        );
        assert_eq!(
            cert.authorizes_sender(&run_id(1), "coordinator", 1, 7, &run_key),
            Err(CertError::Expired {
                epoch: 7,
                from: 2,
                to: 6
            })
        );
    }

    #[test]
    fn a_cert_from_an_untrusted_base_is_not_accepted() {
        // A validly self-consistent cert, but signed by a base identity the receiver does not trust
        // (e.g. an attacker's own base key): never accepted, because the chain must terminate at the
        // *expected* base identity, not merely at *some* base identity.
        let attacker_base = key(50);
        let attacker_run_key = peer_id(&key(51));
        let cert = issue(
            &attacker_base,
            run_id(1),
            "coordinator",
            1,
            0,
            10,
            attacker_run_key,
        );
        let store = [cert];
        let trusted_base = peer_id(&key(10)); // the honest coordinator base, NOT the attacker's
        assert_eq!(
            verify_certified_sender(
                &run_id(1),
                "coordinator",
                1,
                4,
                &attacker_run_key,
                &trusted_base,
                &store
            ),
            Err(CertError::NoCertifiedChain)
        );
    }

    #[test]
    fn certificate_round_trips_through_canonical_cbor() {
        let base = key(1);
        let run_key = peer_id(&key(2));
        let cert = issue(&base, run_id(9), "trainer", 7, 0, 5, run_key);
        let bytes = crate::canonical::to_canonical_vec(&cert).unwrap();
        let back: RunKeyCertificate = crate::canonical::from_canonical_slice(&bytes).unwrap();
        assert_eq!(cert, back);
        assert!(back.verify_chain().is_ok());
    }
}
