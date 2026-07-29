// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Certified per-run keys — the signing-oracle evolution **around** the frozen frame
//! envelope (architecture §4.3; ABI §12.1).
//!
//! At admission the host generates a fresh CSPRNG software keypair for the role-instance and
//! signs a **certificate** with the base machine identity, binding the per-run public key to the
//! full execution identity `(run_id, epoch, role, incarnation, module_hash)` (ABI §8.1). All run
//! traffic is then signed with the per-run key — that key is exactly the `sender` field of the
//! frozen §12.1 frame envelope — and peers authenticate it by verifying this certificate chain
//! back to a base identity the run's genesis/Authority configuration names.
//!
//! **This is additive and lands strictly around the frozen envelope (ABI §12.1):** the certificate
//! is a *separate distribution record*, never a `frame-envelope` field. The certificate layer MUST
//! NOT add, remove, or change any frame-envelope field. The frame verifier is retained: a receiver
//! still verifies the frame signature over `sender` exactly as before
//! ([`crate::sign::verify_bytes`] / `daemon-vhc-session::attach`); the certificate check is an
//! *additional* layer that authenticates the `sender` per-run key to the base identity.
//!
//! **Epoch binding and rotation.** A certificate binds ONE epoch. An epoch change (a committed
//! module upgrade) REBINDS the same per-run key by issuing a new certificate for the new epoch —
//! journal identity stays stable across the fence. Full key rotation happens only on incarnation
//! change: a new incarnation generates a fresh key and receives a fresh certificate, and
//! incarnation monotonicity supersedes every prior incarnation for the role slot.
//!
//! Why this shape (architecture §4.3): it scopes every signature to one run — meaningless in any
//! other run, role, epoch, incarnation, module, daemon service, or protocol version (the domain
//! tag + the scope fields bind it); it works with hardware-backed, non-exportable base identity
//! keys (the base key is touched **once per binding** to issue the cert, not per frame); and
//! expiry is structural — a certificate dies with its epoch/incarnation, never by wall clock.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Signature};
use crate::error::VhcProtoError;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};

/// The domain-separation tag every run-key certificate body carries at ABI major 2. Distinct from
/// the frame-envelope domain (`daemon-vhc/frame/2`, ABI §12.1) so a certificate signature can never
/// be replayed as a frame signature or vice versa.
pub const CERT_DOMAIN_V2: &str = "daemon-vhc/cert/2";

/// The scope a run-key certificate binds a per-run key to: the full ABI §8.1 execution identity.
/// Every field is part of the signed preimage; verification compares all of them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertScope {
    /// The run's cryptographic identity: the genesis-envelope hash (ABI §8.1 `run_id`).
    pub run_id: Hash,
    /// The transition-chain epoch this binding is valid for. An epoch change reissues the
    /// certificate (same key, new binding); a certificate never spans epochs.
    pub epoch: u64,
    /// The envelope-level role label this per-run key acts as.
    pub role: String,
    /// The never-reused monotonic role-instance incarnation id (ABI §8.1 `instance`).
    pub instance: u64,
    /// The pinned module blob the role-instance runs at this epoch (ABI §8.1 `module_hash`).
    pub module_hash: Hash,
}

/// The signed body of a run-key certificate: the binding the base identity attests to. Every field
/// is part of the signed preimage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunKeyCertBody {
    /// Domain-separation tag — MUST be [`CERT_DOMAIN_V2`].
    pub domain: String,
    /// The full execution-identity scope this certificate binds.
    pub scope: CertScope,
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
/// scope/epoch/module/chain refusal).
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
    /// The frame's epoch is not the single epoch the certificate binds. An epoch change reissues
    /// the certificate; a stale-epoch certificate never authenticates the new epoch's frames.
    EpochMismatch {
        /// The frame epoch checked.
        epoch: u64,
        /// The one epoch the certificate binds.
        bound: u64,
    },
    /// The frame's `module` is not the module hash the certificate binds — a per-run key certified
    /// for one module blob never authenticates traffic attributed to another.
    ModuleMismatch,
    /// The certified `run_key` is not the frame's `sender` — a per-run key with no matching cert
    /// (the downgrade case: an uncertified signer presenting under a cert that is not its own).
    SenderNotCertified,
    /// No certificate in the store authenticates the frame's `sender` for its scope+epoch.
    NoCertifiedChain,
    /// The per-run key is dead: explicitly revoked by a signed record, or its incarnation is
    /// superseded by a higher one for the same role slot ([`crate::revocation`]).
    Revoked,
    /// The sender presents under a seat-governed role but is not the claimant bound at the
    /// highest VERIFIED leadership term the receiver holds ([`crate::seat::SeatTermLedger`]) —
    /// a fenced (superseded) coordinator, dead regardless of its certificate ([SEAT-3] v2).
    SeatSuperseded,
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
            Self::EpochMismatch { epoch, bound } => {
                write!(
                    f,
                    "frame epoch {epoch} is not the certificate's bound epoch {bound} \
                     (an epoch change reissues the certificate)"
                )
            }
            Self::ModuleMismatch => {
                write!(f, "certificate module hash does not match the frame module")
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
            Self::Revoked => {
                write!(
                    f,
                    "the per-run key is revoked or its incarnation is superseded"
                )
            }
            Self::SeatSuperseded => {
                write!(
                    f,
                    "the sender is not the seat claimant bound at the highest verified \
                     leadership term (a fenced coordinator)"
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
    /// per-run key for the execution-identity `scope`. The base key is touched exactly here (once
    /// per binding; an epoch change re-issues with the same `run_key`), never per frame
    /// (architecture §4.3).
    ///
    /// # Errors
    /// A signing failure (canonical-CBOR encode / ed25519), surfaced as [`VhcProtoError`].
    pub fn issue(
        base_key: &SigningKey,
        scope: CertScope,
        run_key: PeerId,
    ) -> Result<Self, VhcProtoError> {
        let body = RunKeyCertBody {
            domain: CERT_DOMAIN_V2.to_string(),
            scope,
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
    /// identity's signature over the body verifies (`verify_strict`). This does not check scope —
    /// see [`RunKeyCertificate::authorizes_sender`].
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

    /// Whether this certificate's binding matches `scope` exactly (all five execution-identity
    /// fields). Chain-independent (call [`RunKeyCertificate::verify_chain`] too).
    #[must_use]
    pub fn covers(&self, scope: &CertScope) -> bool {
        self.body.scope == *scope
    }

    /// The full certified-sender check for one frame: the chain verifies, the scope matches field
    /// by field (run, role, incarnation, then the bound epoch, then the bound module), and the
    /// certified `run_key` **is** the frame's `sender`. This is what authenticates a certified
    /// per-run key — and what refuses a downgraded / uncertified sender.
    ///
    /// # Errors
    /// The applicable [`CertError`] (wrong domain, bad chain, scope mismatch, epoch mismatch,
    /// module mismatch, or sender-not-certified).
    pub fn authorizes_sender(&self, scope: &CertScope, sender: &PeerId) -> Result<(), CertError> {
        self.verify_chain()?;
        let bound = &self.body.scope;
        if bound.run_id != scope.run_id
            || bound.role != scope.role
            || bound.instance != scope.instance
        {
            return Err(CertError::ScopeMismatch);
        }
        if bound.epoch != scope.epoch {
            return Err(CertError::EpochMismatch {
                epoch: scope.epoch,
                bound: bound.epoch,
            });
        }
        if bound.module_hash != scope.module_hash {
            return Err(CertError::ModuleMismatch);
        }
        if self.body.run_key != *sender {
            return Err(CertError::SenderNotCertified);
        }
        Ok(())
    }
}

/// Authenticate a frame's `sender` per-run key against a set of certificates, trusting only
/// certificates chained to `trusted_base` (the base machine identity a peer is willing to accept a
/// per-run key from — named by the run's genesis/Authority configuration, never ambient config).
/// Returns `Ok` on the first certificate that [`RunKeyCertificate::authorizes_sender`] accepts.
///
/// This is the cert-aware acceptance that layers **around** the frame-signature check (the
/// retained verifier): the caller must have already verified the frame signature over `sender`;
/// this establishes that `sender` is a legitimately certified per-run key for the frame's scope.
///
/// # Errors
/// [`CertError::NoCertifiedChain`] if no trusted certificate authenticates the sender.
pub fn verify_certified_sender(
    scope: &CertScope,
    sender: &PeerId,
    trusted_base: &PeerId,
    certs: &[RunKeyCertificate],
) -> Result<(), CertError> {
    for cert in certs {
        if cert.base_identity != *trusted_base {
            continue;
        }
        if cert.authorizes_sender(scope, sender).is_ok() {
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

    fn module(n: u8) -> Hash {
        Hash([n; 32])
    }

    fn scope(run: Hash, role: &str, inst: u64, epoch: u64, module_hash: Hash) -> CertScope {
        CertScope {
            run_id: run,
            epoch,
            role: role.to_string(),
            instance: inst,
            module_hash,
        }
    }

    fn issue(base: &SigningKey, s: CertScope, run_key: PeerId) -> RunKeyCertificate {
        RunKeyCertificate::issue(base, s, run_key).unwrap()
    }

    #[test]
    fn a_certified_per_run_key_authenticates_to_its_base_identity() {
        let base = key(1);
        let run_key = peer_id(&key(2));
        let s = scope(run_id(9), "trainer", 7, 3, module(0xAA));
        let cert = issue(&base, s.clone(), run_key);
        assert!(cert.verify_chain().is_ok());
        // The certified sender at the bound scope is authenticated.
        assert!(cert.authorizes_sender(&s, &run_key).is_ok());
    }

    #[test]
    fn a_tampered_certificate_breaks_the_chain() {
        let base = key(1);
        let run_key = peer_id(&key(2));
        let mut cert = issue(
            &base,
            scope(run_id(9), "trainer", 7, 0, module(0xAA)),
            run_key,
        );
        cert.body.scope.instance = 8; // re-scope without re-signing
        assert_eq!(cert.verify_chain(), Err(CertError::BadChain));
    }

    #[test]
    fn wrong_domain_is_rejected() {
        let base = key(1);
        let run_key = peer_id(&key(2));
        let mut cert = issue(
            &base,
            scope(run_id(9), "trainer", 7, 0, module(0xAA)),
            run_key,
        );
        cert.body.domain = "daemon-vhc/frame/2".into(); // a frame tag, not a cert tag
        assert!(matches!(
            cert.verify_chain(),
            Err(CertError::WrongDomain { .. })
        ));
    }

    // -- the signature-downgrade matrix (tier-1) ---------------------------------------------------
    //
    // "a certified signer is refused by no cell that should accept it, and vice versa": a certified
    // per-run key is accepted for exactly its (run, epoch, role, incarnation, module) binding and
    // refused everywhere else; an uncertified / downgraded sender is refused.

    #[test]
    fn downgrade_uncertified_sender_is_refused_certified_sender_is_accepted() {
        let coord_base = key(10);
        let coord_run_key = peer_id(&key(11));
        let s = scope(run_id(1), "coordinator", 1, 4, module(0xCC));
        let cert = issue(&coord_base, s.clone(), coord_run_key);
        let store = [cert];
        let base_id = peer_id(&coord_base);

        // ACCEPT: the certified per-run key, at exactly its bound scope.
        assert!(verify_certified_sender(&s, &coord_run_key, &base_id, &store).is_ok());

        // REFUSE (downgrade): a different, uncertified key presenting on the same scope.
        let impostor = peer_id(&key(99));
        assert_eq!(
            verify_certified_sender(&s, &impostor, &base_id, &store),
            Err(CertError::NoCertifiedChain)
        );
    }

    #[test]
    fn certified_sender_is_refused_out_of_scope_epoch_and_module() {
        let base = key(10);
        let run_key = peer_id(&key(11));
        let bound = scope(run_id(1), "coordinator", 1, 4, module(0xCC));
        let cert = issue(&base, bound.clone(), run_key);
        let base_id = peer_id(&base);
        let store = [cert.clone()];

        // Wrong run.
        assert_eq!(
            verify_certified_sender(
                &scope(run_id(2), "coordinator", 1, 4, module(0xCC)),
                &run_key,
                &base_id,
                &store
            ),
            Err(CertError::NoCertifiedChain)
        );
        // Wrong role.
        assert_eq!(
            verify_certified_sender(
                &scope(run_id(1), "trainer", 1, 4, module(0xCC)),
                &run_key,
                &base_id,
                &store
            ),
            Err(CertError::NoCertifiedChain)
        );
        // Wrong incarnation.
        assert_eq!(
            verify_certified_sender(
                &scope(run_id(1), "coordinator", 2, 4, module(0xCC)),
                &run_key,
                &base_id,
                &store
            ),
            Err(CertError::NoCertifiedChain)
        );
        // A stale (or future) epoch: the binding is per-epoch; a change reissues the cert.
        assert_eq!(
            cert.authorizes_sender(
                &scope(run_id(1), "coordinator", 1, 5, module(0xCC)),
                &run_key
            ),
            Err(CertError::EpochMismatch { epoch: 5, bound: 4 })
        );
        assert_eq!(
            cert.authorizes_sender(
                &scope(run_id(1), "coordinator", 1, 3, module(0xCC)),
                &run_key
            ),
            Err(CertError::EpochMismatch { epoch: 3, bound: 4 })
        );
        // A different module blob: the key is certified for one pinned module only.
        assert_eq!(
            cert.authorizes_sender(
                &scope(run_id(1), "coordinator", 1, 4, module(0xDD)),
                &run_key
            ),
            Err(CertError::ModuleMismatch)
        );
    }

    #[test]
    fn a_cert_from_an_untrusted_base_is_not_accepted() {
        // A validly self-consistent cert, but signed by a base identity the receiver does not trust
        // (e.g. an attacker's own base key): never accepted, because the chain must terminate at the
        // *expected* base identity, not merely at *some* base identity.
        let attacker_base = key(50);
        let attacker_run_key = peer_id(&key(51));
        let s = scope(run_id(1), "coordinator", 1, 4, module(0xCC));
        let cert = issue(&attacker_base, s.clone(), attacker_run_key);
        let store = [cert];
        let trusted_base = peer_id(&key(10)); // the honest coordinator base, NOT the attacker's
        assert_eq!(
            verify_certified_sender(&s, &attacker_run_key, &trusted_base, &store),
            Err(CertError::NoCertifiedChain)
        );
    }

    #[test]
    fn an_epoch_change_rebinds_the_same_key_with_a_new_certificate() {
        // Rotation policy: an epoch change REBINDS the same per-run key (journal identity stays
        // stable); only an incarnation change rotates the key itself.
        let base = key(1);
        let run_key = peer_id(&key(2));
        let epoch0 = scope(run_id(9), "trainer", 7, 0, module(0xAA));
        let epoch1 = scope(run_id(9), "trainer", 7, 1, module(0xBB));
        let cert0 = issue(&base, epoch0.clone(), run_key);
        let cert1 = issue(&base, epoch1.clone(), run_key);
        let base_id = peer_id(&base);
        let store = [cert0, cert1];
        assert!(verify_certified_sender(&epoch0, &run_key, &base_id, &store).is_ok());
        assert!(verify_certified_sender(&epoch1, &run_key, &base_id, &store).is_ok());
    }

    #[test]
    fn certificate_round_trips_through_canonical_cbor() {
        let base = key(1);
        let run_key = peer_id(&key(2));
        let cert = issue(
            &base,
            scope(run_id(9), "trainer", 7, 5, module(0xAB)),
            run_key,
        );
        let bytes = crate::canonical::to_canonical_vec(&cert).unwrap();
        let back: RunKeyCertificate = crate::canonical::from_canonical_slice(&bytes).unwrap();
        assert_eq!(cert, back);
        assert!(back.verify_chain().is_ok());
    }
}
