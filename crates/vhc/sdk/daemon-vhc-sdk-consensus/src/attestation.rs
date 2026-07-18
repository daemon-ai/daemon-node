// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Checkpoint attestation tiers (architecture §5.3; refactor §9 Phase E).
//!
//! Attestation is **two-tiered, and the thresholds are coordinator-module policy**:
//!
//! - A **digest attestation** — *"the checkpoint's declared digest equals my consensus state at
//!   fence E"* — can be signed immediately by any live peer, since every peer already holds that
//!   digest. This breaks the chicken-and-egg of a brand-new checkpoint: it proves *consistency with
//!   consensus state* before anyone has restored it.
//! - A **restore attestation** — *"I loaded the full manifest and it verified"* — accumulates
//!   afterwards and is the stronger claim: it proves *recoverability*, not merely consistency.
//!
//! The launch coordinator module requires **K digest attestations** for initial join-eligibility
//! ([`AttestationPolicy::join_eligibility`]) and **prefers restore-attested checkpoints once any
//! exist** ([`AttestationPolicy::preferred_checkpoint`]). Both are pure policy over an
//! [`AttestationLedger`] of authenticated attestations — the vocabulary E3 (cold join) drives.
//!
//! Each attestation is a small signed record over a checkpoint's **content hash**
//! ([`crate::checkpoint::CheckpointManifest::content_hash`]) under a domain-separated preimage, so a
//! signature is meaningless in any other protocol/run and forgery is an ed25519 problem. Signing
//! reuses the same primitives as [`crate::authority`] (`sign_canonical` / `verify_sig`).

use crate::messages::CheckpointAttestation as WireAttestation;
use daemon_vhc_proto::{
    peer_id, sign_canonical, to_canonical_vec, verify_sig, Hash, PeerId, Signature, SigningKey,
    StateDigest, VerifyOutcome,
};
use serde::{Deserialize, Serialize};

/// Domain-separation tag bound into every attestation preimage (so a signature is scoped to
/// checkpoint attestation and cannot be replayed as any other signed frame) — the registry
/// constant, re-exported.
pub use daemon_vhc_proto::domains::CHECKPOINT_ATTESTATION_DOMAIN as ATTEST_DOMAIN;

/// Which claim an attestation makes about a checkpoint (architecture §5.3). Numeric values are
/// permanent wire tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(into = "u64", try_from = "u64")]
pub enum AttestationTier {
    /// *Consistency with consensus state*: the checkpoint's declared digest equals the signer's
    /// consensus state at the fence. Signable immediately by any live peer.
    Digest,
    /// *Recoverability*: the signer loaded the full manifest and it verified. The stronger claim,
    /// accumulating only after a peer has actually restored.
    Restore,
}

impl AttestationTier {
    /// The permanent numeric wire tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Digest => 0,
            Self::Restore => 1,
        }
    }
}

impl From<AttestationTier> for u64 {
    fn from(t: AttestationTier) -> u64 {
        t.tag()
    }
}

impl TryFrom<u64> for AttestationTier {
    type Error = AttestationError;
    fn try_from(v: u64) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Digest),
            1 => Ok(Self::Restore),
            other => Err(AttestationError::UnknownTier(other)),
        }
    }
}

/// The signed body of a checkpoint attestation. The `signer` is bound in the body, so an
/// attestation is self-describing; verification is an ed25519 check of `signer` over the
/// domain-separated canonical encoding of this body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationBody {
    /// The claim tier (digest / restore).
    pub tier: AttestationTier,
    /// The run identity (genesis hash) the checkpoint belongs to.
    pub run_id: Hash,
    /// The epoch the checkpoint captures.
    pub epoch: u64,
    /// The round the checkpoint captures.
    pub round: u64,
    /// The attested checkpoint's **content hash** (`CheckpointManifest::content_hash`).
    pub checkpoint: Hash,
    /// The consensus-state digest the checkpoint reproduces (the value a digest attestation asserts
    /// equals the signer's own consensus state at the fence).
    pub digest: StateDigest,
    /// The attesting peer's identity (ed25519 public key).
    pub signer: PeerId,
}

impl AttestationBody {
    /// The domain-separated signing preimage: canonical CBOR of `(ATTEST_DOMAIN, self)`.
    ///
    /// # Errors
    /// Propagates a codec error (structurally unreachable for this type).
    pub fn preimage(&self) -> Result<Vec<u8>, AttestationError> {
        to_canonical_vec(&(ATTEST_DOMAIN, self)).map_err(|e| AttestationError::Codec(e.to_string()))
    }

    /// Sign this body with `sk` (whose public key MUST equal `self.signer`), producing a
    /// [`SignedAttestation`].
    ///
    /// # Errors
    /// [`AttestationError::SignerMismatch`] if `sk` is not `self.signer`; [`AttestationError::Codec`]
    /// on an encoding failure.
    pub fn sign(self, sk: &SigningKey) -> Result<SignedAttestation, AttestationError> {
        if peer_id(sk) != self.signer {
            return Err(AttestationError::SignerMismatch);
        }
        let sig = sign_canonical(sk, &(ATTEST_DOMAIN, &self))
            .map_err(|e| AttestationError::Codec(e.to_string()))?;
        Ok(SignedAttestation { body: self, sig })
    }
}

/// A checkpoint attestation plus the signature over its domain-separated preimage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedAttestation {
    /// The attestation body.
    pub body: AttestationBody,
    /// The ed25519 signature by `body.signer` over `body.preimage()`.
    pub sig: Signature,
}

impl SignedAttestation {
    /// Verify the signature by `body.signer` over the domain-separated preimage.
    ///
    /// # Errors
    /// [`AttestationError::BadSignature`] on an invalid signature, [`AttestationError::Malformed`]
    /// on a malformed key/signature, or [`AttestationError::Codec`] on an encoding failure.
    pub fn verify(&self) -> Result<(), AttestationError> {
        let preimage = self.body.preimage()?;
        match verify_sig(&self.body.signer.0, &self.sig.0, &preimage) {
            VerifyOutcome::Valid => Ok(()),
            VerifyOutcome::Invalid => Err(AttestationError::BadSignature),
            VerifyOutcome::Malformed => Err(AttestationError::Malformed),
        }
    }

    /// Project onto the control-plane wire form (`VhcMessage::CheckpointAttestation`) — the E3
    /// coordinator-flow carrier. Field-for-field; the inner signature rides verbatim.
    #[must_use]
    pub fn to_wire(&self) -> WireAttestation {
        WireAttestation {
            tier: self.body.tier.tag(),
            run_id: self.body.run_id,
            epoch: self.body.epoch,
            round: self.body.round,
            checkpoint: self.body.checkpoint,
            digest: self.body.digest,
            signer: self.body.signer,
            sig: self.sig,
        }
    }

    /// Reconstruct from the control-plane wire form. Decodes the tier tag; does **not** verify —
    /// call [`verify`](Self::verify) (the coordinator's `tick` does).
    ///
    /// # Errors
    /// [`AttestationError::UnknownTier`] on an unrecognized tier tag.
    pub fn from_wire(wire: &WireAttestation) -> Result<Self, AttestationError> {
        Ok(Self {
            body: AttestationBody {
                tier: AttestationTier::try_from(wire.tier)?,
                run_id: wire.run_id,
                epoch: wire.epoch,
                round: wire.round,
                checkpoint: wire.checkpoint,
                digest: wire.digest,
                signer: wire.signer,
            },
            sig: wire.sig,
        })
    }
}

/// An accumulating set of **verified** checkpoint attestations, deduped by
/// `(checkpoint, tier, signer)` so one peer's re-sent attestation never inflates a count.
///
/// Serializable + comparable because the coordinator carries it in `CoordinatorState` (the E3
/// coordinator flow): the ledger is consensus-visible state that must round-trip the state
/// snapshot byte-identically.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationLedger {
    entries: Vec<SignedAttestation>,
}

impl AttestationLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Verify and record an attestation. Returns `Ok(true)` if it was newly recorded, `Ok(false)`
    /// if it duplicated an existing `(checkpoint, tier, signer)` entry.
    ///
    /// # Errors
    /// The attestation's verification error (invalid/malformed signature).
    pub fn record(&mut self, att: SignedAttestation) -> Result<bool, AttestationError> {
        att.verify()?;
        let dup = self.entries.iter().any(|e| {
            e.body.checkpoint == att.body.checkpoint
                && e.body.tier == att.body.tier
                && e.body.signer == att.body.signer
        });
        if dup {
            return Ok(false);
        }
        self.entries.push(att);
        Ok(true)
    }

    /// Distinct signers who attested `checkpoint` at `tier`.
    #[must_use]
    pub fn signers(&self, checkpoint: &Hash, tier: AttestationTier) -> Vec<PeerId> {
        let mut signers: Vec<PeerId> = self
            .entries
            .iter()
            .filter(|e| e.body.checkpoint == *checkpoint && e.body.tier == tier)
            .map(|e| e.body.signer)
            .collect();
        signers.sort_unstable();
        signers.dedup();
        signers
    }

    /// The number of distinct signers who attested `checkpoint` at `tier`.
    #[must_use]
    pub fn count(&self, checkpoint: &Hash, tier: AttestationTier) -> u32 {
        u32::try_from(self.signers(checkpoint, tier).len()).unwrap_or(u32::MAX)
    }

    /// Whether `checkpoint` has at least one restore attestation.
    #[must_use]
    pub fn is_restore_attested(&self, checkpoint: &Hash) -> bool {
        self.count(checkpoint, AttestationTier::Restore) > 0
    }
}

/// The coordinator-module attestation policy (architecture §5.3): the K-digest join gate and the
/// restore-attested preference. Thresholds are coordinator policy, so they live in a value the
/// coordinator module configures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttestationPolicy {
    /// The number of distinct digest attestations required for a checkpoint to gate initial
    /// join-eligibility (the launch coordinator's `K`).
    pub k_digest: u32,
}

impl AttestationPolicy {
    /// A policy requiring `k_digest` digest attestations for join-eligibility.
    #[must_use]
    pub fn new(k_digest: u32) -> Self {
        Self { k_digest }
    }

    /// Whether `checkpoint` is join-eligible under this policy: it carries at least `k_digest`
    /// distinct digest attestations (architecture §5.3).
    #[must_use]
    pub fn join_eligibility(
        &self,
        ledger: &AttestationLedger,
        checkpoint: &Hash,
    ) -> JoinEligibility {
        let have = ledger.count(checkpoint, AttestationTier::Digest);
        if have >= self.k_digest {
            JoinEligibility::Eligible {
                digest_attestations: have,
            }
        } else {
            JoinEligibility::Ineligible {
                have,
                need: self.k_digest,
            }
        }
    }

    /// Choose the checkpoint a late joiner should restore from, among `candidates`, preferring
    /// **restore-attested** checkpoints once any exist and only considering **join-eligible** ones
    /// (architecture §5.3). Among equally-preferred candidates, the one with the most digest
    /// attestations wins; ties break by content-hash order for determinism. Returns `None` if no
    /// candidate is join-eligible.
    #[must_use]
    pub fn preferred_checkpoint(
        &self,
        ledger: &AttestationLedger,
        candidates: &[Hash],
    ) -> Option<Hash> {
        candidates
            .iter()
            .filter(|c| {
                matches!(
                    self.join_eligibility(ledger, c),
                    JoinEligibility::Eligible { .. }
                )
            })
            .max_by(|a, b| {
                // Preference order: restore-attested first, then digest-attestation count, then
                // (for a total order) the content hash itself.
                let ka = (
                    ledger.is_restore_attested(a),
                    ledger.count(a, AttestationTier::Digest),
                    **a,
                );
                let kb = (
                    ledger.is_restore_attested(b),
                    ledger.count(b, AttestationTier::Digest),
                    **b,
                );
                ka.cmp(&kb)
            })
            .copied()
    }
}

/// The join-eligibility verdict for a checkpoint under an [`AttestationPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinEligibility {
    /// The checkpoint has enough digest attestations to gate initial join.
    Eligible {
        /// The distinct digest attestations observed (≥ `k_digest`).
        digest_attestations: u32,
    },
    /// Not yet enough digest attestations.
    Ineligible {
        /// Distinct digest attestations observed.
        have: u32,
        /// The threshold required.
        need: u32,
    },
}

impl JoinEligibility {
    /// Whether the checkpoint is join-eligible.
    #[must_use]
    pub fn is_eligible(&self) -> bool {
        matches!(self, Self::Eligible { .. })
    }
}

/// Errors surfaced by the attestation layer. (`Display`/`Error` hand-written to keep the crate
/// dependency-light + wasm32-clean.)
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttestationError {
    /// The signing key's public identity did not match the body's declared `signer`.
    SignerMismatch,
    /// The signature did not verify over the domain-separated preimage.
    BadSignature,
    /// A public key or signature was structurally malformed.
    Malformed,
    /// An attestation-tier wire tag was not recognized.
    UnknownTier(u64),
    /// An attestation (de)serialization step failed.
    Codec(String),
}

impl core::fmt::Display for AttestationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SignerMismatch => {
                write!(f, "signing key does not match the attestation's signer")
            }
            Self::BadSignature => write!(f, "attestation signature does not verify"),
            Self::Malformed => write!(f, "malformed public key or signature"),
            Self::UnknownTier(t) => write!(f, "unknown attestation tier tag {t}"),
            Self::Codec(e) => write!(f, "attestation codec error: {e}"),
        }
    }
}

impl std::error::Error for AttestationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{CheckpointManifest, SectionKind};

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn manifest(round: u64, tag: u8) -> CheckpointManifest {
        CheckpointManifest::builder(
            Hash([1; 32]),
            0,
            round,
            Hash([2; 32]),
            StateDigest([tag; 16]),
        )
        .section("consensus", SectionKind::Consensus, 1, &[tag; 8])
        .section("module", SectionKind::Module, 1, &[tag; 32])
        .build()
        .unwrap()
    }

    fn attest(sk: &SigningKey, m: &CheckpointManifest, tier: AttestationTier) -> SignedAttestation {
        AttestationBody {
            tier,
            run_id: m.run_id,
            epoch: m.epoch,
            round: m.round,
            checkpoint: m.content_hash().unwrap(),
            digest: m.digest,
            signer: peer_id(sk),
        }
        .sign(sk)
        .unwrap()
    }

    #[test]
    fn attestation_sign_verify_round_trips_and_rejects_tampering() {
        let sk = key(1);
        let m = manifest(5, 0xAA);
        let att = attest(&sk, &m, AttestationTier::Digest);
        att.verify().unwrap();
        // Wire round-trip preserves verifiability.
        let wire = to_canonical_vec(&att).unwrap();
        let back: SignedAttestation = daemon_vhc_proto::from_canonical_slice(&wire).unwrap();
        back.verify().unwrap();
        // Tampering with the attested checkpoint invalidates the signature.
        let mut tampered = att.clone();
        tampered.body.checkpoint = Hash([0xff; 32]);
        assert_eq!(tampered.verify(), Err(AttestationError::BadSignature));
    }

    #[test]
    fn signer_mismatch_is_refused() {
        let sk = key(1);
        let m = manifest(5, 0xAA);
        let body = AttestationBody {
            tier: AttestationTier::Digest,
            run_id: m.run_id,
            epoch: m.epoch,
            round: m.round,
            checkpoint: m.content_hash().unwrap(),
            digest: m.digest,
            signer: peer_id(&key(2)), // not sk
        };
        assert_eq!(body.sign(&sk), Err(AttestationError::SignerMismatch));
    }

    #[test]
    fn k_digest_attestations_gate_join_eligibility() {
        let m = manifest(5, 0xAA);
        let ckpt = m.content_hash().unwrap();
        let policy = AttestationPolicy::new(3);
        let mut ledger = AttestationLedger::new();

        // Below K → ineligible; the count is reported.
        assert_eq!(
            policy.join_eligibility(&ledger, &ckpt),
            JoinEligibility::Ineligible { have: 0, need: 3 }
        );
        for seed in 1..=2u8 {
            ledger
                .record(attest(&key(seed), &m, AttestationTier::Digest))
                .unwrap();
        }
        assert!(!policy.join_eligibility(&ledger, &ckpt).is_eligible());

        // The Kth distinct signer flips eligibility.
        ledger
            .record(attest(&key(3), &m, AttestationTier::Digest))
            .unwrap();
        assert_eq!(
            policy.join_eligibility(&ledger, &ckpt),
            JoinEligibility::Eligible {
                digest_attestations: 3
            }
        );

        // A duplicate signer does not inflate the count.
        assert!(!ledger
            .record(attest(&key(3), &m, AttestationTier::Digest))
            .unwrap());
        assert_eq!(ledger.count(&ckpt, AttestationTier::Digest), 3);
    }

    #[test]
    fn restore_attested_checkpoint_is_preferred_once_it_exists() {
        // Two competing join-eligible checkpoints, both with K digest attestations. Before any
        // restore attestation, preference falls to the deterministic tie-break; once one gains a
        // restore attestation, it is preferred (recoverability > consistency, architecture §5.3).
        let policy = AttestationPolicy::new(2);
        let older = manifest(5, 0x11);
        let newer = manifest(6, 0x22);
        let (ch_old, ch_new) = (older.content_hash().unwrap(), newer.content_hash().unwrap());
        let mut ledger = AttestationLedger::new();
        for seed in 1..=2u8 {
            ledger
                .record(attest(&key(seed), &older, AttestationTier::Digest))
                .unwrap();
            ledger
                .record(attest(&key(seed), &newer, AttestationTier::Digest))
                .unwrap();
        }
        // Both eligible; the preference is stable (tie-break) but arbitrary between them here.
        let before = policy
            .preferred_checkpoint(&ledger, &[ch_old, ch_new])
            .unwrap();
        assert!(before == ch_old || before == ch_new);

        // A peer restores the OLDER checkpoint and signs a restore attestation.
        ledger
            .record(attest(&key(9), &older, AttestationTier::Restore))
            .unwrap();
        assert!(ledger.is_restore_attested(&ch_old));
        assert!(!ledger.is_restore_attested(&ch_new));
        // Now the restore-attested one is preferred regardless of tie-break.
        assert_eq!(
            policy.preferred_checkpoint(&ledger, &[ch_old, ch_new]),
            Some(ch_old)
        );

        // A checkpoint below K is never chosen even if restore-attested.
        let thin = manifest(7, 0x33);
        let ch_thin = thin.content_hash().unwrap();
        ledger
            .record(attest(&key(1), &thin, AttestationTier::Digest))
            .unwrap(); // only 1 < K=2
        ledger
            .record(attest(&key(9), &thin, AttestationTier::Restore))
            .unwrap();
        assert_eq!(policy.preferred_checkpoint(&ledger, &[ch_thin]), None);
    }
}
