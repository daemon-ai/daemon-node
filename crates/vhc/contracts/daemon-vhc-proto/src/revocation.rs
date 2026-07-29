// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Signed revocation records for certified per-run keys (architecture §4.3 companion to
//! [`crate::cert`]).
//!
//! A revocation record is the base identity's signed statement that a per-run key it previously
//! certified is dead: receivers refuse the key's frames from ingestion on, regardless of any
//! still-chain-valid certificate. Records are distributed on the control plane best-effort;
//! **supersession is the safety floor** — a certificate for a HIGHER incarnation of the same
//! `(run, role)` slot implicitly revokes every lower incarnation, and peers that never observe an
//! explicit revocation still enforce that ordering (incarnations are never reused, ABI §8.1, so
//! the ordering is total and rollback-free).
//!
//! Replay protection: every record carries a **monotonic per-`(run, role)` sequence**. A ledger
//! ingests a record only if its sequence is strictly greater than the last one ingested for that
//! slot — a captured old record can never be re-presented to roll state back (revocation is
//! monotone anyway; the sequence also gives supersession statements a total order).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Signature};
use crate::cert::{CertError, CertScope, RunKeyCertificate};
use crate::error::VhcProtoError;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};

/// The domain-separation tag every run-key revocation body carries at ABI major 2. Distinct from
/// both the frame domain (`daemon-vhc/frame/2`) and the certificate domain (`daemon-vhc/cert/2`)
/// so no signature is replayable across record kinds.
pub const REVOCATION_DOMAIN_V2: &str = "daemon-vhc/revocation/2";

/// The signed body of a run-key revocation: the statement the base identity attests to. Every
/// field is part of the signed preimage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunKeyRevocationBody {
    /// Domain-separation tag — MUST be [`REVOCATION_DOMAIN_V2`].
    pub domain: String,
    /// The run whose per-run key is revoked (the genesis-envelope hash, ABI §8.1 `run_id`).
    pub run_id: Hash,
    /// The envelope-level role label the revoked key acted as.
    pub role: String,
    /// The role-instance incarnation whose key is revoked (ABI §8.1 `instance`).
    pub instance: u64,
    /// The revoked per-run public key.
    pub revoked_key: PeerId,
    /// The monotonic per-`(run, role)` revocation sequence — replay protection: a ledger ingests
    /// only strictly-increasing sequences for a slot.
    pub sequence: u64,
}

/// A run-key revocation record: the signed [`RunKeyRevocationBody`], the base machine identity
/// that signed it, and its ed25519 signature over the canonical CBOR of the body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunKeyRevocation {
    /// The signed statement.
    pub body: RunKeyRevocationBody,
    /// The base machine identity (the same authority that issued the key's certificate).
    pub base_identity: PeerId,
    /// ed25519 signature by `base_identity` over the canonical CBOR of `body`.
    pub sig: Signature,
}

/// Why a revocation record (or a ledger ingest) was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RevocationError {
    /// The body's domain tag is not [`REVOCATION_DOMAIN_V2`].
    WrongDomain {
        /// The tag actually carried.
        got: String,
    },
    /// The base identity's signature over the body does not verify.
    BadChain,
    /// The record's base identity is not one the receiver trusts for this run.
    UntrustedBase,
    /// The record's sequence is not strictly greater than the last ingested sequence for its
    /// `(run, role)` slot — a replayed or stale record, refused typed.
    StaleSequence {
        /// The sequence carried by the refused record.
        got: u64,
        /// The highest sequence already ingested for the slot.
        last: u64,
    },
}

impl core::fmt::Display for RevocationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongDomain { got } => {
                write!(
                    f,
                    "revocation domain `{got}` is not `{REVOCATION_DOMAIN_V2}`"
                )
            }
            Self::BadChain => write!(
                f,
                "revocation signature does not verify to the base identity"
            ),
            Self::UntrustedBase => write!(f, "revocation base identity is not trusted"),
            Self::StaleSequence { got, last } => write!(
                f,
                "revocation sequence {got} is not greater than the last ingested {last} \
                 (replayed or stale record)"
            ),
        }
    }
}

impl std::error::Error for RevocationError {}

impl From<RevocationError> for VhcProtoError {
    fn from(e: RevocationError) -> Self {
        VhcProtoError::Signature(e.to_string())
    }
}

impl RunKeyRevocation {
    /// Issue a revocation: the base identity `base_key` declares the per-run key `revoked_key`
    /// (bound to `(run_id, role, instance)`) dead, at `sequence` in the slot's monotonic stream.
    ///
    /// # Errors
    /// A signing failure (canonical-CBOR encode / ed25519), surfaced as [`VhcProtoError`].
    pub fn issue(
        base_key: &SigningKey,
        run_id: Hash,
        role: impl Into<String>,
        instance: u64,
        revoked_key: PeerId,
        sequence: u64,
    ) -> Result<Self, VhcProtoError> {
        let body = RunKeyRevocationBody {
            domain: REVOCATION_DOMAIN_V2.to_string(),
            run_id,
            role: role.into(),
            instance,
            revoked_key,
            sequence,
        };
        let sig = sign_canonical(base_key, &body)?;
        Ok(Self {
            body,
            base_identity: peer_id(base_key),
            sig,
        })
    }

    /// Verify the record chain: the domain tag is [`REVOCATION_DOMAIN_V2`] and the base
    /// identity's signature over the body verifies (`verify_strict`).
    ///
    /// # Errors
    /// [`RevocationError::WrongDomain`] or [`RevocationError::BadChain`].
    pub fn verify_chain(&self) -> Result<(), RevocationError> {
        if self.body.domain != REVOCATION_DOMAIN_V2 {
            return Err(RevocationError::WrongDomain {
                got: self.body.domain.clone(),
            });
        }
        verify_canonical(&self.base_identity, &self.sig, &self.body)
            .map_err(|_| RevocationError::BadChain)
    }
}

/// One `(run, role, base identity)` slot key of the ledger.
///
/// The base identity is LOAD-BEARING: a role names a duty, not a seat. A run whose roster
/// carries two trainers has two independent `(role = "trainer")` incarnation ladders — one per
/// base identity — and each base may only supersede (or revoke) its OWN keys. A `(run, role)`
/// key conflated them: the peer with the higher incarnation (a box that had churned more) set
/// the floor for the whole role and every sibling's live key was refused as revoked. Observed
/// live on the two-box WAN rung as a `CertRevoked` refusal stream against the slower-churning
/// trainer.
type SlotKey = (Hash, String, PeerId);

/// A receiver's revocation state for a run: explicitly revoked keys (from ingested signed
/// records, replay-protected) plus the **supersession floor** — the highest incarnation each
/// `(run, role, base)` slot has been observed certified at. Explicit revocation is best-effort
/// delivery; the floor is the safety guarantee that holds under partition.
#[derive(Debug, Default)]
pub struct RevocationLedger {
    /// Explicitly revoked `(run, role, instance, key)` bindings.
    revoked: BTreeSet<(Hash, String, u64, PeerId)>,
    /// The highest ingested revocation sequence per slot (replay protection).
    last_sequence: BTreeMap<SlotKey, u64>,
    /// The highest incarnation observed certified per slot (supersession floor).
    incarnation_floor: BTreeMap<SlotKey, u64>,
}

impl RevocationLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one signed revocation record: chain-verify, require the issuing base to be
    /// `trusted_base`, and enforce the slot's strictly-monotonic sequence (replay protection).
    ///
    /// # Errors
    /// The applicable [`RevocationError`]; a refused record changes no ledger state.
    pub fn ingest(
        &mut self,
        record: &RunKeyRevocation,
        trusted_base: &PeerId,
    ) -> Result<(), RevocationError> {
        record.verify_chain()?;
        if record.base_identity != *trusted_base {
            return Err(RevocationError::UntrustedBase);
        }
        let slot = (
            record.body.run_id,
            record.body.role.clone(),
            record.base_identity,
        );
        if let Some(&last) = self.last_sequence.get(&slot) {
            if record.body.sequence <= last {
                return Err(RevocationError::StaleSequence {
                    got: record.body.sequence,
                    last,
                });
            }
        }
        self.last_sequence.insert(slot, record.body.sequence);
        self.revoked.insert((
            record.body.run_id,
            record.body.role.clone(),
            record.body.instance,
            record.body.revoked_key,
        ));
        Ok(())
    }

    /// Observe a certified incarnation for a slot (called when a certificate is accepted into a
    /// store): the base's supersession floor advances to the highest incarnation seen — implicit
    /// revocation of every lower incarnation OF THAT BASE, enforced even when no explicit record
    /// arrives. `base` is the certifying base identity: supersession is per identity ladder,
    /// never across roster siblings sharing the role (see [`SlotKey`]).
    pub fn observe_certified_incarnation(
        &mut self,
        run_id: Hash,
        role: &str,
        base: PeerId,
        instance: u64,
    ) {
        let slot = (run_id, role.to_string(), base);
        let floor = self.incarnation_floor.entry(slot).or_insert(instance);
        *floor = (*floor).max(instance);
    }

    /// Judge a certified sender against the ledger: refused typed if the key was explicitly
    /// revoked for its `(run, role, instance)`, or if its incarnation is below ITS OWN base's
    /// supersession floor (a higher incarnation of the same identity wins, with or without an
    /// explicit record). `base` is the base identity whose certificate authenticated the sender
    /// — a sibling's ladder never judges this sender (see [`SlotKey`]).
    ///
    /// # Errors
    /// [`CertError::Revoked`] when the sender's key is dead by either rule.
    pub fn judge(
        &self,
        scope: &CertScope,
        sender: &PeerId,
        base: &PeerId,
    ) -> Result<(), CertError> {
        if self
            .revoked
            .contains(&(scope.run_id, scope.role.clone(), scope.instance, *sender))
        {
            return Err(CertError::Revoked);
        }
        if let Some(&floor) = self
            .incarnation_floor
            .get(&(scope.run_id, scope.role.clone(), *base))
        {
            if scope.instance < floor {
                return Err(CertError::Revoked);
            }
        }
        Ok(())
    }

    /// The highest certified incarnation observed for `(run, role)` ACROSS ALL base identities
    /// — the single-slot fencing floor. Frame-sender supersession is per base ladder
    /// ([`SlotKey`]: roster siblings sharing a role run parallel ladders that never judge each
    /// other), but a SEAT is one slot per role by construction: its fencing-token lineage spans
    /// claimants, so a seat judgment consults the cross-base maximum — a stale claimant is dead
    /// once ANY successor holds a higher token, whichever base certified it.
    #[must_use]
    pub fn role_floor(&self, run_id: &Hash, role: &str) -> Option<u64> {
        self.incarnation_floor
            .iter()
            .filter(|((r, ro, _), _)| r == run_id && ro == role)
            .map(|(_, floor)| *floor)
            .max()
    }

    /// Observe every certificate in a store (their scopes advance their base's supersession
    /// floor).
    pub fn observe_certificates(&mut self, certs: &[RunKeyCertificate]) {
        for cert in certs {
            self.observe_certified_incarnation(
                cert.body.scope.run_id,
                &cert.body.scope.role,
                cert.base_identity,
                cert.body.scope.instance,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn run(n: u8) -> Hash {
        Hash([n; 32])
    }

    fn scope(instance: u64) -> CertScope {
        CertScope {
            run_id: run(1),
            epoch: 0,
            role: "trainer".into(),
            instance,
            module_hash: Hash([0xAA; 32]),
        }
    }

    #[test]
    fn a_revocation_record_round_trips_and_chain_verifies() {
        let base = key(1);
        let revoked = peer_id(&key(2));
        let rec = RunKeyRevocation::issue(&base, run(1), "trainer", 3, revoked, 1).unwrap();
        assert!(rec.verify_chain().is_ok());
        let bytes = crate::canonical::to_canonical_vec(&rec).unwrap();
        let back: RunKeyRevocation = crate::canonical::from_canonical_slice(&bytes).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn a_tampered_or_wrong_domain_record_is_refused() {
        let base = key(1);
        let revoked = peer_id(&key(2));
        let mut rec = RunKeyRevocation::issue(&base, run(1), "trainer", 3, revoked, 1).unwrap();
        rec.body.instance = 4; // re-scope without re-signing
        assert_eq!(rec.verify_chain(), Err(RevocationError::BadChain));

        let mut wrong = RunKeyRevocation::issue(&base, run(1), "trainer", 3, revoked, 1).unwrap();
        wrong.body.domain = "daemon-vhc/cert/2".into(); // a cert tag, not a revocation tag
        assert!(matches!(
            wrong.verify_chain(),
            Err(RevocationError::WrongDomain { .. })
        ));
    }

    #[test]
    fn an_ingested_revocation_kills_the_key_and_replays_are_refused() {
        let base = key(1);
        let base_id = peer_id(&base);
        let victim = peer_id(&key(2));
        let mut ledger = RevocationLedger::new();

        // Before ingestion the key is live.
        assert!(ledger.judge(&scope(3), &victim, &base_id).is_ok());

        let rec = RunKeyRevocation::issue(&base, run(1), "trainer", 3, victim, 1).unwrap();
        ledger.ingest(&rec, &base_id).unwrap();
        assert_eq!(
            ledger.judge(&scope(3), &victim, &base_id),
            Err(CertError::Revoked)
        );

        // Replay of the same record (same sequence) is refused typed — no state change.
        assert_eq!(
            ledger.ingest(&rec, &base_id),
            Err(RevocationError::StaleSequence { got: 1, last: 1 })
        );
        // A lower sequence is equally stale.
        let stale =
            RunKeyRevocation::issue(&base, run(1), "trainer", 2, peer_id(&key(9)), 0).unwrap();
        assert_eq!(
            ledger.ingest(&stale, &base_id),
            Err(RevocationError::StaleSequence { got: 0, last: 1 })
        );
        // The next sequence ingests.
        let next =
            RunKeyRevocation::issue(&base, run(1), "trainer", 4, peer_id(&key(9)), 2).unwrap();
        ledger.ingest(&next, &base_id).unwrap();
    }

    #[test]
    fn a_record_from_an_untrusted_base_is_refused() {
        let attacker = key(50);
        let victim = peer_id(&key(2));
        let rec = RunKeyRevocation::issue(&attacker, run(1), "trainer", 3, victim, 1).unwrap();
        let mut ledger = RevocationLedger::new();
        let trusted = peer_id(&key(1));
        assert_eq!(
            ledger.ingest(&rec, &trusted),
            Err(RevocationError::UntrustedBase)
        );
        // The refused record changed nothing: the key is still live.
        assert!(ledger.judge(&scope(3), &victim, &trusted).is_ok());
    }

    #[test]
    fn supersession_fences_lower_incarnations_without_an_explicit_record() {
        // Under partition a peer may never see the explicit revocation — observing the HIGHER
        // incarnation's certificate is enough: the floor advances and the old key is refused.
        let base_id = peer_id(&key(1));
        let old_key = peer_id(&key(2));
        let mut ledger = RevocationLedger::new();
        ledger.observe_certified_incarnation(run(1), "trainer", base_id, 4);
        assert_eq!(
            ledger.judge(&scope(3), &old_key, &base_id),
            Err(CertError::Revoked)
        );
        // The current incarnation stays live.
        assert!(ledger.judge(&scope(4), &peer_id(&key(3)), &base_id).is_ok());
        // The floor never rolls back.
        ledger.observe_certified_incarnation(run(1), "trainer", base_id, 2);
        assert_eq!(
            ledger.judge(&scope(3), &old_key, &base_id),
            Err(CertError::Revoked)
        );
    }

    #[test]
    fn a_siblings_ladder_never_supersedes_this_base() {
        // Two roster seats share the role "trainer" but are DIFFERENT base identities: one
        // seat churning to a high incarnation must not fence the other's live low incarnation.
        // (The regression: a (run, role)-keyed floor let the churnier trainer revoke its
        // sibling — observed live as a CertRevoked stream on the two-box WAN rung.)
        let churny = peer_id(&key(1));
        let steady = peer_id(&key(2));
        let steady_key = peer_id(&key(3));
        let mut ledger = RevocationLedger::new();
        ledger.observe_certified_incarnation(run(1), "trainer", churny, 7);
        // The steady sibling at incarnation 1 stays live — its own ladder has no higher rung.
        assert!(ledger.judge(&scope(1), &steady_key, &steady).is_ok());
        // The churny base's own old rungs are fenced as before.
        assert_eq!(
            ledger.judge(&scope(1), &steady_key, &churny),
            Err(CertError::Revoked)
        );
    }
}
