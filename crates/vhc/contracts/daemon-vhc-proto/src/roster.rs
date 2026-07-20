// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **iroh roster record** — a node's signed, registry-served statement of its iroh transport
//! reachability for one run (architecture §6.3; the transport analogue of the coordinator seat
//! lease).
//!
//! Each admitted node publishes one record per run: its iroh `EndpointId` (the transport public
//! key, distinct from every signing identity — §7.2) plus how to reach it (direct socket
//! addresses and/or a relay URL), as a canonical-CBOR [`RosterRecordBody`] signed by the node's
//! **certified per-run key**, carried with its [`RunKeyCertificate`] (the seat-lease
//! distribution shape — never a frame-envelope field, ABI §12.1). Peers fetch the run's roster
//! from the registry and verify every entry themselves BEFORE trusting an address: the registry
//! is untrusted CAS storage — it can withhold a record but cannot forge one, because trust comes
//! from the signature chain to a genesis-named base identity, never from storage.
//!
//! **Staleness is precedence, never wall clock.** A record's freshness key is
//! `(incarnation, issued_at_ms)`, ordered lexicographically: a higher incarnation (a rejoined
//! node) supersedes every record of a prior incarnation, and within one incarnation a later
//! `issued_at_ms` (a re-addressed node) supersedes an earlier one. Readers group records by
//! `(role, certificate base identity)` — the stable per-node key (the per-run signing key
//! rotates with the incarnation; the base identity does not) — and keep only the maximum
//! freshness key per group, so a withheld-newer/served-older registry can delay but never
//! roll back a reader that has already observed the newer record.
//!
//! The registry-side acceptance rule is the pure [`RosterSlot::fold`] — normative monotonic
//! upsert semantics every conforming roster registry implements bit-for-bit (the shared test
//! vectors `tests/fixtures/roster-vectors.json` pin it, exactly as the seat CAS vectors pin the
//! seat fold). The fold validates **structure only** (domain tag, dialability, size caps, slot
//! consistency, the freshness-key monotonicity); it MUST NOT verify signatures or judge
//! authority — peers do that.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, IrohId, PeerId, Signature};
use crate::cert::{CertError, CertScope, RunKeyCertificate};
use crate::domains::ROSTER_RECORD_DOMAIN;
use crate::error::VhcProtoError;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};

/// Cap on the direct addresses one record may carry (structural — a roster entry is a small
/// reachability hint, never a bulk address dump).
pub const MAX_ROSTER_DIRECT_ADDRS: usize = 8;
/// Cap on one direct-address string (`ip:port`; a bracketed IPv6 with port fits comfortably).
pub const MAX_ROSTER_ADDR_LEN: usize = 64;
/// Cap on the relay URL string.
pub const MAX_ROSTER_RELAY_LEN: usize = 256;
/// Per-run entry cap a conforming registry enforces before folding a NEW entry key (an existing
/// key's upsert is never blocked by the cap).
pub const MAX_ROSTER_ENTRIES: usize = 64;

/// The signed body of an iroh roster record. Every field is part of the signed preimage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterRecordBody {
    /// Domain-separation tag — MUST be [`ROSTER_RECORD_DOMAIN`].
    pub domain: String,
    /// The run's cryptographic identity: the genesis-envelope hash (ABI §8.1 `run_id`).
    pub run_id: Hash,
    /// The envelope role label this node runs (part of the reader's grouping key).
    pub role: String,
    /// The run epoch the signing identity is bound to (the certificate binds one epoch; an epoch
    /// change re-publishes under the reissued certificate).
    pub epoch: u64,
    /// The never-reused monotonic role-instance incarnation id (ABI §8.1 `instance`) — the major
    /// component of the freshness key: a higher incarnation supersedes every prior one.
    pub incarnation: u64,
    /// The publisher's certified per-run public key — the key that signs this record.
    pub sender: PeerId,
    /// The pinned module blob the publisher runs (ABI §8.1 `module_hash`; must match the
    /// embedded certificate's binding).
    pub module_hash: Hash,
    /// The publisher's iroh `EndpointId` (32 raw bytes — the transport public key, §7.2). Also
    /// the registry's entry key within a run.
    pub endpoint_id: IrohId,
    /// Direct socket addresses (`"ip:port"` strings); may be empty for relay-only reachability.
    #[serde(default)]
    pub direct_addrs: Vec<String>,
    /// Home relay URL (NAT-proof reachability); `None` for direct-only (LAN/loopback).
    #[serde(default)]
    pub relay_url: Option<String>,
    /// Publisher wall clock at issue (milliseconds) — the minor component of the freshness key:
    /// within one incarnation, a later issue supersedes (the re-address republish).
    pub issued_at_ms: u64,
}

/// An iroh roster record: the signed [`RosterRecordBody`], the publisher's
/// [`RunKeyCertificate`] (travelling beside the body as a separate distribution record), and the
/// publisher per-run key's ed25519 signature over the canonical CBOR of the body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterRecord {
    /// The signed reachability statement.
    pub body: RosterRecordBody,
    /// The certificate authorizing [`RosterRecordBody::sender`] for
    /// `(run_id, epoch, role, incarnation, module_hash)`.
    pub certificate: RunKeyCertificate,
    /// ed25519 signature by [`RosterRecordBody::sender`] over the canonical CBOR of `body`.
    pub sig: Signature,
}

/// Why a roster record was refused by a **verifier** (a peer, or the typed layers of the publish
/// constructor). Registry-side refusals are [`RosterDecision`]s — structural only.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RosterRecordError {
    /// The body's domain tag is not [`ROSTER_RECORD_DOMAIN`].
    WrongDomain {
        /// The tag actually carried.
        got: String,
    },
    /// The record names no dialable reachability (no direct address and no relay URL).
    NotDialable,
    /// The record exceeds a structural size cap (address count / address length / relay length).
    OverCap {
        /// Which cap was exceeded (diagnostic only).
        what: &'static str,
    },
    /// The role label is empty (the reader's grouping key would be meaningless).
    EmptyRole,
    /// `issued_at_ms` is zero — an unstamped record has no freshness-key minor component.
    UnstampedIssue,
    /// The sender per-run key's signature over the body does not verify.
    BadSignature,
    /// The embedded certificate does not authorize the sender for the record's scope
    /// (chain / scope / epoch / module / sender refusal, carried typed).
    Cert(CertError),
    /// The certificate's base identity is not one the run's genesis/Authority names.
    UntrustedBase,
}

impl core::fmt::Display for RosterRecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongDomain { got } => {
                write!(f, "roster domain `{got}` is not `{ROSTER_RECORD_DOMAIN}`")
            }
            Self::NotDialable => write!(f, "roster record names no dialable reachability"),
            Self::OverCap { what } => write!(f, "roster record exceeds the {what} cap"),
            Self::EmptyRole => write!(f, "roster record carries an empty role label"),
            Self::UnstampedIssue => write!(f, "roster record carries no issue timestamp"),
            Self::BadSignature => write!(f, "roster signature does not verify to the sender"),
            Self::Cert(e) => write!(f, "roster certificate refused: {e}"),
            Self::UntrustedBase => {
                write!(f, "roster certificate base identity is not genesis-trusted")
            }
        }
    }
}

impl std::error::Error for RosterRecordError {}

impl From<CertError> for RosterRecordError {
    fn from(e: CertError) -> Self {
        Self::Cert(e)
    }
}

impl From<RosterRecordError> for VhcProtoError {
    fn from(e: RosterRecordError) -> Self {
        VhcProtoError::Validation(e.to_string())
    }
}

impl RosterRecordBody {
    /// Structural validation — the invariants BOTH sides enforce (the registry before storing,
    /// every verifier before trusting): domain tag, a non-empty role, dialability, the size
    /// caps, and a non-zero issue stamp. Signature-free by design.
    ///
    /// # Errors
    /// The applicable [`RosterRecordError`].
    pub fn validate(&self) -> Result<(), RosterRecordError> {
        if self.domain != ROSTER_RECORD_DOMAIN {
            return Err(RosterRecordError::WrongDomain {
                got: self.domain.clone(),
            });
        }
        if self.role.is_empty() {
            return Err(RosterRecordError::EmptyRole);
        }
        if self.direct_addrs.is_empty() && self.relay_url.is_none() {
            return Err(RosterRecordError::NotDialable);
        }
        if self.direct_addrs.len() > MAX_ROSTER_DIRECT_ADDRS {
            return Err(RosterRecordError::OverCap {
                what: "direct-address count",
            });
        }
        if self
            .direct_addrs
            .iter()
            .any(|a| a.len() > MAX_ROSTER_ADDR_LEN)
        {
            return Err(RosterRecordError::OverCap {
                what: "direct-address length",
            });
        }
        if self
            .relay_url
            .as_ref()
            .is_some_and(|u| u.len() > MAX_ROSTER_RELAY_LEN)
        {
            return Err(RosterRecordError::OverCap {
                what: "relay-url length",
            });
        }
        if self.issued_at_ms == 0 {
            return Err(RosterRecordError::UnstampedIssue);
        }
        Ok(())
    }

    /// The execution-identity scope this record publishes under — what the embedded certificate
    /// must bind (the seat-lease scope rule).
    #[must_use]
    pub fn cert_scope(&self) -> CertScope {
        CertScope {
            run_id: self.run_id,
            epoch: self.epoch,
            role: self.role.clone(),
            instance: self.incarnation,
            module_hash: self.module_hash,
        }
    }

    /// The freshness key: `(incarnation, issued_at_ms)`, ordered lexicographically. Higher
    /// supersedes; equality is the idempotent republish.
    #[must_use]
    pub fn freshness(&self) -> (u64, u64) {
        (self.incarnation, self.issued_at_ms)
    }
}

impl RosterRecord {
    /// Author a record: validate the body structurally, require `run_key` to be the body's
    /// sender, and sign the canonical CBOR of the body with the sender per-run key. The
    /// certificate is issued separately by the base identity (once per binding) and travels here.
    ///
    /// # Errors
    /// A structural violation, a sender/key mismatch, or a signing failure.
    pub fn publish(
        run_key: &SigningKey,
        certificate: RunKeyCertificate,
        body: RosterRecordBody,
    ) -> Result<Self, VhcProtoError> {
        body.validate()?;
        if peer_id(run_key) != body.sender {
            return Err(VhcProtoError::Validation(
                "roster record sender is not the signing per-run key".into(),
            ));
        }
        let sig = sign_canonical(run_key, &body)?;
        Ok(Self {
            body,
            certificate,
            sig,
        })
    }

    /// Verify structure + the sender's self-signature over the body. This is the registry-free
    /// half of acceptance; authority is [`RosterRecord::authorize`].
    ///
    /// # Errors
    /// The applicable [`RosterRecordError`].
    pub fn verify_signature(&self) -> Result<(), RosterRecordError> {
        self.body.validate()?;
        verify_canonical(&self.body.sender, &self.sig, &self.body)
            .map_err(|_| RosterRecordError::BadSignature)
    }

    /// The full peer-side acceptance: structure, self-signature, the certificate chain to a
    /// **genesis-trusted** base identity, and the certificate's binding over exactly this
    /// record's scope and sender. There is no wall-clock expiry — staleness is the freshness-key
    /// precedence a reader applies across records (compose with a supersession judgment over
    /// [`RosterRecordBody::freshness`], grouped by `(role, base identity)`).
    ///
    /// # Errors
    /// The applicable [`RosterRecordError`]. The registry MUST NOT run this check — authority is
    /// never the registry's judgment.
    pub fn authorize(&self, trusted_bases: &[PeerId]) -> Result<(), RosterRecordError> {
        self.verify_signature()?;
        if !trusted_bases.contains(&self.certificate.base_identity) {
            return Err(RosterRecordError::UntrustedBase);
        }
        self.certificate
            .authorizes_sender(&self.body.cert_scope(), &self.body.sender)?;
        Ok(())
    }

    /// The reader's stable per-node grouping key: `(role, certificate base identity)`. The
    /// per-run signing key rotates with the incarnation and the endpoint id is transport-owned;
    /// the base identity is the durable node identity a chain verifies back to.
    #[must_use]
    pub fn group_key(&self) -> (String, PeerId) {
        (self.body.role.clone(), self.certificate.base_identity)
    }
}

// -- the registry-side slot + the normative monotonic-upsert fold ----------------------------------

/// The registry's structural verdict on one roster publish. Purely structural — an `Accepted`
/// says nothing about authority (peers judge that; the registry is untrusted storage).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RosterDecision {
    /// Stored; the slot now reflects the record.
    Accepted,
    /// The record violates a structural invariant (domain, dialability, caps, slot consistency).
    RejectedStructural {
        /// A human-readable reason (diagnostic only; never authority).
        reason: String,
    },
    /// The record's freshness key is below the stored one — a stale republish. The publisher
    /// re-reads and republishes at a fresher key (or accepts that a newer incarnation exists).
    RejectedStale {
        /// The stored incarnation.
        stored_incarnation: u64,
        /// The stored issue stamp.
        stored_issued_at_ms: u64,
    },
}

/// One run-endpoint roster slot as the registry stores it (the entry key is the record's
/// `endpoint_id` within the run; the run itself keys the outer map).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterSlot {
    /// The stored record, if the endpoint has ever published.
    pub record: Option<RosterRecord>,
}

impl RosterSlot {
    /// A never-published slot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The **normative registry fold** — pure: `(slot, record) → (next slot, decision)`. A
    /// refused publish never mutates the slot. Every conforming roster registry (the local
    /// fixture, the cloud implementation) applies exactly this function under its single-writer
    /// serialization; the shared test vectors pin it.
    ///
    /// Structural only (registry posture): the body invariants, slot-key consistency
    /// (`run_id`/`endpoint_id` against the stored record), and the freshness-key monotonic
    /// upsert — accept `(incarnation, issued_at_ms) >=` stored (equality is the idempotent
    /// republish; `>=` on the lexicographic pair is what lets a same-incarnation re-address
    /// through), refuse below. NO signature verification, NO authority judgment.
    #[must_use]
    pub fn fold(&self, record: &RosterRecord) -> (Self, RosterDecision) {
        let refuse = |d: RosterDecision| (self.clone(), d);
        if let Err(e) = record.body.validate() {
            return refuse(RosterDecision::RejectedStructural {
                reason: e.to_string(),
            });
        }
        if let Some(cur) = &self.record {
            if cur.body.run_id != record.body.run_id
                || cur.body.endpoint_id != record.body.endpoint_id
            {
                return refuse(RosterDecision::RejectedStructural {
                    reason: "publish run/endpoint does not match the slot".into(),
                });
            }
            if record.body.freshness() < cur.body.freshness() {
                return refuse(RosterDecision::RejectedStale {
                    stored_incarnation: cur.body.incarnation,
                    stored_issued_at_ms: cur.body.issued_at_ms,
                });
            }
        }
        (
            Self {
                record: Some(record.clone()),
            },
            RosterDecision::Accepted,
        )
    }
}

/// The response every roster publish returns: the structural verdict plus the slot's stored
/// record after the fold (the current record, on a refusal — what a stale publisher re-reads).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterMutationResponse {
    /// The registry's structural verdict.
    pub decision: RosterDecision,
    /// The slot's stored record after the fold.
    pub record: Option<RosterRecord>,
}

/// The typed snapshot `GET {base}/runs/:id/roster` returns: every stored record for the run, in
/// registry storage order (readers apply their own verification + freshness precedence — the
/// order carries no meaning).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterSnapshot {
    /// The stored records.
    pub entries: Vec<RosterRecord>,
}

/// Reader-side supersession: reduce verified records to the freshest one per
/// `(role, certificate base identity)` group ([`RosterRecord::group_key`]), by the
/// lexicographic freshness key. Callers pass records that already passed
/// [`RosterRecord::authorize`] — this is pure precedence, not trust.
#[must_use]
pub fn freshest_per_node(records: Vec<RosterRecord>) -> Vec<RosterRecord> {
    let mut best: std::collections::BTreeMap<(String, PeerId), RosterRecord> =
        std::collections::BTreeMap::new();
    for record in records {
        let key = record.group_key();
        match best.get(&key) {
            Some(cur) if cur.body.freshness() >= record.body.freshness() => {}
            _ => {
                best.insert(key, record);
            }
        }
    }
    best.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cert::RunKeyCertificate;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn body(sender: PeerId, incarnation: u64, issued_at_ms: u64) -> RosterRecordBody {
        RosterRecordBody {
            domain: ROSTER_RECORD_DOMAIN.to_string(),
            run_id: Hash([1; 32]),
            role: "trainer".into(),
            epoch: 0,
            incarnation,
            sender,
            module_hash: Hash([0xCC; 32]),
            endpoint_id: IrohId([0x55; 32]),
            direct_addrs: vec!["127.0.0.1:4550".into()],
            relay_url: None,
            issued_at_ms,
        }
    }

    /// A record published by `run_key(seed)` at `(incarnation, issued_at_ms)`, certified by `base`.
    fn record(base: &SigningKey, seed: u8, incarnation: u64, issued_at_ms: u64) -> RosterRecord {
        let run_key = key(seed);
        let sender = peer_id(&run_key);
        let b = body(sender, incarnation, issued_at_ms);
        let cert = RunKeyCertificate::issue(base, b.cert_scope(), sender).unwrap();
        RosterRecord::publish(&run_key, cert, b).unwrap()
    }

    #[test]
    fn a_record_round_trips_through_canonical_cbor_and_authorizes() {
        let base = key(1);
        let r = record(&base, 2, 0, 1_000);
        let bytes = crate::canonical::to_canonical_vec(&r).unwrap();
        let back: RosterRecord = crate::canonical::from_canonical_slice(&bytes).unwrap();
        assert_eq!(r, back);
        assert!(back.authorize(&[peer_id(&base)]).is_ok());
    }

    #[test]
    fn tampered_body_wrong_domain_and_undialable_are_refused_typed() {
        let base = key(1);
        // Re-address without re-signing: the signature breaks.
        let mut r = record(&base, 2, 0, 1_000);
        r.body.direct_addrs = vec!["10.0.0.9:1".into()];
        assert_eq!(r.verify_signature(), Err(RosterRecordError::BadSignature));

        let run_key = key(2);
        let mut b = body(peer_id(&run_key), 0, 1_000);
        b.domain = "daemon-vhc/seat-lease/1.0.0".into(); // a seat tag on a roster body
        assert!(matches!(
            b.validate(),
            Err(RosterRecordError::WrongDomain { .. })
        ));

        let mut b = body(peer_id(&run_key), 0, 1_000);
        b.direct_addrs.clear();
        b.relay_url = None;
        assert_eq!(b.validate(), Err(RosterRecordError::NotDialable));

        // Relay-only reachability IS dialable (the legacy hosted-relay form).
        let mut b = body(peer_id(&run_key), 0, 1_000);
        b.direct_addrs.clear();
        b.relay_url = Some("http://relay.example:3340".into());
        assert!(b.validate().is_ok());

        let mut b = body(peer_id(&run_key), 0, 1_000);
        b.issued_at_ms = 0;
        assert_eq!(b.validate(), Err(RosterRecordError::UnstampedIssue));

        let mut b = body(peer_id(&run_key), 0, 1_000);
        b.direct_addrs = (0..=MAX_ROSTER_DIRECT_ADDRS)
            .map(|i| format!("127.0.0.1:{i}"))
            .collect();
        assert!(matches!(
            b.validate(),
            Err(RosterRecordError::OverCap { .. })
        ));
    }

    #[test]
    fn publish_constructor_refuses_foreign_keys() {
        let base = key(1);
        let run_key = key(2);
        let b = body(peer_id(&run_key), 0, 1_000);
        let cert = RunKeyCertificate::issue(&base, b.cert_scope(), peer_id(&run_key)).unwrap();
        // A key that is not the body's sender cannot author the record.
        assert!(RosterRecord::publish(&key(3), cert, b).is_err());
    }

    #[test]
    fn authorize_refuses_untrusted_base_and_out_of_scope_certificates() {
        let base = key(1);
        let r = record(&base, 2, 0, 1_000);
        // Untrusted base: the same object under a different trust set is refused.
        assert_eq!(
            r.authorize(&[peer_id(&key(9))]),
            Err(RosterRecordError::UntrustedBase)
        );

        // A certificate bound to a different scope (wrong incarnation) is a typed cert refusal.
        let run_key = key(2);
        let sender = peer_id(&run_key);
        let b = body(sender, 3, 1_000);
        let mut wrong = b.cert_scope();
        wrong.instance = 4;
        let cert = RunKeyCertificate::issue(&base, wrong, sender).unwrap();
        let r2 = RosterRecord::publish(&run_key, cert, b).unwrap();
        assert!(matches!(
            r2.authorize(&[peer_id(&base)]),
            Err(RosterRecordError::Cert(_))
        ));
    }

    #[test]
    fn fold_is_a_monotonic_freshness_upsert() {
        let base = key(1);
        let slot = RosterSlot::new();

        // First publish on a virgin slot.
        let first = record(&base, 2, 1, 1_000);
        let (slot, decision) = slot.fold(&first);
        assert_eq!(decision, RosterDecision::Accepted);
        assert_eq!(slot.record.as_ref().unwrap().body.issued_at_ms, 1_000);

        // Same incarnation, later issue: the re-address republish supersedes.
        let readdressed = record(&base, 2, 1, 2_000);
        let (slot, decision) = slot.fold(&readdressed);
        assert_eq!(decision, RosterDecision::Accepted);
        assert_eq!(slot.record.as_ref().unwrap().body.issued_at_ms, 2_000);

        // The idempotent republish (equal freshness key) is accepted.
        let (slot, decision) = slot.fold(&readdressed);
        assert_eq!(decision, RosterDecision::Accepted);

        // A stale republish (earlier issue, same incarnation) is refused with the stored key.
        let stale = record(&base, 2, 1, 1_500);
        let (slot, decision) = slot.fold(&stale);
        assert_eq!(
            decision,
            RosterDecision::RejectedStale {
                stored_incarnation: 1,
                stored_issued_at_ms: 2_000
            }
        );

        // A higher incarnation supersedes even with an EARLIER wall clock (incarnation is the
        // major key — wall clock never overrides a rejoin).
        let rejoined = record(&base, 2, 2, 500);
        let (slot, decision) = slot.fold(&rejoined);
        assert_eq!(decision, RosterDecision::Accepted);
        assert_eq!(slot.record.as_ref().unwrap().body.incarnation, 2);

        // ...and the prior incarnation can never come back.
        let ghost = record(&base, 2, 1, 9_000);
        let (_, decision) = slot.fold(&ghost);
        assert!(matches!(decision, RosterDecision::RejectedStale { .. }));
    }

    #[test]
    fn fold_refuses_a_mismatched_slot_key_structurally() {
        let base = key(1);
        let first = record(&base, 2, 1, 1_000);
        let (slot, _) = RosterSlot::new().fold(&first);
        // Same endpoint key path, different run: slot-key inconsistency.
        let mut foreign = record(&base, 2, 2, 2_000);
        foreign.body.run_id = Hash([9; 32]);
        // (Signature broken by the mutation, but the fold never checks signatures — the
        // structural slot-key check is what refuses it.)
        let (_, decision) = slot.fold(&foreign);
        assert!(matches!(
            decision,
            RosterDecision::RejectedStructural { .. }
        ));
    }

    #[test]
    fn freshest_per_node_applies_incarnation_then_issue_precedence_per_group() {
        let base_a = key(1);
        let base_b = key(9);
        // Node A (base_a): incarnation 1 then a rejoin at 2 (older wall clock, still wins).
        let a_old = record(&base_a, 2, 1, 5_000);
        let a_new = record(&base_a, 3, 2, 1_000);
        // Node B (base_b): one record.
        let b_only = record(&base_b, 4, 1, 2_000);
        let reduced = freshest_per_node(vec![a_old, a_new.clone(), b_only.clone()]);
        assert_eq!(reduced.len(), 2, "one record per (role, base) group");
        assert!(reduced.contains(&a_new));
        assert!(reduced.contains(&b_only));
    }

    #[test]
    fn snapshot_and_mutation_response_round_trip_through_canonical_cbor() {
        let base = key(1);
        let r = record(&base, 2, 0, 1_000);
        let snapshot = RosterSnapshot {
            entries: vec![r.clone()],
        };
        let bytes = crate::canonical::to_canonical_vec(&snapshot).unwrap();
        let back: RosterSnapshot = crate::canonical::from_canonical_slice(&bytes).unwrap();
        assert_eq!(back, snapshot);

        for decision in [
            RosterDecision::Accepted,
            RosterDecision::RejectedStructural { reason: "r".into() },
            RosterDecision::RejectedStale {
                stored_incarnation: 2,
                stored_issued_at_ms: 7,
            },
        ] {
            let resp = RosterMutationResponse {
                decision,
                record: Some(r.clone()),
            };
            let bytes = crate::canonical::to_canonical_vec(&resp).unwrap();
            let back: RosterMutationResponse =
                crate::canonical::from_canonical_slice(&bytes).unwrap();
            assert_eq!(back, resp);
        }
    }
}
