// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **archive head record** — the product wire contract for incremental authenticated
//! journal-archive publication (architecture §4.4; runbook §3.4).
//!
//! A **journal chain** is one sealed-segment series of one on-disk journal home, scoped
//! `(run, role, base identity, chain_instance)`:
//!
//! - `run` / `role` name the envelope seat the journal records.
//! - The **base identity** is the durable node identity the publishing per-run key's
//!   certificate chains to — the stable half of the scope (per-run keys rotate with the
//!   incarnation; the base does not).
//! - `chain_instance` is the **founding incarnation** of the series: the incarnation that
//!   created segment 0 of the journal home. A live-upgrade seam (§8.1/§10.3) CONTINUES the
//!   series — later identity spans keep publishing under the founding instance, and each head
//!   names the span (`instance`/`epoch`/`module`) that sealed its segment.
//!
//! On every segment seal the publisher uploads the sealed segment bytes to the content plane
//! (the segment's BLAKE3 is its content address — the same `complete_file_blake3` the §8.2 chain
//! threads) and then publishes an [`ArchiveHeadRecord`]: the signed claim "segment `segment` of
//! this chain has content address `segment_hash`, extending `prev_hash`". Heads are small,
//! content-addressed, and offline-verifiable: the record carries the publishing per-run key's
//! certificate, so any third party holding the run's genesis-trusted base identities verifies it
//! with no registry trust at all.
//!
//! **Successor linking**: a fresh incarnation after a restart opens a NEW journal home — a new
//! chain starting at segment 0. Its first head names the predecessor chain's last published
//! attested head by content address ([`ArchiveHeadBody::predecessor`]: the BLAKE3 of that head
//! record's canonical CBOR), so a run's chains form a verifiable succession walk. `None` marks
//! the run's first chain for that `(role, base)`.
//!
//! **Conflict rejection**: within a chain, heads extend densely (`segment == stored + 1`,
//! `prev_hash == stored segment_hash`). A head at an already-stored height must be byte-identical
//! (idempotent republish after a crash between seal and acknowledge); anything else is refused
//! typed — two signed heads at one height that do not extend one another are portable fork
//! evidence (architecture §4.3). The registry-side fold ([`ArchiveChainSlot::fold`]) enforces
//! exactly this structure and NOTHING more: like the roster fold, the registry is untrusted
//! storage — authority is the verifier's judgment ([`ArchiveHeadRecord::authorize`]), never the
//! registry's.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Signature};
use crate::cert::{CertError, CertScope, RunKeyCertificate};
use crate::domains::ARCHIVE_HEAD_DOMAIN;
use crate::error::VhcProtoError;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};

/// Cap on the role label (mirrors the roster's structural discipline; a head is a small claim,
/// never a bulk carrier).
pub const MAX_ARCHIVE_ROLE_LEN: usize = 64;

/// The signed body of an archive head record. Every field is part of the signed preimage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveHeadBody {
    /// Domain-separation tag — MUST be [`ARCHIVE_HEAD_DOMAIN`].
    pub domain: String,
    /// The run's cryptographic identity: the genesis-envelope hash (ABI §8.1 `run_id`).
    pub run_id: Hash,
    /// The envelope role label whose journal this chain records.
    pub role: String,
    /// The **founding incarnation** of the segment series (the chain-scope component; see the
    /// module docs). Constant across every head of one chain.
    pub chain_instance: u64,
    /// The 0-based segment ordinal — the chain height.
    pub segment: u64,
    /// The sealed segment's content address (BLAKE3 of the complete segment file — §8.2).
    pub segment_hash: Hash,
    /// The previous segment's content address (the chain link; all-zero at segment 0).
    pub prev_hash: Hash,
    /// The number of records in the sealed segment.
    pub records: u64,
    /// The identity span that sealed this segment: the incarnation id in the segment's header
    /// (equals `chain_instance` until a live-upgrade seam advances it).
    pub instance: u64,
    /// The identity span's transition-chain epoch position.
    pub epoch: u64,
    /// The identity span's pinned module hash.
    pub module: Hash,
    /// Segment 0 of a successor chain only: the BLAKE3 content address of the predecessor
    /// chain's last published [`ArchiveHeadRecord`] (its canonical CBOR). `None` on the run's
    /// first chain for this `(role, base)` and on every segment above 0.
    #[serde(default)]
    pub predecessor: Option<Hash>,
    /// The **freshness claim**: the highest round the sealing span had COMMITTED (its own
    /// module's `round_metrics` outcome) when this segment sealed. `None` when no round had
    /// committed yet on this chain (and on heads published before the claim existed —
    /// `skip_serializing_if` keeps the `None` encoding byte-identical to the pre-claim wire
    /// form, so old signatures and predecessor content addresses stay valid).
    ///
    /// This is what a joiner's staleness judgment (`CheckpointStale`) compares against: the
    /// latest verified committed round of the coordinator lineage — signed, certificate-chained
    /// evidence, never registry metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub round: Option<u64>,
}

/// An archive head record: the signed [`ArchiveHeadBody`], the publisher per-run key's
/// [`RunKeyCertificate`], and that key's ed25519 signature over the canonical CBOR of the body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveHeadRecord {
    /// The signed chain-extension claim.
    pub body: ArchiveHeadBody,
    /// The certificate authorizing the signing per-run key for the sealing identity span
    /// (`run_id`, `epoch`, `role`, `instance`, `module`). Its `base_identity` is the chain
    /// scope's base component.
    pub certificate: RunKeyCertificate,
    /// The signing per-run key (the certificate's subject).
    pub signer: PeerId,
    /// ed25519 signature by [`ArchiveHeadRecord::signer`] over the canonical CBOR of `body`.
    pub sig: Signature,
}

/// Why an archive head record was refused by a **verifier** (a peer, an assembler, or the typed
/// layers of the publish constructor). Registry-side refusals are [`ArchiveHeadDecision`]s —
/// structural only.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArchiveHeadError {
    /// The body's domain tag is not [`ARCHIVE_HEAD_DOMAIN`].
    WrongDomain {
        /// The tag actually carried.
        got: String,
    },
    /// The role label is empty or over the structural cap.
    BadRole,
    /// Segment 0 must extend the all-zero genesis link; a later segment must not.
    BadGenesisLink,
    /// A predecessor link on a segment above 0 (succession is claimed once, at the chain's
    /// founding head).
    MisplacedPredecessor,
    /// The sealing span's incarnation is below the founding instance (spans only advance).
    SpanBelowFounding,
    /// The signer per-run key's signature over the body does not verify.
    BadSignature,
    /// The embedded certificate does not authorize the signer for the sealing span's scope.
    Cert(CertError),
    /// The certificate's base identity is not one the run's genesis/Authority names.
    UntrustedBase,
}

impl core::fmt::Display for ArchiveHeadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongDomain { got } => {
                write!(
                    f,
                    "archive head domain `{got}` is not `{ARCHIVE_HEAD_DOMAIN}`"
                )
            }
            Self::BadRole => write!(f, "archive head role label is empty or over the cap"),
            Self::BadGenesisLink => write!(
                f,
                "archive head genesis link violated (segment 0 extends the all-zero hash; \
                 later segments never do)"
            ),
            Self::MisplacedPredecessor => {
                write!(f, "archive head carries a predecessor link above segment 0")
            }
            Self::SpanBelowFounding => write!(
                f,
                "archive head sealing span precedes the chain's founding instance"
            ),
            Self::BadSignature => write!(f, "archive head signature does not verify to the signer"),
            Self::Cert(e) => write!(f, "archive head certificate refused: {e}"),
            Self::UntrustedBase => {
                write!(
                    f,
                    "archive head certificate base identity is not genesis-trusted"
                )
            }
        }
    }
}

impl std::error::Error for ArchiveHeadError {}

impl From<CertError> for ArchiveHeadError {
    fn from(e: CertError) -> Self {
        Self::Cert(e)
    }
}

impl From<ArchiveHeadError> for VhcProtoError {
    fn from(e: ArchiveHeadError) -> Self {
        VhcProtoError::Validation(e.to_string())
    }
}

impl ArchiveHeadBody {
    /// Structural validation — the invariants BOTH sides enforce (the registry before folding,
    /// every verifier before trusting): domain tag, role cap, the genesis-link rule, the
    /// predecessor placement rule, and span monotonicity. Signature-free by design.
    ///
    /// # Errors
    /// The applicable [`ArchiveHeadError`].
    pub fn validate(&self) -> Result<(), ArchiveHeadError> {
        if self.domain != ARCHIVE_HEAD_DOMAIN {
            return Err(ArchiveHeadError::WrongDomain {
                got: self.domain.clone(),
            });
        }
        if self.role.is_empty() || self.role.len() > MAX_ARCHIVE_ROLE_LEN {
            return Err(ArchiveHeadError::BadRole);
        }
        let genesis = self.prev_hash == Hash([0; 32]);
        if (self.segment == 0) != genesis {
            return Err(ArchiveHeadError::BadGenesisLink);
        }
        if self.predecessor.is_some() && self.segment != 0 {
            return Err(ArchiveHeadError::MisplacedPredecessor);
        }
        if self.instance < self.chain_instance {
            return Err(ArchiveHeadError::SpanBelowFounding);
        }
        Ok(())
    }

    /// The sealing span's execution-identity scope — what the embedded certificate must bind.
    #[must_use]
    pub fn cert_scope(&self) -> CertScope {
        CertScope {
            run_id: self.run_id,
            epoch: self.epoch,
            role: self.role.clone(),
            instance: self.instance,
            module_hash: self.module,
        }
    }
}

impl ArchiveHeadRecord {
    /// Author a record: validate the body structurally, require `run_key` to be the
    /// certificate's subject, and sign the canonical CBOR of the body with the per-run key.
    ///
    /// # Errors
    /// A structural violation, a signer/key mismatch, or a signing failure.
    pub fn publish(
        run_key: &SigningKey,
        certificate: RunKeyCertificate,
        body: ArchiveHeadBody,
    ) -> Result<Self, VhcProtoError> {
        body.validate()?;
        let signer = peer_id(run_key);
        if certificate.body.run_key != signer {
            return Err(VhcProtoError::Validation(
                "archive head signer is not the certificate's per-run key".into(),
            ));
        }
        let sig = sign_canonical(run_key, &body)?;
        Ok(Self {
            body,
            certificate,
            signer,
            sig,
        })
    }

    /// Verify structure + the signer's self-signature over the body. This is the registry-free
    /// half of acceptance; authority is [`ArchiveHeadRecord::authorize`].
    ///
    /// # Errors
    /// The applicable [`ArchiveHeadError`].
    pub fn verify_signature(&self) -> Result<(), ArchiveHeadError> {
        self.body.validate()?;
        verify_canonical(&self.signer, &self.sig, &self.body)
            .map_err(|_| ArchiveHeadError::BadSignature)
    }

    /// The full verifier-side acceptance: structure, self-signature, the certificate chain to a
    /// **genesis-trusted** base identity, and the certificate's binding over exactly the sealing
    /// span's scope and this signer.
    ///
    /// # Errors
    /// The applicable [`ArchiveHeadError`]. The registry MUST NOT run this check — authority is
    /// never the registry's judgment.
    pub fn authorize(&self, trusted_bases: &[PeerId]) -> Result<(), ArchiveHeadError> {
        self.verify_signature()?;
        if !trusted_bases.contains(&self.certificate.base_identity) {
            return Err(ArchiveHeadError::UntrustedBase);
        }
        self.certificate
            .authorizes_sender(&self.body.cert_scope(), &self.signer)?;
        Ok(())
    }

    /// The chain scope this head publishes under: `(role, base identity, chain_instance)`
    /// within its run — the registry's slot key and the reader's grouping key.
    #[must_use]
    pub fn chain_key(&self) -> (String, PeerId, u64) {
        (
            self.body.role.clone(),
            self.certificate.base_identity,
            self.body.chain_instance,
        )
    }

    /// The record's content address: the BLAKE3 of its canonical CBOR — what a successor
    /// chain's `predecessor` field names.
    ///
    /// # Errors
    /// [`VhcProtoError`] on a canonical-encode failure.
    pub fn content_address(&self) -> Result<Hash, VhcProtoError> {
        Ok(crate::blake3_hash(&crate::to_canonical_vec(self)?))
    }
}

// -- the registry-side slot + the normative structural fold ---------------------------------------

/// The registry's structural verdict on one archive-head publish. Purely structural — an
/// `Accepted` says nothing about authority (verifiers judge that; the registry is untrusted
/// storage).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveHeadDecision {
    /// Stored; the chain slot now extends to this head.
    Accepted,
    /// Byte-identical to the head already stored at this height — the idempotent republish
    /// (a crash between seal and acknowledge re-sends).
    AlreadyStored,
    /// The record violates a structural invariant (domain, caps, link rules, slot consistency).
    RejectedStructural {
        /// A human-readable reason (diagnostic only; never authority).
        reason: String,
    },
    /// The head does not extend the stored chain: wrong next ordinal, a broken `prev_hash`
    /// link, or a NON-identical head at a stored height. The refusal carries the stored tip so
    /// the publisher (or any auditor) can extract the two-head fork evidence.
    RejectedNonExtending {
        /// The stored chain tip's segment ordinal.
        stored_segment: u64,
        /// The stored chain tip's segment content address.
        stored_segment_hash: Hash,
    },
}

/// One chain's registry slot: the accepted heads, dense from segment 0. The slot key within a
/// run is [`ArchiveHeadRecord::chain_key`]; the run itself keys the outer map.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveChainSlot {
    /// The accepted heads, ascending by segment ordinal (dense: `heads[i].body.segment == i`).
    pub heads: Vec<ArchiveHeadRecord>,
}

impl ArchiveChainSlot {
    /// An empty chain slot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The normative **structural fold** a conforming registry applies to one publish:
    ///
    /// 1. Structural validation of the body ([`ArchiveHeadBody::validate`]).
    /// 2. A height at or below the stored tip must be **byte-identical** to the stored head
    ///    (canonical CBOR equality) — the idempotent republish; anything else is
    ///    [`ArchiveHeadDecision::RejectedNonExtending`] (fork evidence surface).
    /// 3. A new height must be exactly `stored tip + 1` (0 on an empty slot) with `prev_hash`
    ///    equal to the stored tip's `segment_hash` — dense, linked extension only.
    /// 4. The chain scope fields (`role`, `chain_instance`, `run_id`) must match the slot.
    ///
    /// Signature/authority checks are deliberately absent (untrusted storage).
    pub fn fold(&mut self, record: ArchiveHeadRecord) -> ArchiveHeadDecision {
        if let Err(e) = record.body.validate() {
            return ArchiveHeadDecision::RejectedStructural {
                reason: e.to_string(),
            };
        }
        if let Some(first) = self.heads.first() {
            if record.body.run_id != first.body.run_id
                || record.body.role != first.body.role
                || record.body.chain_instance != first.body.chain_instance
            {
                return ArchiveHeadDecision::RejectedStructural {
                    reason: "chain scope fields do not match the slot".into(),
                };
            }
        }
        let tip = self.heads.last();
        let next = self.heads.len() as u64;
        let seg = record.body.segment;
        if seg < next {
            // A stored height: idempotent iff byte-identical.
            let stored = &self.heads[usize::try_from(seg).unwrap_or(usize::MAX)];
            if *stored == record {
                return ArchiveHeadDecision::AlreadyStored;
            }
            return ArchiveHeadDecision::RejectedNonExtending {
                stored_segment: stored.body.segment,
                stored_segment_hash: stored.body.segment_hash,
            };
        }
        if seg != next || tip.is_some_and(|t| record.body.prev_hash != t.body.segment_hash) {
            return match tip {
                Some(t) => ArchiveHeadDecision::RejectedNonExtending {
                    stored_segment: t.body.segment,
                    stored_segment_hash: t.body.segment_hash,
                },
                None => ArchiveHeadDecision::RejectedStructural {
                    reason: format!("first head must be segment 0, got {seg}"),
                },
            };
        }
        self.heads.push(record);
        ArchiveHeadDecision::Accepted
    }
}

// -- the reader-side chain verification (shared by every verifier) --------------------------------
//
// The assembler (`daemon-vhc-observe`), the replay reader (`xtask vhc-replay`), the node's join
// transaction, and the worker's reconstruction executor all make the SAME judgment over a
// published head snapshot: authorize every record against the genesis-trusted bases, group by
// chain scope, re-fold each chain structurally, and (for recovery) order the role's chains
// founding-first by succession links. That judgment lives here — in the wire-contract crate —
// so production hosts never link the oracle tooling to make it (ABI §12.5 [OWN-3]).

/// One verified chain: its ordered head records and their content addresses. Produced by
/// [`verify_chains`]; consumed by the assembler, the replay reader, the node's recovery
/// resolution, and the worker's reconstruction executor.
#[derive(Clone, Debug)]
pub struct VerifiedChain {
    /// The chain's role label.
    pub role: String,
    /// The chain scope's base identity (the certificate issuer).
    pub base: PeerId,
    /// The chain-scope instance (the founding incarnation).
    pub chain_instance: u64,
    /// The chain's head records, segment order (dense from 0 — the fold enforced it).
    pub heads: Vec<ArchiveHeadRecord>,
    /// The founding head's `predecessor` (a prior chain's terminal head content address).
    pub predecessor: Option<Hash>,
    /// The terminal (highest-segment) head's own content address.
    pub terminal_address: Hash,
}

/// Why a head snapshot failed reader-side verification (typed; a snapshot that does not verify
/// is never partially trusted).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChainVerifyError {
    /// A published head failed reader-side authorization (ABI §8.8 \[AR-4\]) or names a
    /// different run.
    Unauthorized {
        /// The head's role label.
        role: String,
        /// The head's chain height.
        segment: u64,
        /// The typed refusal.
        detail: String,
    },
    /// A chain's snapshot does not re-fold densely (corrupt or forked archive slots).
    ChainFold {
        /// The chain key `role/base-hex/instance`.
        key: String,
        /// The fold refusal.
        detail: String,
    },
    /// The recovery lineage cannot be ordered (missing founding chain, broken or ambiguous
    /// succession links).
    Lineage(String),
    /// A head record does not re-encode canonically (content addressing failed).
    Codec(String),
}

impl core::fmt::Display for ChainVerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unauthorized {
                role,
                segment,
                detail,
            } => write!(f, "head (role {role}, segment {segment}) refused: {detail}"),
            Self::ChainFold { key, detail } => write!(f, "chain {key} does not fold: {detail}"),
            Self::Lineage(detail) => write!(f, "chain lineage: {detail}"),
            Self::Codec(detail) => write!(f, "record encode: {detail}"),
        }
    }
}

impl std::error::Error for ChainVerifyError {}

fn chain_key_str(role: &str, base: &PeerId, instance: u64) -> String {
    format!("{role}/{}/{instance}", base.to_hex())
}

/// Authorize every head record against the genesis-trusted bases (ABI §8.8 \[AR-4\]), group by
/// chain scope, and re-fold every chain through the normative structural fold — the reader-side
/// verification every consumer of a head snapshot runs.
///
/// # Errors
/// [`ChainVerifyError::Unauthorized`] / [`ChainVerifyError::ChainFold`].
pub fn verify_chains(
    run_id: &Hash,
    trusted: &[PeerId],
    records: Vec<ArchiveHeadRecord>,
) -> Result<Vec<VerifiedChain>, ChainVerifyError> {
    let mut by_chain: std::collections::BTreeMap<(String, PeerId, u64), Vec<ArchiveHeadRecord>> =
        std::collections::BTreeMap::new();
    for record in records {
        if record.body.run_id != *run_id {
            return Err(ChainVerifyError::Unauthorized {
                role: record.body.role.clone(),
                segment: record.body.segment,
                detail: "head names a different run".into(),
            });
        }
        record
            .authorize(trusted)
            .map_err(|e| ChainVerifyError::Unauthorized {
                role: record.body.role.clone(),
                segment: record.body.segment,
                detail: e.to_string(),
            })?;
        by_chain.entry(record.chain_key()).or_default().push(record);
    }

    let mut chains: Vec<VerifiedChain> = Vec::new();
    for ((role, base, instance), mut heads) in by_chain {
        heads.sort_by_key(|r| r.body.segment);
        let key = chain_key_str(&role, &base, instance);
        let mut slot = ArchiveChainSlot::new();
        for head in &heads {
            match slot.fold(head.clone()) {
                ArchiveHeadDecision::Accepted | ArchiveHeadDecision::AlreadyStored => {}
                refused => {
                    return Err(ChainVerifyError::ChainFold {
                        key,
                        detail: format!("{refused:?}"),
                    })
                }
            }
        }
        let predecessor = heads.first().and_then(|h| h.body.predecessor);
        let terminal_address = heads
            .last()
            .ok_or_else(|| ChainVerifyError::ChainFold {
                key: key.clone(),
                detail: "empty chain slot".into(),
            })?
            .content_address()
            .map_err(|e| ChainVerifyError::Codec(e.to_string()))?;
        chains.push(VerifiedChain {
            role,
            base,
            chain_instance: instance,
            heads,
            predecessor,
            terminal_address,
        });
    }
    Ok(chains)
}

/// Order one role's chains founding-first by their succession links (ABI §8.8 \[AR-1\]: a
/// successor chain's founding head names the predecessor chain's terminal head by content
/// address). The recovery walk: the coordinator lineage for reconstruction and replay, a
/// trainer's own lineage for its journal custody.
///
/// # Errors
/// [`ChainVerifyError::Lineage`] on a missing founding chain or broken/ambiguous links.
pub fn coordinator_lineage<'a>(
    chains: &'a [VerifiedChain],
    role: &str,
) -> Result<Vec<&'a VerifiedChain>, ChainVerifyError> {
    let of_role: Vec<&VerifiedChain> = chains.iter().filter(|c| c.role == role).collect();
    if of_role.is_empty() {
        return Err(ChainVerifyError::Lineage(format!(
            "no {role}-role chains in the snapshot"
        )));
    }
    let founding: Vec<&&VerifiedChain> =
        of_role.iter().filter(|c| c.predecessor.is_none()).collect();
    let [start] = founding.as_slice() else {
        return Err(ChainVerifyError::Lineage(format!(
            "expected exactly one founding {role} chain, found {}",
            founding.len()
        )));
    };
    let mut by_predecessor: std::collections::BTreeMap<Hash, &VerifiedChain> =
        std::collections::BTreeMap::new();
    for chain in &of_role {
        if let Some(pred) = chain.predecessor {
            if by_predecessor.insert(pred, chain).is_some() {
                return Err(ChainVerifyError::Lineage(format!(
                    "two {role} chains name the same predecessor {}",
                    pred.to_hex()
                )));
            }
        }
    }
    let mut lineage: Vec<&VerifiedChain> = vec![start];
    let mut cursor = *start;
    while let Some(next) = by_predecessor.get(&cursor.terminal_address) {
        lineage.push(next);
        cursor = next;
    }
    if lineage.len() != of_role.len() {
        return Err(ChainVerifyError::Lineage(format!(
            "{} {role} chain(s) do not link into the lineage",
            of_role.len() - lineage.len()
        )));
    }
    Ok(lineage)
}

/// The lineage's freshness statement: the highest [`ArchiveHeadBody::round`] claim across the
/// verified chains, or `None` when no head carries one. This — signed, certificate-chained
/// evidence from the run's own archive — is the head estimate a joiner's staleness judgment
/// uses, never registry metadata.
#[must_use]
pub fn latest_round_claim(lineage: &[&VerifiedChain]) -> Option<u64> {
    lineage
        .iter()
        .flat_map(|c| c.heads.iter())
        .filter_map(|h| h.body.round)
        .max()
}

/// The genesis-trusted base identities a reader authorizes archive heads against: the envelope's
/// `identities.coordinator_set` plus `identities.coordinator` (ABI §8.8 \[AR-4\]).
#[must_use]
pub fn envelope_trusted_bases(envelope: &crate::genesis::GenesisEnvelope) -> Vec<PeerId> {
    let mut trusted: Vec<PeerId> = envelope.identities.coordinator_set.clone();
    if let Some(coord) = envelope.identities.coordinator {
        if !trusted.contains(&coord) {
            trusted.push(coord);
        }
    }
    trusted
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cert::RunKeyCertificate;

    fn base_key() -> SigningKey {
        SigningKey::from_bytes(&[0xB0; 32])
    }

    fn run_key() -> SigningKey {
        SigningKey::from_bytes(&[0x4A; 32])
    }

    fn cert_for(instance: u64, module: Hash) -> RunKeyCertificate {
        RunKeyCertificate::issue(
            &base_key(),
            CertScope {
                run_id: Hash([0x1D; 32]),
                epoch: 0,
                role: "coordinator".into(),
                instance,
                module_hash: module,
            },
            peer_id(&run_key()),
        )
        .expect("cert")
    }

    fn body(segment: u64, prev: Hash, seg_hash: Hash) -> ArchiveHeadBody {
        ArchiveHeadBody {
            domain: ARCHIVE_HEAD_DOMAIN.into(),
            run_id: Hash([0x1D; 32]),
            role: "coordinator".into(),
            chain_instance: 1,
            segment,
            segment_hash: seg_hash,
            prev_hash: prev,
            records: 10,
            instance: 1,
            epoch: 0,
            module: Hash([0x2A; 32]),
            predecessor: None,
            round: None,
        }
    }

    fn head(segment: u64, prev: Hash, seg_hash: Hash) -> ArchiveHeadRecord {
        ArchiveHeadRecord::publish(
            &run_key(),
            cert_for(1, Hash([0x2A; 32])),
            body(segment, prev, seg_hash),
        )
        .expect("publish")
    }

    /// The full verifier path holds end to end: a published head authorizes against the issuing
    /// base and refuses an impostor trust set.
    #[test]
    fn a_published_head_authorizes_against_the_genesis_trusted_base() {
        let h = head(0, Hash([0; 32]), Hash([0xAA; 32]));
        h.authorize(&[peer_id(&base_key())]).expect("authorized");
        assert_eq!(
            h.authorize(&[peer_id(&SigningKey::from_bytes(&[9; 32]))]),
            Err(ArchiveHeadError::UntrustedBase)
        );
    }

    /// Tampering with the signed claim after publication is a signature refusal — the claim is
    /// the preimage, not the record wrapper.
    #[test]
    fn a_tampered_body_fails_the_self_signature() {
        let mut h = head(0, Hash([0; 32]), Hash([0xAA; 32]));
        h.body.segment_hash = Hash([0xBB; 32]);
        assert_eq!(h.verify_signature(), Err(ArchiveHeadError::BadSignature));
    }

    /// The structural rules that make a head self-describing: the genesis link is exactly the
    /// segment-0 shape, succession is claimed only at the founding head, and a sealing span
    /// never precedes the founding instance.
    #[test]
    fn structural_validation_enforces_link_and_span_rules() {
        // Segment 1 claiming the genesis link.
        let mut b = body(1, Hash([0; 32]), Hash([0xAA; 32]));
        assert_eq!(b.validate(), Err(ArchiveHeadError::BadGenesisLink));
        // Segment 0 claiming a non-genesis link.
        b = body(0, Hash([0x01; 32]), Hash([0xAA; 32]));
        assert_eq!(b.validate(), Err(ArchiveHeadError::BadGenesisLink));
        // A predecessor link above segment 0.
        b = body(1, Hash([0x01; 32]), Hash([0xAA; 32]));
        b.predecessor = Some(Hash([0xCC; 32]));
        assert_eq!(b.validate(), Err(ArchiveHeadError::MisplacedPredecessor));
        // A sealing span below the founding instance.
        b = body(0, Hash([0; 32]), Hash([0xAA; 32]));
        b.instance = 0;
        assert_eq!(b.validate(), Err(ArchiveHeadError::SpanBelowFounding));
    }

    /// The registry fold accepts exactly the dense, linked extension; re-publishing a stored
    /// head is idempotent; a conflicting head at a stored height is refused with the stored tip
    /// (the fork-evidence surface); a gap is refused.
    #[test]
    fn the_fold_accepts_dense_linked_extension_and_refuses_forks_and_gaps() {
        let mut slot = ArchiveChainSlot::new();
        let h0 = head(0, Hash([0; 32]), Hash([0xA0; 32]));
        let h1 = head(1, Hash([0xA0; 32]), Hash([0xA1; 32]));

        assert_eq!(slot.fold(h0.clone()), ArchiveHeadDecision::Accepted);
        assert_eq!(slot.fold(h1.clone()), ArchiveHeadDecision::Accepted);
        // Idempotent republish.
        assert_eq!(slot.fold(h1.clone()), ArchiveHeadDecision::AlreadyStored);
        // A conflicting head at a stored height: fork evidence, typed.
        let fork = head(1, Hash([0xA0; 32]), Hash([0xFF; 32]));
        assert_eq!(
            slot.fold(fork),
            ArchiveHeadDecision::RejectedNonExtending {
                stored_segment: 1,
                stored_segment_hash: Hash([0xA1; 32]),
            }
        );
        // A gap (segment 3 after 1): refused with the stored tip.
        let gap = head(3, Hash([0xA1; 32]), Hash([0xA3; 32]));
        assert_eq!(
            slot.fold(gap),
            ArchiveHeadDecision::RejectedNonExtending {
                stored_segment: 1,
                stored_segment_hash: Hash([0xA1; 32]),
            }
        );
        // A broken prev link at the right ordinal: refused the same way.
        let broken = head(2, Hash([0xEE; 32]), Hash([0xA2; 32]));
        assert_eq!(
            slot.fold(broken),
            ArchiveHeadDecision::RejectedNonExtending {
                stored_segment: 1,
                stored_segment_hash: Hash([0xA1; 32]),
            }
        );
        // The correct extension still lands after the refusals.
        let h2 = head(2, Hash([0xA1; 32]), Hash([0xA2; 32]));
        assert_eq!(slot.fold(h2), ArchiveHeadDecision::Accepted);
        assert_eq!(slot.heads.len(), 3);
    }

    /// An empty slot accepts only segment 0 (a chain publishes from its founding head; a
    /// mid-chain first publish would leave an unverifiable prefix).
    #[test]
    fn an_empty_slot_accepts_only_the_founding_head() {
        let mut slot = ArchiveChainSlot::new();
        let h1 = head(1, Hash([0xA0; 32]), Hash([0xA1; 32]));
        assert!(matches!(
            slot.fold(h1),
            ArchiveHeadDecision::RejectedStructural { .. }
        ));
    }
}
