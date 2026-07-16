// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The record archive (architecture §4.4; refactor §8/D2).
//!
//! The coordinator module's full state is a deterministic function of its journal — small, signed,
//! hash-chained control frames. The **record archive** publishes the A1 segmented journal's sealed
//! segments ([`super::segment`]) as **content-addressed** blobs (the content address is the
//! segment's BLAKE3 — the same `complete_file_blake3` the §8.2 chain already threads), each covered
//! by an **attested chain head** ([`AttestedHead`]), with declared **replication**, **retention**,
//! and **GC keyed to checkpoint attestation**. Consequences (architecture §4.4):
//!
//! - **State reconstruction is deterministic**: any node holding the archived segments rebuilds the
//!   coordinator state bit-exactly — a standby is a late joiner of the coordinator role.
//! - **Every decision is offline re-verifiable**: the segments are the substrate the D2 consensus
//!   replay tier re-verifies digests from (architecture §3.6).
//! - **Fork detection**: honest peers gossip attested chain heads; **two heads that do not extend
//!   one another are self-contained, third-party-verifiable evidence** ([`ForkEvidence`],
//!   architecture §4.3/§10) — the equivocation drill's portable output.
//!
//! ## The `Authority` seam (reconciled onto D1's contract — D2 sitting 3)
//!
//! "Is this chain head authoritative?" is an `Authority` question, answered by D1's contract: the
//! archive holds the run's typed [`AuthorityConfig`] (decoded from the genesis envelope's opaque
//! `authority` section) and judges every head through [`AuthorityConfig::authorize`] over the
//! head's canonical preimage + presented [`RecordSig`]s — `SingleKey` (one coordinator signature)
//! and `ThresholdKeys` (an `m`-of-`n` quorum with typed sub-quorum/duplicate/unknown-signer
//! refusals) both work unchanged through this one seam. The pre-D1 identity-comparison stub is
//! gone; the content-addressing, replication, retention, and fork-comparison mechanism is
//! untouched (the host contributes mechanism a guest cannot fake — content-hash verification and
//! the journal — architecture §4.2).

use std::collections::BTreeMap;

use daemon_vhc_proto::{sign_canonical, to_canonical_vec, Hash, SigningKey};
use daemon_vhc_sdk_consensus::{AuthorityConfig, RecordSig};
use serde::{Deserialize, Serialize};

use super::segment::{scan_bytes, SegmentHeader};
use super::JournalError;

/// A chain head — the gossiped, third-party-verifiable claim "segment `segment` of this run's
/// coordinator journal has content address `segment_hash`, extending `prev_hash`" (architecture
/// §4.3/§4.4). Every field of the execution-identity scope travels with it, so an equivocation
/// comparison needs nothing outside two heads (§4.3, §12.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainHead {
    /// The genesis/frozen-envelope run id (§8.1).
    pub run_id: Hash,
    /// The transition-chain epoch position.
    pub epoch: u64,
    /// The envelope-level role label (`coordinator` for the coordinator journal).
    pub role: String,
    /// The never-reused durable role-instance incarnation id (§8.1).
    pub instance: u64,
    /// The pinned coordinator module hash.
    pub module: Hash,
    /// The 0-based segment ordinal — the chain height.
    pub segment: u64,
    /// The sealed segment's content address (BLAKE3 of the complete segment file — §8.2).
    pub segment_hash: Hash,
    /// The previous segment's content address (the chain link; all-zero at genesis).
    pub prev_hash: Hash,
    /// The number of records in the segment.
    pub records: u64,
}

/// A chain head carrying the signatures that make it authoritative under the run's declared
/// `Authority` (D1's contract, architecture §4.2): one [`RecordSig`] under `SingleKey`, an
/// `m`-of-`n` quorum under `ThresholdKeys`. The preimage every signature covers is the canonical
/// CBOR of [`ChainHead`]; whether the presented set suffices is judged by
/// [`AuthorityConfig::authorize`] — never by an identity comparison here (the D2 sitting-3
/// reconciliation of the old labelled `SingleKey` seam).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttestedHead {
    /// The head claim (the signed preimage's value).
    pub body: ChainHead,
    /// The presented signatures over the head's canonical bytes.
    pub sigs: Vec<RecordSig>,
}

impl AttestedHead {
    /// Attest a head under one signing key (the `SingleKey` custody path — also one member's
    /// share of a threshold quorum).
    ///
    /// # Errors
    /// [`JournalError::Codec`] if the head cannot be canonically encoded/signed.
    pub fn single(key: &SigningKey, body: ChainHead) -> Result<Self, JournalError> {
        Self::attest(std::slice::from_ref(key), body)
    }

    /// Attest a head under several signing keys (a `ThresholdKeys` quorum: each member signs the
    /// same canonical head preimage).
    ///
    /// # Errors
    /// [`JournalError::Codec`] if the head cannot be canonically encoded/signed.
    pub fn attest(keys: &[SigningKey], body: ChainHead) -> Result<Self, JournalError> {
        let mut sigs = Vec::with_capacity(keys.len());
        for key in keys {
            let sig = sign_canonical(key, &body)
                .map_err(|e| JournalError::Codec(format!("head sign: {e}")))?;
            sigs.push(RecordSig {
                signer: daemon_vhc_proto::peer_id(key),
                sig,
            });
        }
        Ok(Self { body, sigs })
    }

    /// The canonical signed preimage (what every [`RecordSig`] covers).
    ///
    /// # Errors
    /// [`JournalError::Codec`] on a canonical-encode failure.
    pub fn preimage(&self) -> Result<Vec<u8>, JournalError> {
        to_canonical_vec(&self.body).map_err(|e| JournalError::Codec(format!("head encode: {e}")))
    }
}

/// Portable, self-contained fork evidence (architecture §4.3/§10) — verifiable by any third party
/// from the two attested heads plus the run's declared `AuthorityConfig` alone.
#[derive(Clone, Debug, PartialEq)]
pub enum ForkEvidence {
    /// Two authoritative heads at the **same** chain height `(run_id, epoch, role, instance,
    /// segment)` with **different** `segment_hash`: the authority served divergent histories at one
    /// height. This is the sharpest form — a coordinator that eclipsed two peer sets and served
    /// each an internally-consistent history is exposed the moment the two heads meet (§10).
    DivergentHead {
        /// The accepted head at this height.
        a: Box<AttestedHead>,
        /// The conflicting head at the same height.
        b: Box<AttestedHead>,
    },
    /// A head at segment `N+1` whose `prev_hash` does not match the accepted segment `N`'s content
    /// address: its chain does not extend the accepted one (a non-extending head, §10).
    NonExtending {
        /// The accepted head at segment `N`.
        accepted: Box<AttestedHead>,
        /// The head at `N+1` that fails to extend it.
        conflicting: Box<AttestedHead>,
    },
}

/// The archive's declared replication policy (architecture §4.4).
#[derive(Clone, Copy, Debug)]
pub struct ReplicationPolicy {
    /// The target number of replicas across standbys + storage a segment is durable at.
    pub factor: u32,
}

impl Default for ReplicationPolicy {
    fn default() -> Self {
        Self { factor: 2 }
    }
}

/// The archive's declared retention / GC policy (architecture §4.4).
#[derive(Clone, Copy, Debug)]
pub struct RetentionPolicy {
    /// How many trailing segments below the head to keep unconditionally (the retention horizon).
    pub horizon_segments: u64,
    /// Whether GC is gated on checkpoint attestation: records inside the latest attested
    /// checkpoint's catch-up window are never collected (architecture §4.4).
    pub gc_keyed_to_attestation: bool,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            horizon_segments: 4,
            gc_keyed_to_attestation: true,
        }
    }
}

/// Errors surfaced by the record archive.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArchiveError {
    /// A published segment did not end with a valid `seal` record (only sealed, immutable segments
    /// are archivable — §8.2).
    #[error("segment is not sealed (only sealed segments are archivable)")]
    NotSealed,
    /// A chain head failed the run's declared `Authority` (D1's typed [`AuthError`]
    /// (`daemon_vhc_sdk_consensus::AuthError`): wrong/unknown/duplicate signer, bad signature,
    /// sub-quorum, malformed).
    #[error("chain head is not authoritative ({0})")]
    Unauthoritative(String),
    /// The underlying journal scan failed (bad header / broken chain).
    #[error("journal error: {0}")]
    Journal(String),
}

impl From<JournalError> for ArchiveError {
    fn from(e: JournalError) -> Self {
        Self::Journal(e.to_string())
    }
}

/// A stored, content-addressed sealed segment.
#[derive(Clone, Debug)]
struct StoredSegment {
    header: SegmentHeader,
    bytes: Vec<u8>,
    replicas: u32,
}

/// The record archive: a content-addressed store of signed, hash-chained sealed journal segments
/// with declared replication, retention, and attestation-keyed GC, plus fork detection over
/// gossiped attested heads (architecture §4.4).
pub struct RecordArchive {
    /// The run's declared trust topology (D1's typed `AuthorityConfig`, decoded from the genesis
    /// envelope's opaque `authority` section): every head is judged through
    /// [`AuthorityConfig::authorize`].
    authority: AuthorityConfig,
    replication: ReplicationPolicy,
    retention: RetentionPolicy,
    /// segment content address -> stored segment.
    segments: BTreeMap<Hash, StoredSegment>,
    /// segment ordinal -> the first accepted authoritative head at that height.
    accepted: BTreeMap<u64, AttestedHead>,
    /// The latest checkpoint-attested segment ordinal (the GC floor when attestation-keyed).
    attested_segment: Option<u64>,
}

impl RecordArchive {
    /// A fresh archive for a run whose trust topology is `authority` (the D1 contract: `SingleKey`
    /// or `ThresholdKeys`, plus the declared records channel).
    #[must_use]
    pub fn new(
        authority: AuthorityConfig,
        replication: ReplicationPolicy,
        retention: RetentionPolicy,
    ) -> Self {
        Self {
            authority,
            replication,
            retention,
            segments: BTreeMap::new(),
            accepted: BTreeMap::new(),
            attested_segment: None,
        }
    }

    /// The run's declared `AuthorityConfig` (what heads are judged through).
    #[must_use]
    pub fn authority(&self) -> &AuthorityConfig {
        &self.authority
    }

    /// Whether a head carries sufficient authority under the run's declared topology — the
    /// reconciled D1 seam: the head's canonical preimage + presented [`RecordSig`]s go through
    /// [`AuthorityConfig::authorize`] (`SingleKey`: the one coordinator signature; `ThresholdKeys`:
    /// an `m`-of-`n` quorum). Identity comparison no longer lives here.
    #[must_use]
    pub fn head_is_authoritative(&self, head: &AttestedHead) -> bool {
        self.head_authorize(head).is_ok()
    }

    /// The typed form of [`RecordArchive::head_is_authoritative`]: D1's `AuthError` on refusal
    /// (the adversarial suite asserts *which* guard fired).
    ///
    /// # Errors
    /// The topology's `AuthError` (wrong/unknown/duplicate signer, bad signature, sub-quorum).
    pub fn head_authorize(
        &self,
        head: &AttestedHead,
    ) -> Result<(), daemon_vhc_sdk_consensus::AuthError> {
        let preimage =
            head.preimage()
                .map_err(|e| daemon_vhc_sdk_consensus::AuthError::Config {
                    reason: e.to_string(),
                })?;
        self.authority.authorize(&preimage, &head.sigs).map(|_| ())
    }

    /// Publish a sealed segment's complete file bytes content-addressed (architecture §4.4). The
    /// content address is the segment's BLAKE3 (== the §8.2 `complete_file_blake3`), so a tampered
    /// segment cannot masquerade under another's address. Idempotent: re-publishing the same bytes
    /// bumps the replica count toward [`ReplicationPolicy::factor`].
    ///
    /// # Errors
    /// [`ArchiveError::NotSealed`] if the segment is not cleanly sealed; [`ArchiveError::Journal`]
    /// if it does not scan.
    pub fn publish_segment(&mut self, bytes: Vec<u8>) -> Result<Hash, ArchiveError> {
        let scan = scan_bytes(&bytes)?;
        if !scan.sealed {
            return Err(ArchiveError::NotSealed);
        }
        let address = Hash(scan.complete_file_blake3);
        self.segments
            .entry(address)
            .and_modify(|s| s.replicas = s.replicas.saturating_add(1).min(self.replication.factor))
            .or_insert(StoredSegment {
                header: scan.header,
                bytes,
                replicas: 1,
            });
        Ok(address)
    }

    /// Record an additional replica of an already-published segment (a standby/storage confirmed a
    /// copy). Returns the replica count, capped at the policy factor.
    #[must_use]
    pub fn replicate(&mut self, address: &Hash) -> u32 {
        if let Some(s) = self.segments.get_mut(address) {
            s.replicas = s.replicas.saturating_add(1).min(self.replication.factor);
            s.replicas
        } else {
            0
        }
    }

    /// Whether a segment has reached the declared replication factor (durable).
    #[must_use]
    pub fn is_durable(&self, address: &Hash) -> bool {
        self.segments
            .get(address)
            .is_some_and(|s| s.replicas >= self.replication.factor)
    }

    /// Fetch a segment's bytes by content address (the content-addressed get; hash-verified on
    /// publish, so the returned bytes hash to `address`).
    #[must_use]
    pub fn fetch(&self, address: &Hash) -> Option<&[u8]> {
        self.segments.get(address).map(|s| s.bytes.as_slice())
    }

    /// The accepted authoritative head at a given segment height, if any.
    #[must_use]
    pub fn head_at(&self, segment: u64) -> Option<&AttestedHead> {
        self.accepted.get(&segment)
    }

    /// Ingest a gossiped attested chain head (architecture §4.4). Judges it through the run's
    /// declared `Authority` ([`AuthorityConfig::authorize`]), then checks for a fork against the
    /// accepted chain:
    ///
    /// - a **different** authoritative head at the **same** height → [`ForkEvidence::DivergentHead`];
    /// - a head at `N+1` whose `prev_hash` ≠ the accepted segment `N`'s hash →
    ///   [`ForkEvidence::NonExtending`].
    ///
    /// A non-conflicting head is accepted (recorded as the head at its height). Returns
    /// `Ok(Some(evidence))` on a fork (the head is NOT accepted — the accepted history stands and
    /// the caller now holds portable evidence), `Ok(None)` on a clean accept.
    ///
    /// # Errors
    /// [`ArchiveError::Unauthoritative`] carrying the topology's typed `AuthError`.
    pub fn ingest_head(
        &mut self,
        head: AttestedHead,
    ) -> Result<Option<ForkEvidence>, ArchiveError> {
        self.head_authorize(&head)
            .map_err(|e| ArchiveError::Unauthoritative(e.to_string()))?;
        // Same-height divergence: the authority served two different segment-N histories.
        if let Some(existing) = self.accepted.get(&head.body.segment) {
            if existing.body.segment_hash != head.body.segment_hash {
                return Ok(Some(ForkEvidence::DivergentHead {
                    a: Box::new(existing.clone()),
                    b: Box::new(head),
                }));
            }
            // An exact re-gossip of the accepted head: nothing to do.
            return Ok(None);
        }
        // Non-extension: a head at N+1 whose prev link does not match the accepted segment N.
        if head.body.segment > 0 {
            if let Some(prev) = self.accepted.get(&(head.body.segment - 1)) {
                if prev.body.segment_hash != head.body.prev_hash {
                    return Ok(Some(ForkEvidence::NonExtending {
                        accepted: Box::new(prev.clone()),
                        conflicting: Box::new(head),
                    }));
                }
            }
        }
        self.accepted.insert(head.body.segment, head);
        Ok(None)
    }

    /// Mark the latest checkpoint-attested segment ordinal — the GC floor when GC is
    /// attestation-keyed (architecture §4.4: records inside the attested checkpoint's catch-up
    /// window are never collected).
    pub fn set_attested_checkpoint(&mut self, segment: u64) {
        self.attested_segment = Some(match self.attested_segment {
            Some(cur) => cur.max(segment),
            None => segment,
        });
    }

    /// The highest stored segment ordinal (the archive head).
    #[must_use]
    pub fn head_segment(&self) -> Option<u64> {
        self.segments.values().map(|s| s.header.segment).max()
    }

    /// Garbage-collect segments below the retention horizon (architecture §4.4). A segment is
    /// collectable only if its ordinal is `< head - horizon` **and** (when attestation-keyed) `<`
    /// the latest attested checkpoint's catch-up floor — records inside the attested window are
    /// never collected. Returns the number of segments dropped.
    pub fn gc(&mut self) -> usize {
        let Some(head) = self.head_segment() else {
            return 0;
        };
        let horizon_floor = head.saturating_sub(self.retention.horizon_segments);
        // The attested checkpoint pins a floor no GC may cross when attestation-keyed.
        let floor = if self.retention.gc_keyed_to_attestation {
            match self.attested_segment {
                // Nothing attested yet → keep everything (never collect an unwitnessed prefix).
                None => return 0,
                Some(att) => horizon_floor.min(att),
            }
        } else {
            horizon_floor
        };
        let drop: Vec<Hash> = self
            .segments
            .iter()
            .filter(|(_, s)| s.header.segment < floor)
            .map(|(h, _)| *h)
            .collect();
        for h in &drop {
            self.segments.remove(h);
        }
        drop.len()
    }

    /// The number of segments currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether the archive holds no segments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// Stand-alone fork detection over a set of gossiped attested heads under the run's declared
/// `Authority` (architecture §4.3/§10): the equivocation-drill primitive. Returns the first pair
/// of authoritative heads at the same height with different content — self-contained, portable
/// evidence needing nothing beyond the two heads and the run's `AuthorityConfig`. Heads that fail
/// [`AuthorityConfig::authorize`] are ignored (unsigned/forged/sub-quorum noise cannot manufacture
/// a fork).
#[must_use]
pub fn detect_fork(heads: &[AttestedHead], authority: &AuthorityConfig) -> Option<ForkEvidence> {
    let authoritative = |h: &AttestedHead| {
        h.preimage()
            .is_ok_and(|p| authority.authorize(&p, &h.sigs).is_ok())
    };
    for (i, a) in heads.iter().enumerate() {
        if !authoritative(a) {
            continue;
        }
        for b in &heads[i + 1..] {
            if !authoritative(b) {
                continue;
            }
            if a.body.segment == b.body.segment && a.body.segment_hash != b.body.segment_hash {
                return Some(ForkEvidence::DivergentHead {
                    a: Box::new(a.clone()),
                    b: Box::new(b.clone()),
                });
            }
        }
    }
    None
}
