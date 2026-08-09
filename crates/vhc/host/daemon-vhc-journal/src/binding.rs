// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Head↔segment identity binding — the shared verifier every archive reader runs after
//! `scan_bytes` (ABI §8.8): the signed head's claims must equal what the segment's own header
//! and seal actually say, or the reader is folding foreign bytes that merely content-hash to
//! an attested address it never cross-checked field-by-field.
//!
//! What binds, and why each field:
//!
//! * **sealed, untruncated, seal-count-consistent** — an archive segment is a cleanly rolled,
//!   immutable unit (§8.2); the tag-17 seal's declared record count must equal the records the
//!   scan actually recovered (the seal count EXCLUDES the seal record itself).
//! * **`run_id`, `epoch`, `role`, `module`** — the frozen execution identity in the segment
//!   header (§8.1) must be the identity the head was signed over.
//! * **segment ordinal** — the chain height the head attests is the ordinal the header carries.
//! * **`prev_blake3`** — the header's chain link must be the head's `prev_hash`.
//! * **chain scope** — the header's frozen `instance` is the chain's FOUNDING incarnation
//!   (`ArchiveHeadBody::chain_instance`), NOT the head's `instance` field: an adopted
//!   abandoned-tail head (a successor attesting its crashed predecessor's final segment —
//!   defect 16) legitimately carries the ADOPTER's incarnation there, while the segment header
//!   keeps the predecessor chain's identity (verified against the real c15m archive: 7 of 875
//!   heads are adoptions and differ exactly and only in that field). A claim form that does
//!   not carry the chain scope (the harness `ChainHead`) passes `None` and skips the check.
//!
//! The head's signature/authorization is NOT judged here — `verify_chains` /
//! `AuthorityConfig::authorize` own trust; this is the structural binding below it.

use daemon_vhc_proto::Hash;

use crate::record::Body;
use crate::segment::ScanResult;

/// The signed head fields a scanned segment must bind to. Constructed by each reader from its
/// own claim form (`ArchiveHeadBody`, the harness `ChainHead`).
#[derive(Clone, Debug)]
pub struct HeadClaim<'a> {
    /// The run's cryptographic id (the genesis hash).
    pub run_id: Hash,
    /// The transition-chain epoch position.
    pub epoch: u64,
    /// The envelope role label.
    pub role: &'a str,
    /// The pinned module hash.
    pub module: Hash,
    /// The chain scope's founding incarnation (`ArchiveHeadBody::chain_instance`) — the frozen
    /// journal identity's `instance`. `None` when the claim form carries no chain scope.
    pub chain_instance: Option<u64>,
    /// The 0-based segment ordinal the head attests.
    pub segment: u64,
    /// The previous segment's content address (all-zero at segment 0).
    pub prev_hash: Hash,
    /// The head's declared record count (excluding the seal).
    pub records: u64,
}

impl<'a> HeadClaim<'a> {
    /// The claim of a signed §8.8 archive head record — the full form, chain scope included.
    #[must_use]
    pub fn from_archive_head(body: &'a daemon_vhc_proto::ArchiveHeadBody) -> Self {
        Self {
            run_id: body.run_id,
            epoch: body.epoch,
            role: &body.role,
            module: body.module,
            chain_instance: Some(body.chain_instance),
            segment: body.segment,
            prev_hash: body.prev_hash,
            records: body.records,
        }
    }
}

/// A binding violation: the segment's own header/seal disagree with the signed head's claims.
/// Callers fold this into their reader-local typed refusal (`Segment` / `BadSegment`) — the
/// archive material is wrong, never the network.
#[derive(Debug, thiserror::Error)]
#[error("head↔segment binding: {0}")]
pub struct HeadBindingError(pub String);

/// Verify a scanned segment against the signed head's claims (see the module docs for the
/// field-by-field contract).
///
/// # Errors
/// [`HeadBindingError`] naming the first field that disagrees.
pub fn verify_head_binding(
    scan: &ScanResult,
    claim: &HeadClaim<'_>,
) -> Result<(), HeadBindingError> {
    let err = |detail: String| Err(HeadBindingError(detail));
    if !scan.sealed || scan.truncated {
        return err(format!(
            "attested segment is not a cleanly sealed unit (sealed={}, truncated={})",
            scan.sealed, scan.truncated
        ));
    }
    let h = &scan.header;
    if h.id.run_id != claim.run_id {
        return err(format!(
            "segment header run_id {} != attested {}",
            h.id.run_id.to_hex(),
            claim.run_id.to_hex()
        ));
    }
    if h.id.epoch != claim.epoch {
        return err(format!(
            "segment header epoch {} != attested {}",
            h.id.epoch, claim.epoch
        ));
    }
    if h.id.role != claim.role {
        return err(format!(
            "segment header role {:?} != attested {:?}",
            h.id.role, claim.role
        ));
    }
    if h.id.module != claim.module {
        return err(format!(
            "segment header module {} != attested {}",
            h.id.module.to_hex(),
            claim.module.to_hex()
        ));
    }
    if let Some(chain_instance) = claim.chain_instance {
        if h.id.instance != chain_instance {
            return err(format!(
                "segment header instance {} != attested chain scope {chain_instance}",
                h.id.instance
            ));
        }
    }
    if h.segment != claim.segment {
        return err(format!(
            "segment header ordinal {} != attested {}",
            h.segment, claim.segment
        ));
    }
    if h.prev_blake3 != claim.prev_hash.0 {
        return err(format!(
            "segment header prev link {} != attested {}",
            Hash(h.prev_blake3).to_hex(),
            claim.prev_hash.to_hex()
        ));
    }
    // The seal is the last scanned record of a sealed segment; its declared count excludes
    // itself. Both equalities must hold: seal vs the records actually recovered, and seal vs
    // the head's signed claim.
    let Some(Body::Seal(seal)) = scan.records.last().map(|r| &r.body) else {
        return err("sealed segment's last record is not the tag-17 seal".into());
    };
    let recovered = (scan.records.len() as u64).saturating_sub(1);
    if seal.records != recovered {
        return err(format!(
            "seal declares {} records but the scan recovered {recovered}",
            seal.records
        ));
    }
    if seal.records != claim.records {
        return err(format!(
            "seal declares {} records but the head attests {}",
            seal.records, claim.records
        ));
    }
    Ok(())
}

// The suite lives with `daemon-vhc-observe` (`tests/journal.rs`), the substrate's test home —
// this crate deliberately carries no dev-dependencies.
