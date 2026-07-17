// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **Consensus replay — the third replay tier** (architecture §3.6; refactor §8/D2 acceptance:
//! "digests re-verified from archive + payloads alone, gated in tier-2").
//!
//! Post-consensus state is a pure function of `(records, payloads)`: the committed-input
//! discipline guarantees det state depends on nothing else, so **no journal access beyond the
//! public record archive is needed** — any third party holding the archived segments and the
//! content-addressed payloads can re-verify every consensus decision and every digest. This
//! module is that third party's verifier:
//!
//! 1. **Chain walk from attested heads** — the gossiped [`AttestedHead`]s name the sealed
//!    segments; each is judged through the run's declared `Authority`
//!    ([`RecordArchive::head_authorize`] over D1's `AuthorityConfig` — `SingleKey` and
//!    `ThresholdKeys` alike), checked for chain contiguity (`prev_hash` extends the previous
//!    head), fetched **by content address**, and re-hashed (a third party trusts no store).
//! 2. **Record recovery** — the §8.3 records are recovered from the segment bytes alone: the
//!    tag-10 snapshot is the initial coordinator state, tag-1/tag-3 records are the driving
//!    inputs, and tag-4 publishes carry the coordinator's own signed `RoundOpen`/`RoundRecord`
//!    decisions — the oracle.
//! 3. **Re-derivation** — the recovered inputs re-run inside the sandboxed coordinator module
//!    ([`crate::replay::replay_from_state`] over a [`CoordinatorSandbox`]); every archived
//!    `RoundRecord` must re-derive byte-identically (the coordinator-oracle discipline, now sourced
//!    from the archive and driven through the sandbox — consensus never runs natively).
//! 4. **Digest re-verification from payloads alone** — every committed `RecordEntry` of every
//!    record is checked against the supplied content-addressed payload bytes (`blake3(payload) ==
//!    entry.hash`, `len == entry.size`), and the record's set commitment is **recomputed** from
//!    those pairs (`commit_set`) and compared against the record's committed digest. A missing
//!    payload is the typed [`ConsensusReplayError::MissingPayload`] — the run is reported
//!    incomplete, **never a pass** (the §8.7 `ReplayMissingPayload` discipline).

use std::collections::BTreeMap;

use daemon_vhc_proto::messages::{SignedMessage, SwarmMessage};
use daemon_vhc_proto::{blake3_hash, commit_set, from_canonical_slice, Hash, PeerId};

use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, Input};

use crate::replay::{replay_from_state, CoordinatorSandbox, ReplayError, ReplayReport};

use super::archive::{AttestedHead, RecordArchive};
use super::record::{Body, Record};
use super::segment::scan_bytes;

/// A successful consensus replay: everything a third party re-verified from archive + payloads.
#[derive(Clone, Debug)]
pub struct ConsensusReplayReport {
    /// Sealed segments walked (chain-verified, content-address-re-hashed).
    pub segments_verified: u64,
    /// Records recovered across them (§8.3, seals excluded).
    pub records_recovered: u64,
    /// The inner oracle replay: re-derived records + the final coordinator state hash.
    pub replay: ReplayReport,
    /// Committed `(peer, hash)` entries whose payload bytes re-verified (`blake3` + size).
    pub payload_entries_verified: u64,
    /// `RoundRecord`s whose set commitment was recomputed from the payloads and matched.
    pub set_commitments_verified: u64,
}

/// Why a consensus replay did not complete (typed; an incomplete replay is never a pass).
#[derive(Debug, thiserror::Error)]
pub enum ConsensusReplayError {
    /// A head failed the archive's `SingleKey` authority seam (wrong signer / bad signature).
    #[error("head for segment {segment} is not authoritative")]
    Unauthoritative {
        /// The claimed segment ordinal.
        segment: u64,
    },
    /// The heads do not form one contiguous chain from segment 0.
    #[error("chain broken at segment {segment}: {detail}")]
    ChainBroken {
        /// The segment at which contiguity failed.
        segment: u64,
        /// What failed (missing head, prev mismatch).
        detail: String,
    },
    /// A named segment is absent from the archive replica.
    #[error("segment {segment} ({hash}) missing from the archive", hash = .hash.to_hex())]
    MissingSegment {
        /// The segment ordinal.
        segment: u64,
        /// Its content address.
        hash: Hash,
    },
    /// Fetched bytes do not hash to the head's content address (a lying store).
    #[error("segment {segment} bytes do not match their content address")]
    ContentMismatch {
        /// The segment ordinal.
        segment: u64,
    },
    /// A segment's internal structure failed to scan/decode.
    #[error("segment {segment} scan: {detail}")]
    BadSegment {
        /// The segment ordinal.
        segment: u64,
        /// The scan failure.
        detail: String,
    },
    /// The archived records carry no tag-10 snapshot (no initial coordinator state).
    #[error("archive carries no snapshot record (initial coordinator state)")]
    NoSnapshot,
    /// The inner oracle replay diverged or failed (the archived records do not re-derive).
    #[error("oracle replay: {0}")]
    Replay(#[from] ReplayError),
    /// A committed payload is not in the supplied content-addressed set — the replay is
    /// INCOMPLETE, never a pass (§8.7 `ReplayMissingPayload` discipline).
    #[error("round {round}: committed payload {hash} missing from the supplied payload set",
            hash = .hash.to_hex())]
    MissingPayload {
        /// The round whose record commits the payload.
        round: u64,
        /// The committed content hash.
        hash: Hash,
    },
    /// Supplied payload bytes do not match a committed entry (hash or size).
    #[error("round {round}: payload for peer does not re-verify: {detail}")]
    PayloadMismatch {
        /// The round.
        round: u64,
        /// What mismatched.
        detail: String,
    },
    /// A record's set commitment does not recompute from its committed `(peer, hash)` pairs.
    #[error("round {round}: set commitment does not recompute from the payload set")]
    SetCommitmentMismatch {
        /// The round.
        round: u64,
    },
    /// A record decode failed (tag-4 frame → `SignedMessage`).
    #[error("record decode: {0}")]
    Codec(String),
}

/// The §8.3 record stream's consensus projection (what [`extract_consensus_capture`] recovers).
#[derive(Debug)]
pub struct ConsensusCapture {
    /// The tag-10 initial coordinator state, if the stream carries one.
    pub initial: Option<CoordinatorState>,
    /// The tag-1/3 driving inputs, in record order.
    pub inputs: Vec<Input>,
    /// The tag-4 published decisions (signed), in record order.
    pub published: Vec<SignedMessage>,
}

/// Project a §8.3 record stream into its consensus capture: the tag-10 initial state (if any),
/// the tag-1/3 driving inputs, and the tag-4 published decisions, in record order. Shared by the
/// archive verifier below and the D2 failover drill's standby recovery (which appends the
/// unsealed local journal *tail*'s records after the archived prefix — "resumes from archive +
/// journal").
///
/// # Errors
/// [`ConsensusReplayError::Codec`] on an undecodable snapshot/event/publish body.
pub fn extract_consensus_capture(
    records: &[Record],
) -> Result<ConsensusCapture, ConsensusReplayError> {
    let mut initial: Option<CoordinatorState> = None;
    let mut inputs: Vec<Input> = Vec::new();
    let mut published: Vec<SignedMessage> = Vec::new();
    for record in records {
        match &record.body {
            Body::Snapshot(s) => {
                if initial.is_none() {
                    initial = Some(
                        from_canonical_slice(&s.manifest)
                            .map_err(|e| ConsensusReplayError::Codec(e.to_string()))?,
                    );
                }
            }
            Body::Clock(c) => inputs.push(Input::Clock(c.now)),
            Body::Event(e) => inputs.push(
                from_canonical_slice(&e.frame)
                    .map_err(|e| ConsensusReplayError::Codec(e.to_string()))?,
            ),
            Body::Publish(p) => published.push(
                from_canonical_slice(&p.frame)
                    .map_err(|e| ConsensusReplayError::Codec(e.to_string()))?,
            ),
            _ => {}
        }
    }
    Ok(ConsensusCapture {
        initial,
        inputs,
        published,
    })
}

/// What the archive chain walk recovered: the verified sealed-segment count and the §8.3 records
/// across them (seals excluded), in order.
#[derive(Debug)]
pub struct RecoveredChain {
    /// Sealed segments walked (authenticated heads, contiguous chain, content re-hashed).
    pub segments_verified: u64,
    /// The recovered records.
    pub records: Vec<Record>,
}

/// Walk the attested head chain and recover every record from the archive alone (steps 1–2 of
/// the module contract). Also the failover drill's standby recovery path (refactor §8/D2: a
/// standby "resumes from archive + journal" — this is the archive half).
///
/// # Errors
/// A typed [`ConsensusReplayError`] on an unauthoritative head, a broken chain, a missing or
/// content-mismatched segment, or an unscannable segment.
pub fn recover_chain_from_archive(
    archive: &RecordArchive,
    heads: &[AttestedHead],
) -> Result<RecoveredChain, ConsensusReplayError> {
    let mut by_segment: BTreeMap<u64, &AttestedHead> = BTreeMap::new();
    for head in heads {
        if !archive.head_is_authoritative(head) {
            return Err(ConsensusReplayError::Unauthoritative {
                segment: head.body.segment,
            });
        }
        by_segment.insert(head.body.segment, head);
    }
    let count = by_segment.len() as u64;
    let mut prev_hash: Option<Hash> = None;
    let mut records: Vec<Record> = Vec::new();
    for segment in 0..count {
        let head = by_segment
            .get(&segment)
            .ok_or_else(|| ConsensusReplayError::ChainBroken {
                segment,
                detail: "no authoritative head at this ordinal".into(),
            })?;
        if let Some(prev) = prev_hash {
            if head.body.prev_hash != prev {
                return Err(ConsensusReplayError::ChainBroken {
                    segment,
                    detail: "prev_hash does not extend the previous head".into(),
                });
            }
        }
        let bytes =
            archive
                .fetch(&head.body.segment_hash)
                .ok_or(ConsensusReplayError::MissingSegment {
                    segment,
                    hash: head.body.segment_hash,
                })?;
        // A third party re-hashes what it fetched — the store is untrusted.
        if blake3_hash(bytes) != head.body.segment_hash {
            return Err(ConsensusReplayError::ContentMismatch { segment });
        }
        let scan = scan_bytes(bytes).map_err(|e| ConsensusReplayError::BadSegment {
            segment,
            detail: e.to_string(),
        })?;
        if !scan.sealed {
            return Err(ConsensusReplayError::BadSegment {
                segment,
                detail: "archived segment is not sealed".into(),
            });
        }
        for record in scan.records {
            if !matches!(record.body, Body::Seal(_)) {
                records.push(record);
            }
        }
        prev_hash = Some(head.body.segment_hash);
    }
    Ok(RecoveredChain {
        segments_verified: count,
        records,
    })
}

/// Re-verify a run's consensus from the record archive and the content-addressed payloads alone
/// (see module docs). `heads` are the gossiped attested chain heads naming the sealed segments —
/// they MUST cover a contiguous chain from segment 0; `payloads` maps content hash → payload
/// bytes (the run's update containers, fetched from any payload store).
///
/// # Errors
/// A typed [`ConsensusReplayError`]; an incomplete verification is never a pass.
pub fn replay_consensus_from_archive(
    sandbox: &dyn CoordinatorSandbox,
    archive: &RecordArchive,
    heads: &[AttestedHead],
    payloads: &BTreeMap<Hash, Vec<u8>>,
) -> Result<ConsensusReplayReport, ConsensusReplayError> {
    // -- 1./2. chain walk + record recovery, from the archive alone ------------------------------
    let chain = recover_chain_from_archive(archive, heads)?;
    let count = chain.segments_verified;
    let records_recovered = chain.records.len() as u64;
    let capture = extract_consensus_capture(&chain.records)?;
    let initial = capture.initial.ok_or(ConsensusReplayError::NoSnapshot)?;

    // -- 3. re-derive: the pure tick over the inputs; archived records are the oracle ------------
    let oracle_records: Vec<Input> = capture
        .published
        .iter()
        .filter(|sm| matches!(sm.payload, SwarmMessage::RoundRecord(_)))
        .cloned()
        .map(Input::Message)
        .collect();
    let replay = replay_from_state(
        sandbox,
        initial,
        capture.inputs.into_iter().chain(oracle_records),
    )?;

    // -- 4. digests from payloads alone: every committed entry + every set commitment ------------
    let mut payload_entries_verified = 0u64;
    let mut set_commitments_verified = 0u64;
    for record in &replay.records {
        let mut pairs: Vec<(PeerId, Hash)> = Vec::new();
        for entry in record.inline.iter().flatten() {
            let bytes = payloads
                .get(&entry.hash)
                .ok_or(ConsensusReplayError::MissingPayload {
                    round: record.round,
                    hash: entry.hash,
                })?;
            if blake3_hash(bytes) != entry.hash {
                return Err(ConsensusReplayError::PayloadMismatch {
                    round: record.round,
                    detail: "payload bytes do not hash to the committed entry".into(),
                });
            }
            if bytes.len() as u64 != entry.size {
                return Err(ConsensusReplayError::PayloadMismatch {
                    round: record.round,
                    detail: format!(
                        "payload size {} != committed size {}",
                        bytes.len(),
                        entry.size
                    ),
                });
            }
            pairs.push((entry.peer, entry.hash));
            payload_entries_verified += 1;
        }
        // The record's committed digest recomputes from the payload set alone (§6.4 I3/I6) —
        // the "any third party can re-verify every digest" claim, made executable.
        if commit_set(&pairs).commitment() != record.set {
            return Err(ConsensusReplayError::SetCommitmentMismatch {
                round: record.round,
            });
        }
        set_commitments_verified += 1;
    }

    Ok(ConsensusReplayReport {
        segments_verified: count,
        records_recovered,
        replay,
        payload_entries_verified,
        set_commitments_verified,
    })
}
