// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The lineage semantic fold — the certification kernel's round-vocabulary half (the session
//! crate's `certify_lineage` is the schema-free half; xtask orchestrates both into one
//! verdict).
//!
//! Across a WHOLE coordinator lineage (every span of every chain, in succession order), the
//! published consensus decisions must tell one non-equivocating story:
//!
//! * **Equivocation** — one round, two DIFFERENT published `RoundRecord`s = RED. An IDENTICAL
//!   duplicate deduplicates: replay-forward re-publishing retained records to a rejoining
//!   peer, and a successor chain re-deriving its predecessor's tail, are legal and expected.
//! * **Continuity** — the deduplicated committed rounds must be dense `0..=max`, and first
//!   occurrences must appear in ascending order (a regression means a successor re-derived a
//!   DIFFERENT past — its identical re-publish would have deduplicated).
//! * **Digest conflicts** — one `(peer, round)`, two different state digests = RED; identical
//!   duplicates collapse. (The silent-overwrite defect this replaces: `assemble.rs` used to
//!   keep whichever arrived last.)
//!
//! The fold's deduplicated records then feed [`super::consensus::verify_committed_payloads`]
//! (payload/set closure) — the fold itself never touches payload bytes.

use std::collections::BTreeMap;

use daemon_vhc_proto::PeerId;
use daemon_vhc_sdk_consensus::coordinator::Input;
use daemon_vhc_sdk_consensus::messages::{RoundRecord, VhcMessage};

use super::consensus::{extract_consensus_capture, extract_wire_capture, records_are_wire_form};
use super::record::Record;

/// A semantic-fold refusal: the lineage's published decisions do not tell one
/// non-equivocating, continuous story. Archive-material RED, never module divergence.
#[derive(Debug, thiserror::Error)]
pub enum SemanticFoldError {
    /// Two DIFFERENT `RoundRecord`s published for one round.
    #[error("round {round}: two different RoundRecords published (equivocation)")]
    Equivocation {
        /// The equivocated round.
        round: u64,
    },
    /// The deduplicated committed rounds skip or regress.
    #[error("round continuity: {detail}")]
    Continuity {
        /// What broke.
        detail: String,
    },
    /// Two different state digests published by one peer for one round.
    #[error("peer {peer} round {round}: conflicting state digests", peer = .peer.to_hex())]
    DigestConflict {
        /// The conflicting peer.
        peer: PeerId,
        /// The conflicted round.
        round: u64,
    },
    /// A harness-form capture failed to extract.
    #[error("capture extraction: {0}")]
    Codec(String),
}

/// The fold's product: one deduplicated, continuity-checked consensus story.
#[derive(Debug)]
pub struct SemanticFold {
    /// The deduplicated `RoundRecord`s in ascending round order — the payload-closure input.
    pub records: Vec<RoundRecord>,
    /// Identical duplicate `RoundRecord`s collapsed (replay-forward re-publishes).
    pub duplicates_deduped: u64,
    /// Conflict-checked per-peer digest transcripts: `peer → round → digest`.
    pub by_peer: BTreeMap<PeerId, BTreeMap<u64, [u8; 16]>>,
    /// Identical duplicate digests collapsed.
    pub digest_duplicates_deduped: u64,
}

/// Fold a lineage's records (wire-form production journals or harness-form SDK journals,
/// detected exactly as `assemble` does) into one deduplicated consensus story.
///
/// # Errors
/// A typed [`SemanticFoldError`]; certification is RED on any of them.
pub fn semantic_fold(records: &[Record]) -> Result<SemanticFold, SemanticFoldError> {
    // (round record, publish order) + (peer, round, digest) in lineage order.
    let mut published: Vec<RoundRecord> = Vec::new();
    let mut digests: Vec<(PeerId, u64, [u8; 16])> = Vec::new();
    if records_are_wire_form(records) {
        let capture = extract_wire_capture(records);
        for publish in &capture.published {
            if let VhcMessage::RoundRecord(record) = &publish.message {
                published.push(record.clone());
            }
        }
        for frame in &capture.frames {
            if let VhcMessage::Digest(digest) = &frame.message {
                digests.push((frame.sender, digest.round, digest.digest.0));
            }
        }
    } else {
        let capture = extract_consensus_capture(records)
            .map_err(|e| SemanticFoldError::Codec(e.to_string()))?;
        for sm in &capture.published {
            if let VhcMessage::RoundRecord(record) = &sm.payload {
                published.push(record.clone());
            }
        }
        for input in &capture.inputs {
            if let Input::Message(sm) = input {
                if let VhcMessage::Digest(digest) = &sm.payload {
                    digests.push((sm.signer, digest.round, digest.digest.0));
                }
            }
        }
    }

    // -- equivocation + first-occurrence order ----------------------------------------------
    let mut by_round: BTreeMap<u64, RoundRecord> = BTreeMap::new();
    let mut duplicates_deduped = 0u64;
    let mut last_first_seen: Option<u64> = None;
    for record in published {
        match by_round.get(&record.round) {
            Some(existing) if *existing == record => duplicates_deduped += 1,
            Some(_) => {
                return Err(SemanticFoldError::Equivocation {
                    round: record.round,
                });
            }
            None => {
                if last_first_seen.is_some_and(|last| record.round < last) {
                    return Err(SemanticFoldError::Continuity {
                        detail: format!(
                            "round {} first appears AFTER round {} (a regression that did \
                             not deduplicate)",
                            record.round,
                            last_first_seen.unwrap_or(0)
                        ),
                    });
                }
                last_first_seen = Some(record.round);
                by_round.insert(record.round, record);
            }
        }
    }

    // -- density: the committed rounds are 0..=max, no skips --------------------------------
    if let (Some(min), Some(max)) = (
        by_round.keys().next().copied(),
        by_round.keys().last().copied(),
    ) {
        if min != 0 {
            return Err(SemanticFoldError::Continuity {
                detail: format!("the lineage's first committed round is {min}, not 0"),
            });
        }
        if by_round.len() as u64 != max + 1 {
            let missing = (0..=max).find(|r| !by_round.contains_key(r)).unwrap_or(0);
            return Err(SemanticFoldError::Continuity {
                detail: format!("committed rounds skip round {missing} (0..={max})"),
            });
        }
    }

    // -- per-peer digest transcripts, conflict-checked --------------------------------------
    let mut by_peer: BTreeMap<PeerId, BTreeMap<u64, [u8; 16]>> = BTreeMap::new();
    let mut digest_duplicates_deduped = 0u64;
    for (peer, round, digest) in digests {
        match by_peer.entry(peer).or_default().entry(round) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(digest);
            }
            std::collections::btree_map::Entry::Occupied(o) if *o.get() == digest => {
                digest_duplicates_deduped += 1;
            }
            std::collections::btree_map::Entry::Occupied(_) => {
                return Err(SemanticFoldError::DigestConflict { peer, round });
            }
        }
    }

    Ok(SemanticFold {
        records: by_round.into_values().collect(),
        duplicates_deduped,
        by_peer,
        digest_duplicates_deduped,
    })
}
