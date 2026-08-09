// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **Archive assembly** — pull a run's product archive (ABI §8.8) into the on-disk replay layout
//! (runbook §3.4) and verify every byte on the way in.
//!
//! The inputs are exactly what any third party can obtain: the frozen genesis envelope bytes, a
//! snapshot of the run's published [`ArchiveHeadRecord`]s (from the registry's untrusted archive
//! slots or a filesystem head store), and a content-addressed fetch function over the run's
//! content plane (R2 / filesystem). The assembler:
//!
//! 1. **Authorizes every head** through [`ArchiveHeadRecord::authorize`] against the envelope's
//!    genesis-trusted bases (`identities.coordinator` + `identities.coordinator_set`) — the
//!    §8.8 \[AR-4\] reader-side judgment the untrusted store never makes.
//! 2. **Re-folds every chain** through the normative structural fold
//!    ([`daemon_vhc_proto::archive::ArchiveChainSlot`]): dense ordinals, `prev_hash` linkage —
//!    a snapshot that does not fold is a corrupt (or forked) archive, refused typed.
//! 3. **Fetches every sealed segment** by content address and **re-hashes** it (the store is
//!    untrusted), writing `segments/<hex>.seg`.
//! 4. **Walks the coordinator lineage** (founding chain → succession links) and recovers its
//!    records to enumerate the run's committed payloads, fetching and re-hashing each into
//!    `payloads/<hex>.bin`.
//! 5. **Extracts the per-peer digest transcripts** from the coordinator chain's recorded driving
//!    inputs (every signed [`VhcMessage::Digest`]), writing `peers/<peer>.digests.cbor`.
//! 6. Writes `envelope.cbor` and fetches the genesis-pinned `coordinator.wasm` by its artifact
//!    hash.
//!
//! `heads.cbor` carries the FULL verified head-record set (every role's chains, canonical CBOR
//! `Vec<ArchiveHeadRecord>`): the replay reader re-groups and selects the coordinator lineage
//! itself, and trainer chains ride along as replay/evidence inputs.

// Sanctioned raw-fs home (the journal-substrate pattern): the assembler writes an
// operator-chosen replay-archive directory with fsync + rename atomicity; oracle tooling, never
// linked by a production host graph (vhc-dep-check). No spawn / env mutation here.
#![allow(clippy::disallowed_methods)]

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::Path;

use daemon_vhc_proto::archive::ArchiveHeadRecord;
use daemon_vhc_proto::genesis::{FrozenGenesis, GenesisEnvelope};
use daemon_vhc_proto::{
    blake3_hash, from_canonical_slice, to_canonical_vec, Hash, PeerId, SignedEnvelope,
};

// The reader-side chain verification moved to the wire-contract crate (the node's join
// transaction makes the same judgment without linking this oracle crate); re-exported here so
// the assembler's consumers keep one import surface.
pub use daemon_vhc_proto::archive::{
    coordinator_lineage, envelope_trusted_bases, verify_chains, ChainVerifyError, VerifiedChain,
};
use daemon_vhc_sdk_consensus::coordinator::Input;
use daemon_vhc_sdk_consensus::messages::VhcMessage;

use super::consensus::{extract_consensus_capture, extract_wire_capture, records_are_wire_form};
use super::record::{Body, Record};
use super::segment::scan_bytes;

/// A content-plane fetch: content hash → bytes. The assembler re-hashes everything it fetches,
/// so the fetcher needs no integrity guarantees of its own.
pub type ContentFetch<'a> = dyn FnMut(&Hash) -> Result<Vec<u8>, String> + 'a;

/// What a successful assembly wrote (and verified).
#[derive(Clone, Debug)]
pub struct AssembleReport {
    /// The run id (the envelope bytes' BLAKE3).
    pub run_id: Hash,
    /// Verified chains across every role.
    pub chains_verified: u64,
    /// Head records written to `heads.cbor`.
    pub heads_written: u64,
    /// Segment objects fetched, re-hashed, and written.
    pub segments_written: u64,
    /// Payload objects fetched, re-hashed, and written.
    pub payloads_written: u64,
    /// Per-peer digest transcripts written.
    pub peer_transcripts: u64,
    /// The coordinator lineage's chains, founding first (their `chain_instance`s).
    pub coordinator_lineage: Vec<u64>,
}

/// Why an assembly refused (typed; an incomplete archive is never written silently).
#[derive(Debug, thiserror::Error)]
pub enum AssembleError {
    /// The envelope bytes do not decode / validate.
    #[error("genesis envelope: {0}")]
    Envelope(String),
    /// Reader-side chain verification refused the head snapshot (authorization, fold, lineage —
    /// the shared `daemon_vhc_proto::archive` judgment).
    #[error(transparent)]
    Verify(#[from] ChainVerifyError),
    /// A content fetch failed.
    #[error("fetch {hash}: {detail}", hash = .hash.to_hex())]
    Fetch {
        /// The requested content address.
        hash: Hash,
        /// The transport failure.
        detail: String,
    },
    /// Fetched bytes do not hash to their content address (a lying store).
    #[error("fetched bytes do not match content address {hash}", hash = .hash.to_hex())]
    ContentMismatch {
        /// The claimed address.
        hash: Hash,
    },
    /// A fetched segment does not scan as a sealed segment.
    #[error("segment {hash}: {detail}", hash = .hash.to_hex())]
    BadSegment {
        /// The segment's content address.
        hash: Hash,
        /// The scan failure.
        detail: String,
    },
    /// A record body failed to decode while enumerating payloads/digests.
    #[error("record decode: {0}")]
    Codec(String),
    /// One peer published two DIFFERENT state digests for one round — never silently
    /// last-writer-wins (equivocating evidence must refuse, not overwrite).
    #[error("peer {peer} round {round}: conflicting state digests", peer = .peer.to_hex())]
    DigestConflict {
        /// The conflicting peer.
        peer: PeerId,
        /// The conflicted round.
        round: u64,
    },
    /// Filesystem failure writing the layout.
    #[error("io: {0}")]
    Io(String),
}

impl From<std::io::Error> for AssembleError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// Assemble the §3.4 replay layout for one run under `out` (created if missing) from the
/// envelope bytes, a head-record snapshot, and a content-plane fetch. See the module docs for
/// the verification each step performs.
///
/// # Errors
/// A typed [`AssembleError`]; nothing is reported assembled unless every byte verified.
pub fn assemble_archive(
    out: &Path,
    envelope_bytes: &[u8],
    records: Vec<ArchiveHeadRecord>,
    fetch: &mut ContentFetch<'_>,
) -> Result<AssembleReport, AssembleError> {
    // -- the envelope: run id, trusted bases, the pinned coordinator module ----------------------
    // The registry serves the SIGNED wire form (`SignedEnvelope { bytes, signature, signer }` —
    // the same object the node's assess path decodes), while the §3.4 layout and the RunId are
    // defined over the frozen INNER bytes. Unwrap-and-verify the wire form when that is what
    // arrived (author signature + envelope validation via `FrozenGenesis::open`); accept the
    // bare inner bytes unchanged (fixtures, pre-registry archives).
    let inner_bytes: Vec<u8> = match from_canonical_slice::<SignedEnvelope>(envelope_bytes) {
        Ok(signed) => FrozenGenesis::open(signed.bytes, signed.signature, signed.signer)
            .map_err(|e| AssembleError::Envelope(format!("signed wire form: {e}")))?
            .bytes()
            .to_vec(),
        Err(_) => envelope_bytes.to_vec(),
    };
    let envelope_bytes = inner_bytes.as_slice();
    let envelope: GenesisEnvelope =
        from_canonical_slice(envelope_bytes).map_err(|e| AssembleError::Envelope(e.to_string()))?;
    let run_id = blake3_hash(envelope_bytes);
    let trusted = envelope_trusted_bases(&envelope);
    if trusted.is_empty() {
        return Err(AssembleError::Envelope(
            "envelope names no genesis-trusted base identities".into(),
        ));
    }
    let coordinator_role = envelope
        .roles
        .keys()
        .find(|r| r.contains("coordinator"))
        .cloned()
        .ok_or_else(|| AssembleError::Envelope("envelope names no coordinator role".into()))?;
    let coord_artifact = envelope
        .artifacts
        .get("coordinator.wasm")
        .ok_or_else(|| {
            AssembleError::Envelope("envelope carries no `coordinator.wasm` artifact".into())
        })?
        .blake3;

    // -- authorize every head, group by chain, re-fold every chain -------------------------------
    let chains = verify_chains(&run_id, &trusted, records)?;

    // -- order the coordinator lineage by succession links ---------------------------------------
    let lineage = coordinator_lineage(&chains, &coordinator_role)?;

    // -- write the layout -------------------------------------------------------------------------
    std::fs::create_dir_all(out)?;
    std::fs::create_dir_all(out.join("segments"))?;
    std::fs::create_dir_all(out.join("payloads"))?;
    std::fs::create_dir_all(out.join("peers"))?;
    write_atomic(&out.join("envelope.cbor"), envelope_bytes)?;

    let coord_wasm = fetch_verified(fetch, &coord_artifact)?;
    write_atomic(&out.join("coordinator.wasm"), &coord_wasm)?;

    let all_heads: Vec<ArchiveHeadRecord> = chains
        .iter()
        .flat_map(|c| c.heads.iter().cloned())
        .collect();
    let heads_bytes =
        to_canonical_vec(&all_heads).map_err(|e| AssembleError::Codec(e.to_string()))?;
    write_atomic(&out.join("heads.cbor"), &heads_bytes)?;

    // -- fetch + re-hash every sealed segment of every chain -------------------------------------
    let mut segments_written = 0u64;
    for chain in &chains {
        for head in &chain.heads {
            let bytes = fetch_verified(fetch, &head.body.segment_hash)?;
            scan_bytes(&bytes)
                .map_err(|e| AssembleError::BadSegment {
                    hash: head.body.segment_hash,
                    detail: e.to_string(),
                })
                .and_then(|scan| {
                    // The full head↔segment identity binding (shared verifier): sealed unit,
                    // frozen execution identity, ordinal, prev link, seal count.
                    daemon_vhc_journal::verify_head_binding(
                        &scan,
                        &daemon_vhc_journal::HeadClaim::from_archive_head(&head.body),
                    )
                    .map_err(|e| AssembleError::BadSegment {
                        hash: head.body.segment_hash,
                        detail: e.to_string(),
                    })
                })?;
            write_atomic(
                &out.join("segments")
                    .join(format!("{}.seg", head.body.segment_hash.to_hex())),
                &bytes,
            )?;
            segments_written += 1;
        }
    }

    // -- recover the coordinator lineage's records: payloads + per-peer digests ------------------
    let mut lineage_records: Vec<Record> = Vec::new();
    for chain in &lineage {
        for head in &chain.heads {
            let bytes = fetch_verified(fetch, &head.body.segment_hash)?;
            let scan = scan_bytes(&bytes).map_err(|e| AssembleError::BadSegment {
                hash: head.body.segment_hash,
                detail: e.to_string(),
            })?;
            daemon_vhc_journal::verify_head_binding(
                &scan,
                &daemon_vhc_journal::HeadClaim::from_archive_head(&head.body),
            )
            .map_err(|e| AssembleError::BadSegment {
                hash: head.body.segment_hash,
                detail: e.to_string(),
            })?;
            for record in scan.records {
                if !matches!(record.body, Body::Seal(_)) {
                    lineage_records.push(record);
                }
            }
        }
    }
    // Committed payload hashes (every inline entry of every published RoundRecord) and per-peer
    // digest transcripts (every authoritative Digest frame), read through the journal form the
    // lineage actually carries: PRODUCTION journals record §12.1 wire frames (tag-12/tag-4),
    // harness journals record SDK types directly.
    let mut payload_hashes: BTreeSet<Hash> = BTreeSet::new();
    let mut by_peer: BTreeMap<PeerId, BTreeMap<u64, [u8; 16]>> = BTreeMap::new();
    if records_are_wire_form(&lineage_records) {
        let capture = extract_wire_capture(&lineage_records);
        for publish in &capture.published {
            if let VhcMessage::RoundRecord(record) = &publish.message {
                for entry in record.inline.iter().flatten() {
                    payload_hashes.insert(entry.hash);
                }
            }
        }
        for frame in &capture.frames {
            if let VhcMessage::Digest(digest) = &frame.message {
                insert_peer_digest(&mut by_peer, frame.sender, digest.round, digest.digest.0)?;
            }
        }
    } else {
        let capture = extract_consensus_capture(&lineage_records)
            .map_err(|e| AssembleError::Codec(e.to_string()))?;
        for sm in &capture.published {
            if let VhcMessage::RoundRecord(record) = &sm.payload {
                for entry in record.inline.iter().flatten() {
                    payload_hashes.insert(entry.hash);
                }
            }
        }
        for input in &capture.inputs {
            if let Input::Message(sm) = input {
                if let VhcMessage::Digest(digest) = &sm.payload {
                    insert_peer_digest(&mut by_peer, sm.signer, digest.round, digest.digest.0)?;
                }
            }
        }
    }
    let mut payloads_written = 0u64;
    for hash in &payload_hashes {
        let bytes = fetch_verified(fetch, hash)?;
        write_atomic(
            &out.join("payloads").join(format!("{}.bin", hash.to_hex())),
            &bytes,
        )?;
        payloads_written += 1;
    }
    let mut peer_transcripts = 0u64;
    for (peer, rounds) in &by_peer {
        let transcript: Vec<(u64, [u8; 16])> = rounds.iter().map(|(r, d)| (*r, *d)).collect();
        let bytes =
            to_canonical_vec(&transcript).map_err(|e| AssembleError::Codec(e.to_string()))?;
        write_atomic(
            &out.join("peers")
                .join(format!("{}.digests.cbor", peer.to_hex())),
            &bytes,
        )?;
        peer_transcripts += 1;
    }

    Ok(AssembleReport {
        run_id,
        chains_verified: chains.len() as u64,
        heads_written: all_heads.len() as u64,
        segments_written,
        payloads_written,
        peer_transcripts,
        coordinator_lineage: lineage.iter().map(|c| c.chain_instance).collect(),
    })
}

/// Record one peer's per-round digest: an identical duplicate collapses (replay-forward
/// re-publishes are legal), a DIFFERENT value for the same `(peer, round)` is the typed
/// [`AssembleError::DigestConflict`] — never a silent overwrite.
fn insert_peer_digest(
    by_peer: &mut BTreeMap<PeerId, BTreeMap<u64, [u8; 16]>>,
    peer: PeerId,
    round: u64,
    digest: [u8; 16],
) -> Result<(), AssembleError> {
    match by_peer.entry(peer).or_default().entry(round) {
        std::collections::btree_map::Entry::Vacant(v) => {
            v.insert(digest);
            Ok(())
        }
        std::collections::btree_map::Entry::Occupied(o) if *o.get() == digest => Ok(()),
        std::collections::btree_map::Entry::Occupied(_) => {
            Err(AssembleError::DigestConflict { peer, round })
        }
    }
}

/// Fetch + re-hash content-addressed bytes (the store is untrusted).
fn fetch_verified(fetch: &mut ContentFetch<'_>, hash: &Hash) -> Result<Vec<u8>, AssembleError> {
    let bytes = fetch(hash).map_err(|detail| AssembleError::Fetch {
        hash: *hash,
        detail,
    })?;
    if blake3_hash(&bytes) != *hash {
        return Err(AssembleError::ContentMismatch { hash: *hash });
    }
    Ok(bytes)
}

/// Write bytes via a same-directory temp + rename (a torn assembly never masquerades complete).
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), AssembleError> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}
