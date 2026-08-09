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

use daemon_vhc_proto::{blake3_hash, commit_set, from_canonical_slice, Hash, PeerId};
use daemon_vhc_sdk_consensus::messages::{RoundRecord, SignedMessage, VhcMessage};

use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, Input};

use crate::replay::{replay_from_state, CoordinatorSandbox, ReplayError, ReplayReport};

use super::archive::{AttestedHead, RecordArchive};
use super::record::{Body, Record, RunHeader};
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

/// One authoritative wire frame (a tag-12 record) with its inner [`VhcMessage`] decoded — a
/// PRODUCTION journal's driving input (the live relay verified the signature above the pump
/// before journaling; the segment carrying it is head-attested — reader-side re-verification
/// rides the chain walk, not per-frame signatures).
#[derive(Clone, Debug)]
pub struct WireFrame {
    /// The channel the frame arrived on.
    pub channel: u64,
    /// The per-sender sequence number.
    pub seq: u64,
    /// The frame sender identity.
    pub sender: PeerId,
    /// The decoded module-authored payload.
    pub message: VhcMessage,
}

/// One published decision (a tag-4 record) with its inner [`VhcMessage`] decoded.
#[derive(Clone, Debug)]
pub struct WirePublish {
    /// The channel published on.
    pub channel: u64,
    /// The durable channel-scoped sequence number.
    pub seq: u64,
    /// blake3 of the guest payload bytes.
    pub hash: Hash,
    /// The decoded published payload.
    pub message: VhcMessage,
}

/// A PRODUCTION (§8.3 wire-form) record stream's consensus projection: the tag-0 run header, the
/// tag-12 authoritative frames, and the tag-4 published decisions — each wire frame's inner
/// `VhcMessage` decoded structurally from the §12.1 `[envelope, payload, sig]` form.
///
/// The harness-form counterpart is [`ConsensusCapture`] ([`extract_consensus_capture`]): harness
/// journals record SDK types directly, production journals record the session wire forms. Use
/// [`records_are_wire_form`] to pick.
#[derive(Debug, Default)]
pub struct WireCapture {
    /// The tag-0 run header (identity + admitted config/grants bytes, verbatim), if present.
    pub header: Option<Box<RunHeader>>,
    /// The tag-12 authoritative frames whose payload decodes as a [`VhcMessage`], in record order.
    pub frames: Vec<WireFrame>,
    /// The tag-4 publishes whose payload decodes as a [`VhcMessage`], in record order.
    pub published: Vec<WirePublish>,
}

impl WireCapture {
    /// The published `RoundRecord`s, in publication order — the consensus decisions a verifier
    /// re-checks (digest-from-payloads via [`verify_committed_payloads`]).
    #[must_use]
    pub fn round_records(&self) -> Vec<&RoundRecord> {
        self.published
            .iter()
            .filter_map(|p| match &p.message {
                VhcMessage::RoundRecord(r) => Some(r),
                _ => None,
            })
            .collect()
    }
}

/// Whether a recovered record stream is a PRODUCTION journal (wire form), judged structurally
/// from its first frame-carrying record (tag-4 publish or inline tag-12 frame): the session
/// writes §12.1 `[envelope, payload, sig]` CBOR *arrays*, while harness captures journal the SDK
/// [`SignedMessage`] struct — a CBOR *map*. A stream with no frames yet (a bare prefix) falls
/// back to the tag-0 run header, which only the production session writes unaccompanied.
#[must_use]
pub fn records_are_wire_form(records: &[Record]) -> bool {
    let mut saw_header = false;
    for record in records {
        match &record.body {
            Body::RunHeader(_) => saw_header = true,
            Body::Publish(p) => return frame_is_wire_shaped(&p.frame),
            Body::SignedFrame(sf) => {
                if let Some(frame) = &sf.frame {
                    return frame_is_wire_shaped(frame);
                }
            }
            _ => {}
        }
    }
    saw_header
}

/// Whether `frame` has the §12.1 signed wire SHAPE (`[envelope, payload, sig]` with byte-string
/// payload) — independent of whether the payload decodes as a [`VhcMessage`].
fn frame_is_wire_shaped(frame: &[u8]) -> bool {
    let Ok(v) = ciborium::de::from_reader::<ciborium::value::Value, _>(frame) else {
        return false;
    };
    let ciborium::value::Value::Array(parts) = v else {
        return false;
    };
    matches!(parts.get(1), Some(ciborium::value::Value::Bytes(_)))
}

/// Project a PRODUCTION record stream into its [`WireCapture`]. A tag-12/tag-4 frame whose
/// payload is not a decodable [`VhcMessage`] is skipped (structurally foreign evidence — the
/// reconstruction path's discipline), never an error: the input-replay oracle re-feeds the raw
/// event stream separately, so nothing is lost to this projection.
#[must_use]
pub fn extract_wire_capture(records: &[Record]) -> WireCapture {
    let mut capture = WireCapture::default();
    for record in records {
        match &record.body {
            Body::RunHeader(h) => {
                if capture.header.is_none() {
                    capture.header = Some(h.clone());
                }
            }
            Body::SignedFrame(sf) => {
                let Some(frame) = &sf.frame else {
                    continue; // evidence-by-reference (Phase D) carries no inline bytes
                };
                let Some(message) = wire_frame_message(frame) else {
                    continue;
                };
                capture.frames.push(WireFrame {
                    channel: sf.channel,
                    seq: sf.seq,
                    sender: PeerId(sf.sender.0),
                    message,
                });
            }
            Body::Publish(p) => {
                let Some(message) = wire_frame_message(&p.frame) else {
                    continue;
                };
                capture.published.push(WirePublish {
                    channel: p.channel,
                    seq: p.seq,
                    hash: p.hash,
                    message,
                });
            }
            _ => {}
        }
    }
    capture
}

/// Decode a §12.1 signed wire frame's (`[envelope, payload, sig]`) inner payload as a
/// [`VhcMessage`] — structural CBOR, never a round schema over the outer frame.
fn wire_frame_message(frame: &[u8]) -> Option<VhcMessage> {
    let v: ciborium::value::Value = ciborium::de::from_reader(frame).ok()?;
    let ciborium::value::Value::Array(parts) = v else {
        return None;
    };
    let ciborium::value::Value::Bytes(payload) = parts.into_iter().nth(1)? else {
        return None;
    };
    from_canonical_slice::<VhcMessage>(&payload).ok()
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
    let mut by_segment: BTreeMap<u64, &super::archive::ChainHead> = BTreeMap::new();
    for head in heads {
        if !archive.head_is_authoritative(head) {
            return Err(ConsensusReplayError::Unauthoritative {
                segment: head.body.segment,
            });
        }
        by_segment.insert(head.body.segment, &head.body);
    }
    walk_verified_chain(archive, &by_segment)
}

/// [`recover_chain_from_archive`] for heads whose authority the CALLER has already established —
/// the ABI §8.8 product path, where a head is an `ArchiveHeadRecord` verified through
/// `daemon_vhc_proto::archive::ArchiveHeadRecord::authorize` (per-run key + certificate chain to
/// a genesis-trusted base) rather than an `AttestedHead` under the genesis `AuthorityConfig`.
/// This function judges NOTHING about authority; it performs the structural walk only (chain
/// contiguity, content re-hash, seal + scan). Passing unverified heads here forfeits the oracle's
/// authenticity claim.
///
/// # Errors
/// A typed [`ConsensusReplayError`] on a broken chain, a missing or content-mismatched segment,
/// or an unscannable segment.
pub fn recover_chain_from_verified_heads(
    archive: &RecordArchive,
    heads: &[super::archive::ChainHead],
) -> Result<RecoveredChain, ConsensusReplayError> {
    let mut by_segment: BTreeMap<u64, &super::archive::ChainHead> = BTreeMap::new();
    for head in heads {
        by_segment.insert(head.segment, head);
    }
    walk_verified_chain(archive, &by_segment)
}

/// The shared structural chain walk (steps 1–2 of the module contract, authority already
/// judged): dense ordinals from 0, `prev_hash` linkage, content re-hash, sealed-scan, record
/// recovery.
fn walk_verified_chain(
    archive: &RecordArchive,
    by_segment: &BTreeMap<u64, &super::archive::ChainHead>,
) -> Result<RecoveredChain, ConsensusReplayError> {
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
            if head.prev_hash != prev {
                return Err(ConsensusReplayError::ChainBroken {
                    segment,
                    detail: "prev_hash does not extend the previous head".into(),
                });
            }
        }
        let bytes =
            archive
                .fetch(&head.segment_hash)
                .ok_or(ConsensusReplayError::MissingSegment {
                    segment,
                    hash: head.segment_hash,
                })?;
        // A third party re-hashes what it fetched — the store is untrusted.
        if blake3_hash(bytes) != head.segment_hash {
            return Err(ConsensusReplayError::ContentMismatch { segment });
        }
        let scan = scan_bytes(bytes).map_err(|e| ConsensusReplayError::BadSegment {
            segment,
            detail: e.to_string(),
        })?;
        // The full head↔segment identity binding (shared verifier). The harness `ChainHead`
        // carries no chain scope (its `instance` is the attesting span, which an adopted
        // abandoned-tail head legitimately advances past the segment header's frozen
        // identity — defect 16), so the chain-scope comparison is skipped here.
        daemon_vhc_journal::verify_head_binding(
            &scan,
            &daemon_vhc_journal::HeadClaim {
                run_id: head.run_id,
                epoch: head.epoch,
                role: &head.role,
                module: head.module,
                chain_instance: None,
                segment: head.segment,
                prev_hash: head.prev_hash,
                records: head.records,
            },
        )
        .map_err(|e| ConsensusReplayError::BadSegment {
            segment,
            detail: e.to_string(),
        })?;
        for record in scan.records {
            if !matches!(record.body, Body::Seal(_)) {
                records.push(record);
            }
        }
        prev_hash = Some(head.segment_hash);
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
    let chain = recover_chain_from_archive(archive, heads)?;
    replay_consensus_over_chain(sandbox, chain, payloads)
}

/// [`replay_consensus_from_archive`] for heads the CALLER already authorized (the ABI §8.8
/// product path — see [`recover_chain_from_verified_heads`] for the trust contract).
///
/// # Errors
/// A typed [`ConsensusReplayError`]; an incomplete verification is never a pass.
pub fn replay_consensus_from_verified_archive(
    sandbox: &dyn CoordinatorSandbox,
    archive: &RecordArchive,
    heads: &[super::archive::ChainHead],
    payloads: &BTreeMap<Hash, Vec<u8>>,
) -> Result<ConsensusReplayReport, ConsensusReplayError> {
    let chain = recover_chain_from_verified_heads(archive, heads)?;
    replay_consensus_over_chain(sandbox, chain, payloads)
}

/// Steps 3–4 over an already-recovered chain: sandboxed re-derivation + digest re-verification
/// from the payloads alone.
fn replay_consensus_over_chain(
    sandbox: &dyn CoordinatorSandbox,
    chain: RecoveredChain,
    payloads: &BTreeMap<Hash, Vec<u8>>,
) -> Result<ConsensusReplayReport, ConsensusReplayError> {
    let count = chain.segments_verified;
    let records_recovered = chain.records.len() as u64;
    let capture = extract_consensus_capture(&chain.records)?;
    let initial = capture.initial.ok_or(ConsensusReplayError::NoSnapshot)?;

    // -- 3. re-derive: the pure tick over the inputs; archived records are the oracle ------------
    let oracle_records: Vec<Input> = capture
        .published
        .iter()
        .filter(|sm| matches!(sm.payload, VhcMessage::RoundRecord(_)))
        .cloned()
        .map(Input::Message)
        .collect();
    let replay = replay_from_state(
        sandbox,
        initial,
        capture.inputs.into_iter().chain(oracle_records),
    )?;

    // -- 4. digests from payloads alone: every committed entry + every set commitment ------------
    let (payload_entries_verified, set_commitments_verified) =
        verify_committed_payloads(replay.records.iter(), payloads)?;

    Ok(ConsensusReplayReport {
        segments_verified: count,
        records_recovered,
        replay,
        payload_entries_verified,
        set_commitments_verified,
    })
}

/// The digest-from-payloads re-verification (step 4 of the module contract), over ANY source of
/// round records — the sandboxed re-derivation above, or a production archive's published records
/// after the input-replay oracle proved them bit-reproducible. Every committed `(peer, hash,
/// size)` entry must re-verify against the supplied content-addressed bytes, and every record's
/// set commitment must recompute from those pairs alone (§6.4 I3/I6).
///
/// Returns `(payload entries verified, set commitments verified)`.
///
/// # Errors
/// The typed [`ConsensusReplayError::MissingPayload`] / [`ConsensusReplayError::PayloadMismatch`]
/// / [`ConsensusReplayError::SetCommitmentMismatch`] — an incomplete verification is never a pass.
pub fn verify_committed_payloads<'a>(
    records: impl Iterator<Item = &'a RoundRecord>,
    payloads: &BTreeMap<Hash, Vec<u8>>,
) -> Result<(u64, u64), ConsensusReplayError> {
    let mut payload_entries_verified = 0u64;
    let mut set_commitments_verified = 0u64;
    for record in records {
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
    Ok((payload_entries_verified, set_commitments_verified))
}

#[cfg(test)]
mod tests {
    use daemon_vhc_proto::{commit_set, to_canonical_vec, Hash, PeerId, Seed, StateDigest};
    use daemon_vhc_sdk_consensus::messages::{Digest, Locator, RoundRecord, VhcMessage};

    use super::super::record::{Body, EventRec, PublishRec, Record, RunHeader, SignedFrameRec};
    use super::{extract_wire_capture, records_are_wire_form};

    /// A §12.1 signed wire frame around a canonical `VhcMessage` payload: `[envelope, payload,
    /// sig]` — the outer structure the session writes, with fixture envelope/signature bytes (the
    /// capture never judges them; segment attestation carries the authenticity).
    fn wire_frame(msg: &VhcMessage) -> Vec<u8> {
        let payload = to_canonical_vec(msg).expect("encode payload");
        let v = ciborium::value::Value::Array(vec![
            ciborium::value::Value::Map(vec![(
                ciborium::value::Value::Text("seq".into()),
                ciborium::value::Value::Integer(1.into()),
            )]),
            ciborium::value::Value::Bytes(payload),
            ciborium::value::Value::Bytes(vec![0u8; 64]),
        ]);
        let mut out = Vec::new();
        ciborium::ser::into_writer(&v, &mut out).expect("encode frame");
        out
    }

    fn fixture_header() -> Box<RunHeader> {
        Box::new(RunHeader {
            run_id: Hash([0xAA; 32]),
            epoch: 0,
            role: "coordinator".into(),
            instance: 38,
            module: Hash([0xC0; 32]),
            abi: 2 << 16,
            worlds: std::collections::BTreeMap::new(),
            bridge: false,
            manifest: Vec::new(),
            config: vec![0x01],
            grants: vec![0x02],
            claim: None,
            resource_plan: None,
            resource_plan_hash: None,
            physical_estimate: None,
            physical_estimate_hash: None,
            aggregate_estimate: None,
            aggregate_estimate_hash: None,
            execution_grant: None,
            execution_grant_hash: None,
            channels: Vec::new(),
            device: Vec::new(),
            format: 1,
        })
    }

    /// The wire-form judgment is STRUCTURAL — the first frame-carrying record's CBOR shape — so
    /// a harness journal that also writes a run header (the archive-assembly fixture does) is
    /// still judged harness; only a frameless prefix falls back to the header.
    #[test]
    fn wire_form_is_judged_by_frame_shape_not_the_run_header() {
        let digest = VhcMessage::Digest(Digest {
            round: 1,
            digest: StateDigest([0xD1; 16]),
        });
        // Harness form: header + a tag-4 whose frame is the SDK SignedMessage (a CBOR map).
        let mut map = Vec::new();
        ciborium::ser::into_writer(
            &ciborium::value::Value::Map(vec![(
                ciborium::value::Value::Text("version".into()),
                ciborium::value::Value::Integer(1.into()),
            )]),
            &mut map,
        )
        .expect("encode map frame");
        let harness = [
            Record::new(0, Body::RunHeader(fixture_header())),
            Record::new(
                1,
                Body::Publish(PublishRec {
                    channel: 0,
                    seq: 0,
                    hash: Hash([0x44; 32]),
                    frame: map,
                }),
            ),
        ];
        assert!(!records_are_wire_form(&harness));

        // Production form: the same stream with a §12.1 `[envelope, payload, sig]` frame.
        let production = [
            Record::new(0, Body::RunHeader(fixture_header())),
            Record::new(
                1,
                Body::Publish(PublishRec {
                    channel: 0,
                    seq: 0,
                    hash: Hash([0x44; 32]),
                    frame: wire_frame(&digest),
                }),
            ),
        ];
        assert!(records_are_wire_form(&production));

        // A frameless prefix: only the header can testify.
        let prefix = [Record::new(0, Body::RunHeader(fixture_header()))];
        assert!(records_are_wire_form(&prefix));
        let empty: [Record; 0] = [];
        assert!(!records_are_wire_form(&empty));
        let events_only = [Record::new(
            0,
            Body::Event(EventRec {
                at: 0,
                frame: vec![0x82, 0x00, 0x00],
            }),
        )];
        assert!(!records_are_wire_form(&events_only));
    }

    /// The wire capture decodes tag-12 frames and tag-4 publishes through the `[envelope,
    /// payload, sig]` structure, keeps the header, skips structurally foreign frames, and
    /// projects the published `RoundRecord`s.
    #[test]
    fn wire_capture_decodes_frames_and_publishes_and_skips_foreign_bytes() {
        let peer = PeerId([0x11; 32]);
        let digest = VhcMessage::Digest(Digest {
            round: 3,
            digest: StateDigest([0xD1; 16]),
        });
        let record_msg = VhcMessage::RoundRecord(RoundRecord {
            round: 3,
            set: commit_set(&[]).commitment(),
            drops: Vec::new(),
            next_seed: Seed([0x5E; 32]),
            set_locator: Locator::StoreKey("record-set".into()),
            inline: Some(Vec::new()),
        });
        let records = [
            Record::new(0, Body::RunHeader(fixture_header())),
            Record::new(
                1,
                Body::SignedFrame(SignedFrameRec {
                    channel: 0,
                    seq: 7,
                    sender: Hash(peer.0),
                    frame: Some(wire_frame(&digest)),
                    evidence: None,
                }),
            ),
            // Structurally foreign evidence: never a capture input, never an error.
            Record::new(
                2,
                Body::SignedFrame(SignedFrameRec {
                    channel: 0,
                    seq: 8,
                    sender: Hash(peer.0),
                    frame: Some(vec![0xFF, 0x00]),
                    evidence: None,
                }),
            ),
            Record::new(
                3,
                Body::Publish(PublishRec {
                    channel: 0,
                    seq: 0,
                    hash: Hash([0x44; 32]),
                    frame: wire_frame(&record_msg),
                }),
            ),
        ];

        let capture = extract_wire_capture(&records);
        let header = capture.header.as_ref().expect("header kept");
        assert_eq!(
            (header.instance, header.config.as_slice()),
            (38, &[0x01][..])
        );
        assert_eq!(capture.frames.len(), 1, "the foreign frame is skipped");
        assert_eq!((capture.frames[0].sender, capture.frames[0].seq), (peer, 7));
        assert!(matches!(capture.frames[0].message, VhcMessage::Digest(d) if d.round == 3));
        assert_eq!(capture.published.len(), 1);
        let rounds = capture.round_records();
        assert_eq!(rounds.len(), 1);
        assert_eq!(rounds[0].round, 3);
    }
}
