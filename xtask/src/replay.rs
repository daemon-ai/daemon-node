// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `xtask vhc-replay` — the executable replay / archive-verdict command.
//!
//! Consumes an on-disk **archive directory** produced by (or pulled back from) a run and runs BOTH
//! existing oracle modes from `daemon-vhc-observe`, emitting a per-round, per-peer machine-readable
//! verdict:
//!
//! 1. **Consensus re-derivation (sandboxed).** The pinned coordinator module is driven inside the
//!    real host sandbox — consensus NEVER runs natively — over the archived driving inputs
//!    recovered from the sealed record segments; every archived `RoundRecord` must re-derive
//!    byte-identically, and every committed digest must recompute from the content-addressed
//!    payloads alone. The lane is picked by the journal's form
//!    ([`records_are_wire_form`]): a PRODUCTION archive carries the session's §12.1 wire frames
//!    and is re-driven through the host replay engine (`daemon_vhc_host::run::replay`) from the
//!    recorded event stream — an archive holds the sealed prefix of a possibly-live run, so a
//!    replay that consumes every recorded event and stops (`ScriptExhausted`) is as green as a
//!    recorded terminal outcome, provided every published decision was reproduced; a HARNESS
//!    archive records SDK types directly and replays through
//!    [`replay_consensus_from_verified_archive`] ([`SandboxedCoordinator`]).
//! 2. **Per-peer digest agreement.** Each peer's recorded per-round det-state digest is compared
//!    across peers; a round where any peer's digest differs from the round quorum is a disagreement,
//!    and the earliest such round is the first divergence.
//!
//! # Archive directory layout contract
//!
//! ```text
//! <archive>/
//!   envelope.cbor                     the frozen genesis envelope bytes (authority + the pinned
//!                                     coordinator module hash; its blake3 IS the run id)
//!   coordinator.wasm                  the genesis-pinned coordinator module (blake3 must equal the
//!                                     envelope's `coordinator.wasm` artifact hash)
//!   heads.cbor                        CBOR: [ <archive-head-record> ] — the run's published ABI
//!                                     §8.8 head records (every role's chains; the reader
//!                                     authorizes each against the envelope's genesis-trusted
//!                                     bases and selects the coordinator lineage itself)
//!   segments/<segment_hash_hex>.seg   the sealed record-archive segment bytes (content-addressed;
//!                                     the file stem is the segment's blake3 content address)
//!   payloads/<blake3_hex>.bin         the committed update-container payload objects, by content hash
//!   peers/<peerid_hex>.digests.cbor   CBOR: [ [round, <16-byte digest>] ] — that peer's per-round
//!                                     post-ingest det-state digests
//! ```
//!
//! `envelope.cbor`, `coordinator.wasm`, `heads.cbor`, and `segments/` are REQUIRED (the consensus
//! oracle). `payloads/` is required for the digest-from-payloads re-verification. `peers/` is
//! optional — absent, the per-peer agreement section is empty (the consensus oracle still runs).
//!
//! `xtask vhc-archive-pull` assembles this layout from a run's registry + content plane (the
//! product archive); [`daemon_vhc_observe::assemble_archive`] is the shared verifying core.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use daemon_vhc_observe::journal::archive::{ChainHead, RecordArchive};
use daemon_vhc_observe::journal::verifier::{
    run_replay, ExpectedDecision, GuestUnderReplay, PayloadSource, ReplayOutcome, ReplayPlan,
    ReplayStep,
};
use daemon_vhc_observe::{
    coordinator_lineage, envelope_trusted_bases, extract_wire_capture, records_are_wire_form,
    recover_chain_from_verified_heads, replay_consensus_from_verified_archive, verify_chains,
    verify_committed_payloads, AuthorityConfig, Body, ConsensusReplayError, RecoveredChain,
    ReplayError, ReplicationPolicy, RetentionPolicy,
};
use daemon_vhc_proto::archive::ArchiveHeadRecord;
use daemon_vhc_proto::genesis::GenesisEnvelope;
use daemon_vhc_proto::{blake3_hash, from_canonical_slice, Hash};
use daemon_vhc_session::replay_sandbox::SandboxedCoordinator;

/// The parsed `vhc-replay` inputs.
pub struct Args {
    pub archive: PathBuf,
    pub run: String,
    pub json: bool,
}

/// The consensus-oracle verdict (both re-derivation + digest-from-payloads).
#[derive(Debug, Serialize)]
struct ConsensusVerdict {
    /// Whether the archive re-verified fully (records re-derived + every digest recomputed).
    agree: bool,
    /// Sealed segments walked (chain-verified, content-re-hashed) on a green run.
    segments_verified: u64,
    /// Records recovered across them on a green run.
    records_recovered: u64,
    /// `RoundRecord`s re-derived byte-identically on a green run.
    rounds_verified: u64,
    /// Committed `(peer, hash)` payload entries re-verified on a green run.
    payload_entries_verified: u64,
    /// The first round at which re-derivation / digest re-verification diverged (red runs only).
    first_divergence_round: Option<u64>,
    /// The typed reason on a red run.
    detail: Option<String>,
}

/// One round's per-peer digest-agreement facts.
#[derive(Debug, Serialize)]
struct RoundAgreement {
    round: u64,
    /// The quorum (majority) digest hex for the round.
    quorum_digest: String,
    /// Each `(peer, digest hex, agrees)` observed for the round.
    peers: Vec<PeerRoundDigest>,
    /// Whether every peer that reported this round agreed with the quorum.
    agree: bool,
}

#[derive(Debug, Serialize)]
struct PeerRoundDigest {
    peer: String,
    digest: String,
    agree: bool,
}

/// The per-peer digest-agreement verdict (the G-2 transcript check).
#[derive(Debug, Serialize)]
struct PerPeerVerdict {
    /// The peers whose transcripts were read.
    peers: Vec<String>,
    /// Per-round agreement facts (ascending round).
    rounds: Vec<RoundAgreement>,
    /// The earliest round with any per-peer disagreement, if any.
    first_divergence_round: Option<u64>,
    /// Whether every observed round agreed across every reporting peer.
    agree: bool,
}

/// The full machine-readable verdict the command emits.
#[derive(Debug, Serialize)]
struct ReplayVerdict {
    run_id: String,
    /// The overall verdict: green iff the consensus oracle agreed AND (when peer transcripts are
    /// present) every per-peer round agreed.
    green: bool,
    consensus: ConsensusVerdict,
    per_peer: PerPeerVerdict,
}

fn read_hex_stem_dir(dir: &Path, ext: &str) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(stem) = name.strip_suffix(ext) {
            out.push((stem.to_string(), path));
        }
    }
    out.sort();
    Ok(out)
}

/// Load + verify an archive directory, running both oracle modes.
fn verify_archive(args: &Args) -> Result<ReplayVerdict> {
    let dir = &args.archive;
    anyhow::ensure!(dir.is_dir(), "archive {} is not a directory", dir.display());

    // -- the frozen genesis envelope: authority + the pinned coordinator module hash + run id ----
    let envelope_bytes = std::fs::read(dir.join("envelope.cbor"))
        .with_context(|| format!("read {}/envelope.cbor", dir.display()))?;
    let run_id = blake3_hash(&envelope_bytes);
    anyhow::ensure!(
        run_id.to_hex() == args.run.trim(),
        "envelope blake3 {} does not match --run {}",
        run_id.to_hex(),
        args.run.trim()
    );
    let envelope: GenesisEnvelope = from_canonical_slice(&envelope_bytes)
        .map_err(|e| anyhow::anyhow!("decode envelope: {e}"))?;
    let authority = AuthorityConfig::decode(&envelope.authority)
        .map_err(|e| anyhow::anyhow!("decode authority section: {e}"))?;
    let coord_artifact = envelope
        .artifacts
        .get("coordinator.wasm")
        .context("envelope carries no `coordinator.wasm` artifact")?;

    // -- the pinned coordinator module (blake3 must equal the envelope's artifact hash) ----------
    let coord_wasm = std::fs::read(dir.join("coordinator.wasm"))
        .with_context(|| format!("read {}/coordinator.wasm", dir.display()))?;
    let coord_hash = blake3_hash(&coord_wasm);
    anyhow::ensure!(
        coord_hash == coord_artifact.blake3,
        "coordinator.wasm blake3 {} does not match the genesis-pinned module {}",
        coord_hash.to_hex(),
        coord_artifact.blake3.to_hex()
    );
    let sandbox = SandboxedCoordinator::new(coord_wasm.clone());

    // -- the published archive-head records (ABI §8.8): authorize + select the lineage ----------
    let heads_bytes = std::fs::read(dir.join("heads.cbor"))
        .with_context(|| format!("read {}/heads.cbor", dir.display()))?;
    let records: Vec<ArchiveHeadRecord> = from_canonical_slice(&heads_bytes)
        .map_err(|e| anyhow::anyhow!("decode heads.cbor: {e}"))?;
    anyhow::ensure!(!records.is_empty(), "heads.cbor carries no head records");
    let trusted = envelope_trusted_bases(&envelope);
    anyhow::ensure!(
        !trusted.is_empty(),
        "envelope names no genesis-trusted base identities"
    );
    let chains =
        verify_chains(&run_id, &trusted, records).context("verify published head records")?;
    let coordinator_role = envelope
        .roles
        .keys()
        .find(|r| r.contains("coordinator"))
        .cloned()
        .context("envelope names no coordinator role")?;
    let lineage =
        coordinator_lineage(&chains, &coordinator_role).context("order coordinator lineage")?;
    // Cross-chain (restart-succession) replay lands with coordinator reconstruction; a single
    // uninterrupted chain is the C1/C1.5 shape this oracle certifies today.
    let [chain] = lineage.as_slice() else {
        bail!(
            "coordinator lineage spans {} chains (restart succession); cross-chain replay is \
             not yet certified — replay each chain against its own span",
            lineage.len()
        );
    };
    let heads: Vec<ChainHead> = chain
        .heads
        .iter()
        .map(|record| ChainHead {
            run_id: record.body.run_id,
            epoch: record.body.epoch,
            role: record.body.role.clone(),
            instance: record.body.instance,
            module: record.body.module,
            segment: record.body.segment,
            segment_hash: record.body.segment_hash,
            prev_hash: record.body.prev_hash,
            records: record.body.records,
        })
        .collect();

    // -- the record archive: publish every sealed segment (content-addressed, re-hashed) --------
    let mut archive = RecordArchive::new(
        authority,
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    for (_stem, path) in read_hex_stem_dir(&dir.join("segments"), ".seg")? {
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        archive
            .publish_segment(bytes)
            .map_err(|e| anyhow::anyhow!("publish segment {}: {e}", path.display()))?;
    }

    // -- the content-addressed payload objects ---------------------------------------------------
    let mut payloads: BTreeMap<Hash, Vec<u8>> = BTreeMap::new();
    for (stem, path) in read_hex_stem_dir(&dir.join("payloads"), ".bin")? {
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let hash = blake3_hash(&bytes);
        anyhow::ensure!(
            hash.to_hex() == stem,
            "payload {} does not match its content-addressed name {stem}",
            path.display()
        );
        payloads.insert(hash, bytes);
    }

    // -- oracle mode: consensus re-verification, through the journal form the chain carries ------
    //
    // HARNESS-form archives (SDK types journaled directly, event-driven clock) re-derive through
    // the frames-only sandboxed consensus oracle. PRODUCTION archives journal the §12.1 wire
    // forms and the FULL delivered-event stream (wall-clock timer ticks + completion order are
    // recorded nondeterministic inputs), so they re-verify through the §8.7 worker input-replay
    // engine: the pinned module is re-driven from the recorded events with `payload_get` answered
    // from the archive's content-addressed payload set, every recorded publish must reproduce
    // bit-exactly, and every committed digest must recompute from the payloads alone.
    let consensus = match recover_chain_from_verified_heads(&archive, &heads) {
        Ok(chain) if records_are_wire_form(&chain.records) => {
            wire_consensus_verdict(&coord_wasm, &chain, &payloads)
        }
        Ok(_) => {
            match replay_consensus_from_verified_archive(&sandbox, &archive, &heads, &payloads) {
                Ok(report) => ConsensusVerdict {
                    agree: true,
                    segments_verified: report.segments_verified,
                    records_recovered: report.records_recovered,
                    rounds_verified: report.replay.rounds_verified,
                    payload_entries_verified: report.payload_entries_verified,
                    first_divergence_round: None,
                    detail: None,
                },
                Err(e) => red_verdict(divergence_round(&e), e.to_string()),
            }
        }
        Err(e) => red_verdict(divergence_round(&e), e.to_string()),
    };

    // -- oracle mode: per-peer digest agreement (the G-2 transcript) -----------------------------
    let per_peer = per_peer_verdict(dir)?;

    let green = consensus.agree && per_peer.agree;
    Ok(ReplayVerdict {
        run_id: run_id.to_hex(),
        green,
        consensus,
        per_peer,
    })
}

/// Map a consensus-replay error to the round it first diverged at, where one is knowable.
fn divergence_round(e: &ConsensusReplayError) -> Option<u64> {
    match e {
        ConsensusReplayError::MissingPayload { round, .. }
        | ConsensusReplayError::PayloadMismatch { round, .. }
        | ConsensusReplayError::SetCommitmentMismatch { round } => Some(*round),
        ConsensusReplayError::Replay(ReplayError::Diverged(d)) => Some(d.round),
        _ => None,
    }
}

/// A red consensus verdict carrying its typed reason.
fn red_verdict(first_divergence_round: Option<u64>, detail: String) -> ConsensusVerdict {
    ConsensusVerdict {
        agree: false,
        segments_verified: 0,
        records_recovered: 0,
        rounds_verified: 0,
        payload_entries_verified: 0,
        first_divergence_round,
        detail: Some(detail),
    }
}

/// The PRODUCTION-archive consensus verdict: §8.7 input replay of the coordinator chain
/// (bit-exact decision reproduction) + digest-from-payloads re-verification of every published
/// `RoundRecord`.
fn wire_consensus_verdict(
    coord_wasm: &[u8],
    chain: &RecoveredChain,
    payloads: &BTreeMap<Hash, Vec<u8>>,
) -> ConsensusVerdict {
    match wire_input_replay(coord_wasm, chain, payloads) {
        Ok((rounds_verified, payload_entries_verified)) => ConsensusVerdict {
            agree: true,
            segments_verified: chain.segments_verified,
            records_recovered: chain.records.len() as u64,
            rounds_verified,
            payload_entries_verified,
            first_divergence_round: None,
            detail: None,
        },
        Err(e) => red_verdict(None, e.to_string()),
    }
}

/// Re-drive the archived coordinator journal through the §8.7 input-replay engine and re-verify
/// the committed digests from the payloads. Returns `(round records verified, payload entries
/// verified)`.
fn wire_input_replay(
    coord_wasm: &[u8],
    chain: &RecoveredChain,
    payloads: &BTreeMap<Hash, Vec<u8>>,
) -> Result<(u64, u64)> {
    use daemon_vhc_host::run::{replay, ReplayEnd, ReplayScript, RunIdentity};
    use daemon_vhc_host::{EngineConfig, Worker};

    let capture = extract_wire_capture(&chain.records);
    let header = capture
        .header
        .as_ref()
        .context("production journal carries no tag-0 run header")?;
    anyhow::ensure!(
        header.module.0 == *blake3::hash(coord_wasm).as_bytes(),
        "the journal's recorded module {} is not the genesis-pinned coordinator",
        header.module.to_hex()
    );

    // -- the replay script: the recorded nondeterministic inputs, split by answering mechanism --
    let mut script = ReplayScript::default();
    for record in &chain.records {
        match &record.body {
            Body::Event(e) => script.events.push_back((e.at, e.frame.clone())),
            Body::ReadBack(r) => {
                anyhow::ensure!(
                    r.sidecar.is_none(),
                    "record {}: sidecar-referenced read-back values are not supported by the \
                     archive replay (no sidecar key material rides a product archive)",
                    record.ord
                );
                let value = r.value.clone().unwrap_or_default();
                if r.kind >= 128 {
                    // The retired bridge's reserved journal kinds: never re-fed.
                } else if r.kind == u64::from(daemon_vhc_abi::READBACK_KIND_STREAM_BYTES) {
                    script.stream_bytes.push_back((r.src, value));
                } else if r.kind == u64::from(daemon_vhc_abi::READBACK_KIND_TENSOR_EXPORT) {
                    script.tensor_exports.push_back((r.src, value));
                } else if r.kind == u64::from(daemon_vhc_abi::READBACK_KIND_STATE_SEAL) {
                    script.state_seals.push_back((r.src, value));
                } else {
                    script.readbacks.push_back((r.src, r.kind, value));
                }
            }
            Body::Clock(c) => script.clocks.push_back(c.now),
            Body::TimerArm(t) => script.timer_arms.push_back((t.id, t.delay)),
            Body::TimerCancel(t) => script.timer_cancels.push_back((t.id, t.status)),
            Body::DeviceProfile(d) => script.device_profiles.push_back(d.profile.clone()),
            _ => {}
        }
    }
    script.identity = Some(RunIdentity {
        run_id: header.run_id.0,
        epoch: header.epoch,
        role: header.role.clone(),
        instance: header.instance,
        module: header.module.0,
    });
    // The §8.7 content-addressed re-fetch table: `payload_get` completions materialize their
    // buffers from the archive's payload objects.
    script.payloads = payloads.iter().map(|(h, b)| (h.0, b.clone())).collect();
    // The archive carries the SEALED prefix only (never a live unsealed tail), so the recorded
    // stream legitimately ends before the guest does — a prefix replay, not a divergence.
    script.stop_at_exhaustion = true;

    // -- re-drive the module; the journal's own publishes are the oracle -------------------------
    let worker =
        Worker::new(EngineConfig::default()).map_err(|e| anyhow::anyhow!("engine: {e}"))?;
    let replayed = replay(&worker, coord_wasm, &header.config, &header.grants, script)
        .map_err(|e| anyhow::anyhow!("input replay: {e}"))?;
    match &replayed.end {
        // A recorded outcome (the run terminated inside the archived prefix) or the clean end of
        // a prefix replay (the archive stops at its last sealed segment) both close the drive;
        // everything else is a real finding.
        ReplayEnd::Outcome(_) | ReplayEnd::ScriptExhausted => {}
        other => bail!("the replayed coordinator did not reach a recorded outcome: {other:?}"),
    }
    let plan = ReplayPlan::from_records(&chain.records);
    let mut guest = ReplayedDecisions::new(&replayed, &plan);
    match run_replay(&plan, &mut guest, &NoSidecars) {
        ReplayOutcome::Pass { .. } => {}
        other => bail!("input replay decision verdict: {other:?}"),
    }

    // -- digests from payloads alone: every committed entry + every set commitment ---------------
    let rounds = capture.round_records();
    let (payload_entries_verified, _) = verify_committed_payloads(rounds.iter().copied(), payloads)
        .map_err(|e| anyhow::anyhow!("digest re-verification: {e}"))?;
    Ok((rounds.len() as u64, payload_entries_verified))
}

/// The §8.7 `GuestUnderReplay` seam over the host replay's slice-attributed decisions: the run
/// already happened synchronously (`replay` — a replay can never block, every input is in the
/// journal), so delivery here walks the per-slice groups in recorded order. Ordinals are clerical
/// (journal bookkeeping the replay cannot know); decisions carry the recorded ord so equality
/// judges the substance: channel, seq, payload hash.
struct ReplayedDecisions {
    per_event: Vec<Vec<(u64, u64, [u8; 32])>>, // (channel, seq, payload_hash) per slice
    next_event: usize,
    expected_ords: Vec<u64>,
    next_ord: usize,
}

impl ReplayedDecisions {
    fn new(run: &daemon_vhc_host::run::ReplayedRun, plan: &ReplayPlan) -> Self {
        let mut per_event = vec![Vec::new(); run.events_delivered];
        for d in &run.decisions {
            per_event[d.event_index.min(run.events_delivered.saturating_sub(1))].push((
                d.channel,
                d.seq,
                d.payload_hash,
            ));
        }
        let expected_ords = plan
            .expected
            .iter()
            .map(|e| match e {
                ExpectedDecision::Publish { ord, .. } => *ord,
                _ => 0, // non_exhaustive: future decision kinds carry their own ords
            })
            .collect();
        Self {
            per_event,
            next_event: 0,
            expected_ords,
            next_ord: 0,
        }
    }
}

impl GuestUnderReplay for ReplayedDecisions {
    fn deliver_event(&mut self, _ord: u64, _at: u64, _frame: &[u8]) -> Vec<ExpectedDecision> {
        let group = self
            .per_event
            .get(self.next_event)
            .cloned()
            .unwrap_or_default();
        self.next_event += 1;
        group
            .into_iter()
            .map(|(channel, seq, hash)| {
                let ord = self
                    .expected_ords
                    .get(self.next_ord)
                    .copied()
                    .unwrap_or_default();
                self.next_ord += 1;
                ExpectedDecision::Publish {
                    ord,
                    channel,
                    seq,
                    hash: Hash(hash),
                }
            })
            .collect()
    }

    fn supply_import(&mut self, _step: &ReplayStep) {
        // Inputs were consumed by the synchronous host replay from the same journal.
    }
}

/// Product archives carry no sidecar key material; an oversize read-back would have been refused
/// while building the script, so a sidecar fetch here would itself be a finding.
struct NoSidecars;
impl PayloadSource for NoSidecars {
    fn fetch(
        &self,
        _sref: &daemon_vhc_observe::journal::record::SidecarRef,
        _ord: u64,
    ) -> Option<Vec<u8>> {
        None
    }
}

/// Fold the per-peer transcripts into per-round agreement facts + the first divergence round.
fn per_peer_verdict(dir: &Path) -> Result<PerPeerVerdict> {
    // peer hex -> (round -> digest hex).
    let mut by_peer: BTreeMap<String, BTreeMap<u64, [u8; 16]>> = BTreeMap::new();
    for (stem, path) in read_hex_stem_dir(&dir.join("peers"), ".digests.cbor")? {
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let rounds: Vec<(u64, [u8; 16])> =
            from_canonical_slice(&bytes).map_err(|e| anyhow::anyhow!("decode {stem}: {e}"))?;
        by_peer.entry(stem).or_default().extend(rounds);
    }

    let peers: Vec<String> = by_peer.keys().cloned().collect();
    // Gather every round any peer reported.
    let mut all_rounds: Vec<u64> = by_peer.values().flat_map(|m| m.keys().copied()).collect();
    all_rounds.sort_unstable();
    all_rounds.dedup();

    let hex16 = |d: &[u8; 16]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();

    let mut rounds = Vec::new();
    let mut first_divergence_round = None;
    for round in all_rounds {
        // The quorum digest: the most-reported digest for the round.
        let mut tally: BTreeMap<[u8; 16], usize> = BTreeMap::new();
        for m in by_peer.values() {
            if let Some(d) = m.get(&round) {
                *tally.entry(*d).or_default() += 1;
            }
        }
        let Some((quorum, _)) = tally.into_iter().max_by_key(|(_, n)| *n) else {
            continue;
        };
        let mut peer_facts = Vec::new();
        let mut round_agree = true;
        for (peer, m) in &by_peer {
            if let Some(d) = m.get(&round) {
                let agree = *d == quorum;
                round_agree &= agree;
                peer_facts.push(PeerRoundDigest {
                    peer: peer.clone(),
                    digest: hex16(d),
                    agree,
                });
            }
        }
        if !round_agree && first_divergence_round.is_none() {
            first_divergence_round = Some(round);
        }
        rounds.push(RoundAgreement {
            round,
            quorum_digest: hex16(&quorum),
            peers: peer_facts,
            agree: round_agree,
        });
    }

    let agree = first_divergence_round.is_none();
    Ok(PerPeerVerdict {
        peers,
        rounds,
        first_divergence_round,
        agree,
    })
}

/// Render the verdict as human-readable text.
fn render_text(v: &ReplayVerdict) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "vhc-replay verdict — run {}", v.run_id);
    let _ = writeln!(
        s,
        "overall              : {}",
        if v.green { "GREEN" } else { "RED" }
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "-- consensus oracle (input replay + sandboxed re-derivation) --"
    );
    if v.consensus.agree {
        let _ = writeln!(s, "  verdict            : AGREE");
        let _ = writeln!(
            s,
            "  segments verified  : {}",
            v.consensus.segments_verified
        );
        let _ = writeln!(
            s,
            "  records recovered  : {}",
            v.consensus.records_recovered
        );
        let _ = writeln!(s, "  rounds re-derived  : {}", v.consensus.rounds_verified);
        let _ = writeln!(
            s,
            "  payload entries    : {}",
            v.consensus.payload_entries_verified
        );
    } else {
        let _ = writeln!(s, "  verdict            : DISAGREE");
        if let Some(r) = v.consensus.first_divergence_round {
            let _ = writeln!(s, "  first divergence   : round {r}");
        }
        if let Some(d) = &v.consensus.detail {
            let _ = writeln!(s, "  detail             : {d}");
        }
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "-- per-peer digest agreement --");
    if v.per_peer.peers.is_empty() {
        let _ = writeln!(s, "  (no peer transcripts in the archive)");
    } else {
        let _ = writeln!(s, "  peers              : {}", v.per_peer.peers.len());
        for r in &v.per_peer.rounds {
            let _ = writeln!(
                s,
                "  round {:<4} {}  quorum {}",
                r.round,
                if r.agree { "AGREE   " } else { "DISAGREE" },
                r.quorum_digest
            );
            if !r.agree {
                for p in &r.peers {
                    if !p.agree {
                        let _ = writeln!(s, "      peer {} -> {}", p.peer, p.digest);
                    }
                }
            }
        }
        if let Some(r) = v.per_peer.first_divergence_round {
            let _ = writeln!(s, "  first divergence   : round {r}");
        } else {
            let _ = writeln!(s, "  verdict            : AGREE (all rounds)");
        }
    }
    s
}

/// The `vhc-replay` entry point.
pub fn run(args: Args) -> Result<()> {
    let verdict = verify_archive(&args)?;
    if args.json {
        let json = serde_json::to_string_pretty(&verdict).context("serialize verdict json")?;
        println!("{json}");
    } else {
        print!("{}", render_text(&verdict));
    }
    if !verdict.green {
        bail!("replay verdict is RED (see the per-round divergence above)");
    }
    Ok(())
}
