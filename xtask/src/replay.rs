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
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use serde::Serialize;

use daemon_vhc_net::{ContentStore, FsContentStore};
use daemon_vhc_observe::journal::archive::{ChainHead, RecordArchive};
use daemon_vhc_observe::journal::Record;
use daemon_vhc_observe::{
    coordinator_lineage, envelope_trusted_bases, records_are_wire_form,
    replay_consensus_from_verified_archive, semantic_fold, verify_chains,
    verify_committed_payloads, AuthorityConfig, ConsensusReplayError, ReplayError,
    ReplicationPolicy, RetentionPolicy, SemanticFoldError,
};
use daemon_vhc_proto::archive::ArchiveHeadRecord;
use daemon_vhc_proto::genesis::GenesisEnvelope;
use daemon_vhc_proto::{blake3_hash, from_canonical_slice, to_canonical_vec, Hash, VerifiedChain};
use daemon_vhc_session::reconstruct::{
    certify_lineage, ClosureClass, LineageReplayReport, ReconstructError, ReconstructSpec,
};
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
    /// Coordinator chains in the certified lineage (1 = the degenerate single-chain shape).
    chains: u64,
    /// Incarnation spans the lineage replay drove.
    spans: u64,
    /// Reason-2 seams whose anchor + kind-3 bindings verified.
    seams: u64,
    /// How the lineage closed: `terminal(<outcome>)` (the recorded terminal was reproduced —
    /// a COMPLETE record of a finished run) or `prefix` (a verified sealed prefix). `None` on
    /// the harness lane (no closure question — the harness archive is its own whole story).
    closure: Option<String>,
    /// Per-span replay facts (wire lane only).
    span_facts: Vec<SpanVerdict>,
    /// Per-seam validation facts (wire lane only).
    seam_facts: Vec<SeamVerdict>,
    /// Sealed segments walked (chain-verified, content-re-hashed) on a green run.
    segments_verified: u64,
    /// Records recovered across them on a green run.
    records_recovered: u64,
    /// UNIQUE `RoundRecord`s verified (identical replay-forward duplicates deduplicate).
    rounds_verified: u64,
    /// Identical duplicate `RoundRecord`s collapsed by the semantic fold.
    round_duplicates_deduped: u64,
    /// Committed `(peer, hash)` payload entries re-verified on a green run.
    payload_entries_verified: u64,
    /// `RoundRecord` set commitments recomputed from payloads alone.
    set_commitments_verified: u64,
    /// The failing stage's taxonomy class on a red run (`verify`, `segment`, `transport`,
    /// `replay`, `seam-anchor`, `kind-3`, `equivocation`, `continuity`, `digest-conflict`,
    /// `payload`).
    failure_stage: Option<String>,
    /// The first round at which re-derivation / digest re-verification diverged (red runs only).
    first_divergence_round: Option<u64>,
    /// The typed reason on a red run (carries the first-failing chain/span/round coordinates).
    detail: Option<String>,
}

/// One incarnation span's replay facts.
#[derive(Debug, Serialize)]
struct SpanVerdict {
    /// The span's index in the lineage replay.
    index: usize,
    /// The incarnation id the span's tag-0 header carries.
    instance: u64,
    /// The tag-13 instantiation reason (0 initial, 1 trap-restart, 2 upgrade-activation).
    reason: u64,
    /// Records in the span.
    records: usize,
    /// Recorded publishes the decision gate verified.
    publishes: usize,
    /// How the span's replay ended.
    end: String,
}

/// One reason-2 seam's validation facts.
#[derive(Debug, Serialize)]
struct SeamVerdict {
    /// The SUCCESSOR span's index.
    span: usize,
    /// The anchoring tag-10 manifest byte length (== the predecessor export's manifest).
    anchor_manifest_len: usize,
    /// Recorded kind-3 restore read-backs verified against their staged identities.
    kind3_checked: usize,
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

    // -- the lineage's records, straight from the archive dir (hash-verified here; the kernel
    //    re-runs the FULL head↔segment binding through its own recover step) — the journal-form
    //    detector and the semantic fold both read these.
    let mut lineage_records: Vec<Record> = Vec::new();
    for chain in &lineage {
        for head in &chain.heads {
            let path = dir
                .join("segments")
                .join(format!("{}.seg", head.body.segment_hash.to_hex()));
            let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            anyhow::ensure!(
                blake3_hash(&bytes) == head.body.segment_hash,
                "segment {} does not hash-match its attested head",
                path.display()
            );
            let scan = daemon_vhc_observe::scan_bytes(&bytes)
                .map_err(|e| anyhow::anyhow!("scan segment {}: {e}", head.body.segment))?;
            lineage_records.extend(scan.records);
        }
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

    // -- oracle mode: consensus re-verification, through the journal form the lineage carries ----
    //
    // PRODUCTION archives journal the §12.1 wire forms and the FULL delivered-event stream, and
    // certify through the lineage-certification kernel (single- AND multi-chain: the session's
    // state-carrying seam replay + decision gate, the observe semantic fold, payload closure).
    // HARNESS-form archives (SDK types journaled directly, event-driven clock) re-derive through
    // the frames-only sandboxed consensus oracle, unchanged.
    let consensus = if records_are_wire_form(&lineage_records) {
        wire_lineage_verdict(
            dir,
            run_id,
            &trusted,
            &coordinator_role,
            &envelope,
            &coord_wasm,
            &lineage,
            &lineage_records,
            &payloads,
        )
    } else {
        // The harness archive is single-chain by construction (no restart succession rides the
        // SDK harness journals).
        let [chain] = lineage.as_slice() else {
            bail!(
                "harness-form archive spans {} chains — not a shape the harness oracle replays",
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
        // The record archive: publish every sealed segment (content-addressed, re-hashed).
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
        match replay_consensus_from_verified_archive(&sandbox, &archive, &heads, &payloads) {
            Ok(report) => ConsensusVerdict {
                agree: true,
                chains: 1,
                spans: 0,
                seams: 0,
                closure: None,
                span_facts: Vec::new(),
                seam_facts: Vec::new(),
                segments_verified: report.segments_verified,
                records_recovered: report.records_recovered,
                rounds_verified: report.replay.rounds_verified,
                round_duplicates_deduped: 0,
                payload_entries_verified: report.payload_entries_verified,
                set_commitments_verified: report.set_commitments_verified,
                failure_stage: None,
                first_divergence_round: None,
                detail: None,
            },
            Err(e) => red_verdict(CertifyFailure {
                stage: "harness",
                round: divergence_round(&e),
                detail: e.to_string(),
            }),
        }
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

/// A typed certification failure: the taxonomy class, the first-failing round where one is
/// knowable, and the full detail (which carries chain/span/ordinal coordinates).
struct CertifyFailure {
    stage: &'static str,
    round: Option<u64>,
    detail: String,
}

impl From<ReconstructError> for CertifyFailure {
    fn from(e: ReconstructError) -> Self {
        let stage = match &e {
            ReconstructError::Verify(_) => "verify",
            ReconstructError::Segment { .. } => "segment",
            ReconstructError::Transport { .. } => "transport",
            ReconstructError::Sandbox(_) => "replay",
            ReconstructError::SeamAnchor { .. } => "seam-anchor",
            ReconstructError::Kind3 { .. } => "kind-3",
        };
        Self {
            stage,
            round: None,
            detail: e.to_string(),
        }
    }
}

impl From<SemanticFoldError> for CertifyFailure {
    fn from(e: SemanticFoldError) -> Self {
        let (stage, round) = match &e {
            SemanticFoldError::Equivocation { round } => ("equivocation", Some(*round)),
            SemanticFoldError::Continuity { .. } => ("continuity", None),
            SemanticFoldError::DigestConflict { round, .. } => ("digest-conflict", Some(*round)),
            SemanticFoldError::Codec(_) => ("codec", None),
        };
        Self {
            stage,
            round,
            detail: e.to_string(),
        }
    }
}

/// A red consensus verdict carrying its typed reason.
fn red_verdict(f: CertifyFailure) -> ConsensusVerdict {
    ConsensusVerdict {
        agree: false,
        chains: 0,
        spans: 0,
        seams: 0,
        closure: None,
        span_facts: Vec::new(),
        seam_facts: Vec::new(),
        segments_verified: 0,
        records_recovered: 0,
        rounds_verified: 0,
        round_duplicates_deduped: 0,
        payload_entries_verified: 0,
        set_commitments_verified: 0,
        failure_stage: Some(f.stage.to_string()),
        first_divergence_round: f.round,
        detail: Some(f.detail),
    }
}

/// The PRODUCTION-archive consensus verdict, through the lineage-certification kernel: the
/// session's state-carrying seam replay + per-span decision gate ([`certify_lineage`]), the
/// observe semantic fold (equivocation / continuity / digest conflicts), and payload closure
/// over the deduplicated round records.
#[allow(clippy::too_many_arguments)] // the archive's verified parts, threaded once
fn wire_lineage_verdict(
    dir: &Path,
    run_id: Hash,
    trusted: &[daemon_vhc_proto::PeerId],
    coordinator_role: &str,
    envelope: &GenesisEnvelope,
    coord_wasm: &[u8],
    lineage: &[&VerifiedChain],
    lineage_records: &[Record],
    payloads: &BTreeMap<Hash, Vec<u8>>,
) -> ConsensusVerdict {
    match wire_lineage_certify(
        dir,
        run_id,
        trusted,
        coordinator_role,
        envelope,
        coord_wasm,
        lineage,
        lineage_records,
        payloads,
    ) {
        Ok(v) => v,
        Err(f) => red_verdict(f),
    }
}

#[allow(clippy::too_many_arguments)] // the archive's verified parts, threaded once
fn wire_lineage_certify(
    dir: &Path,
    run_id: Hash,
    trusted: &[daemon_vhc_proto::PeerId],
    coordinator_role: &str,
    envelope: &GenesisEnvelope,
    coord_wasm: &[u8],
    lineage: &[&VerifiedChain],
    lineage_records: &[Record],
    payloads: &BTreeMap<Hash, Vec<u8>>,
) -> Result<ConsensusVerdict, CertifyFailure> {
    let replay_fail = |detail: String| CertifyFailure {
        stage: "replay",
        round: None,
        detail,
    };

    // -- 1. the schema-free kernel replay (session): spans, seams, decisions, closure ------------
    let heads: Vec<ArchiveHeadRecord> = lineage
        .iter()
        .flat_map(|c| c.heads.iter().cloned())
        .collect();
    let segments_total = heads.len() as u64;
    let incarnation = lineage.iter().map(|c| c.chain_instance).max().unwrap_or(0) + 1;
    let role_entry = &envelope.roles[coordinator_role];
    let config = to_canonical_vec(&role_entry.config)
        .map_err(|e| replay_fail(format!("role config: {e}")))?;
    let engine = daemon_vhc_host::Worker::new(daemon_vhc_host::EngineConfig::default())
        .map_err(|e| replay_fail(format!("engine: {e}")))?;
    let linked = daemon_vhc_host::linked_worlds(&engine, coord_wasm)
        .map_err(|e| replay_fail(format!("linked worlds: {e}")))?;
    let grants =
        daemon_vhc_proto::GrantsDoc::author(&linked, &role_entry.grants).to_canonical_bytes();

    // Stage the segments into a throwaway FLAT content store (the kernel fetches attested
    // segments content-addressed; the archive names them `<hex>.seg`).
    let staged = std::env::temp_dir().join(format!("vhc-replay-certify-{}", std::process::id()));
    std::fs::create_dir_all(&staged)
        .map_err(|e| replay_fail(format!("create staging dir: {e}")))?;
    let stage_result = (|| -> Result<LineageReplayReport, CertifyFailure> {
        for (stem, path) in read_hex_stem_dir(&dir.join("segments"), ".seg")
            .map_err(|e| replay_fail(format!("segments dir: {e}")))?
        {
            std::fs::copy(&path, staged.join(stem))
                .map_err(|e| replay_fail(format!("stage {}: {e}", path.display())))?;
        }
        let store: Arc<dyn ContentStore> = Arc::new(
            FsContentStore::open(&staged).map_err(|e| replay_fail(format!("open store: {e}")))?,
        );
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| replay_fail(format!("tokio runtime: {e}")))?;
        rt.block_on(certify_lineage(
            ReconstructSpec {
                heads,
                run_id,
                trusted: trusted.to_vec(),
                role: coordinator_role.to_string(),
                run_label: run_id.to_hex(),
                journal_root: None,
                module: coord_wasm.to_vec(),
                config,
                grants,
                incarnation,
                restore: None,
                sidecar_key: None,
                deadline_ms: 120_000,
            },
            store,
        ))
        .map_err(CertifyFailure::from)
    })();
    let _ = std::fs::remove_dir_all(&staged);
    let report = stage_result?;

    // -- 2. the semantic fold (observe): equivocation, continuity, digest conflicts --------------
    let fold = semantic_fold(lineage_records).map_err(CertifyFailure::from)?;

    // -- 3. payload closure over the deduplicated round records ----------------------------------
    let (payload_entries_verified, set_commitments_verified) =
        verify_committed_payloads(fold.records.iter(), payloads).map_err(|e| CertifyFailure {
            stage: "payload",
            round: divergence_round(&e),
            detail: e.to_string(),
        })?;

    Ok(ConsensusVerdict {
        agree: true,
        chains: lineage.len() as u64,
        spans: report.spans.len() as u64,
        seams: report.seams.len() as u64,
        closure: Some(match report.closure {
            ClosureClass::Terminal { outcome } => format!("terminal({outcome})"),
            ClosureClass::Prefix => "prefix".to_string(),
        }),
        span_facts: report
            .spans
            .iter()
            .map(|s| SpanVerdict {
                index: s.index,
                instance: s.instance,
                reason: s.reason,
                records: s.records,
                publishes: s.publishes,
                end: s.end.clone(),
            })
            .collect(),
        seam_facts: report
            .seams
            .iter()
            .map(|s| SeamVerdict {
                span: s.span,
                anchor_manifest_len: s.anchor_manifest_len,
                kind3_checked: s.kind3_checked,
            })
            .collect(),
        segments_verified: segments_total,
        records_recovered: lineage_records.len() as u64,
        rounds_verified: fold.records.len() as u64,
        round_duplicates_deduped: fold.duplicates_deduped,
        payload_entries_verified,
        set_commitments_verified,
        failure_stage: None,
        first_divergence_round: None,
        detail: None,
    })
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
        if let Some(closure) = &v.consensus.closure {
            let _ = writeln!(s, "  closure            : {closure}");
            let _ = writeln!(
                s,
                "  lineage            : {} chains, {} spans, {} seams",
                v.consensus.chains, v.consensus.spans, v.consensus.seams
            );
            for seam in &v.consensus.seam_facts {
                let _ = writeln!(
                    s,
                    "    seam -> span {:<3} anchor manifest {} B, kind-3 checked {}",
                    seam.span, seam.anchor_manifest_len, seam.kind3_checked
                );
            }
        }
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
        let _ = writeln!(
            s,
            "  rounds verified    : {} unique ({} duplicate replay-forward publishes deduped)",
            v.consensus.rounds_verified, v.consensus.round_duplicates_deduped
        );
        let _ = writeln!(
            s,
            "  payload entries    : {}",
            v.consensus.payload_entries_verified
        );
        let _ = writeln!(
            s,
            "  set commitments    : {}",
            v.consensus.set_commitments_verified
        );
    } else {
        let _ = writeln!(s, "  verdict            : DISAGREE");
        if let Some(stage) = &v.consensus.failure_stage {
            let _ = writeln!(s, "  failing stage      : {stage}");
        }
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
