// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `xtask vhc-replay` — the executable replay / archive-verdict command.
//!
//! Consumes an on-disk **archive directory** produced by (or pulled back from) a run and runs BOTH
//! existing oracle modes from `daemon-vhc-observe`, emitting a per-round, per-peer machine-readable
//! verdict:
//!
//! 1. **Consensus re-derivation (sandboxed).** The pinned coordinator module is driven inside the
//!    real host sandbox ([`SandboxedCoordinator`]) — consensus NEVER runs natively — over the
//!    archived driving inputs recovered from the sealed record segments; every archived
//!    `RoundRecord` must re-derive byte-identically, and every committed digest must recompute from
//!    the content-addressed payloads alone ([`replay_consensus_from_archive`], which internally runs
//!    the input-replay oracle `replay_from_state` and the payload/set-commitment re-verification).
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
//!   heads.cbor                        CBOR: [ { body: <chain-head>, sigs: [ {signer, sig} ] } ]
//!                                     — the attested sealed-chain heads (segment 0 .. N, contiguous)
//!   segments/<segment_hash_hex>.seg   the sealed record-archive segment bytes (content-addressed;
//!                                     the file stem is the segment's blake3 content address)
//!   payloads/<blake3_hex>.bin         the committed update-container payload objects, by content hash
//!   peers/<peerid_hex>.digests.cbor   CBOR: [ [round, <16-byte digest>] ] — that peer's per-round
//!                                     post-ingest det-state digests (the `--watch` transcript)
//! ```
//!
//! `envelope.cbor`, `coordinator.wasm`, `heads.cbor`, and `segments/` are REQUIRED (the consensus
//! oracle). `payloads/` is required for the digest-from-payloads re-verification. `peers/` is
//! optional — absent, the per-peer agreement section is empty (the consensus oracle still runs).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use daemon_vhc_observe::journal::archive::{ChainHead, RecordArchive};
use daemon_vhc_observe::{
    replay_consensus_from_archive, AttestedHead, AuthorityConfig, ConsensusReplayError, RecordSig,
    ReplayError, ReplicationPolicy, RetentionPolicy,
};
use daemon_vhc_proto::genesis::GenesisEnvelope;
use daemon_vhc_proto::{blake3_hash, from_canonical_slice, Hash, PeerId, Signature};
use daemon_vhc_session::replay_sandbox::SandboxedCoordinator;

/// The parsed `vhc-replay` inputs.
pub struct Args {
    pub archive: PathBuf,
    pub run: String,
    pub json: bool,
}

/// A `RecordSig` in serde-friendly form (the sdk `RecordSig` is not `Serialize`).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredSig {
    signer: PeerId,
    sig: Signature,
}

/// An attested head in serde-friendly form (the observe `AttestedHead` is not `Serialize` because
/// its `RecordSig`s are not) — `body` is the serde `ChainHead`, `sigs` the raw signer+signature.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredHead {
    body: ChainHead,
    sigs: Vec<StoredSig>,
}

impl StoredHead {
    fn into_attested(self) -> AttestedHead {
        AttestedHead {
            body: self.body,
            sigs: self
                .sigs
                .into_iter()
                .map(|s| RecordSig {
                    signer: s.signer,
                    sig: s.sig,
                })
                .collect(),
        }
    }
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
    let sandbox = SandboxedCoordinator::new(coord_wasm);

    // -- the attested sealed-chain heads ---------------------------------------------------------
    let heads_bytes = std::fs::read(dir.join("heads.cbor"))
        .with_context(|| format!("read {}/heads.cbor", dir.display()))?;
    let stored_heads: Vec<StoredHead> = from_canonical_slice(&heads_bytes)
        .map_err(|e| anyhow::anyhow!("decode heads.cbor: {e}"))?;
    let heads: Vec<AttestedHead> = stored_heads
        .into_iter()
        .map(StoredHead::into_attested)
        .collect();
    anyhow::ensure!(!heads.is_empty(), "heads.cbor carries no attested heads");

    // -- the record archive: publish every sealed segment + ingest every head -------------------
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
    for head in &heads {
        // A fork or unauthoritative head here is a hard input error (the archive is corrupt).
        archive
            .ingest_head(head.clone())
            .map_err(|e| anyhow::anyhow!("ingest head: {e}"))?;
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

    // -- oracle mode: sandboxed consensus re-derivation + digest-from-payloads -------------------
    let consensus = match replay_consensus_from_archive(&sandbox, &archive, &heads, &payloads) {
        Ok(report) => ConsensusVerdict {
            agree: true,
            segments_verified: report.segments_verified,
            records_recovered: report.records_recovered,
            rounds_verified: report.replay.rounds_verified,
            payload_entries_verified: report.payload_entries_verified,
            first_divergence_round: None,
            detail: None,
        },
        Err(e) => {
            let round = divergence_round(&e);
            ConsensusVerdict {
                agree: false,
                segments_verified: 0,
                records_recovered: 0,
                rounds_verified: 0,
                payload_entries_verified: 0,
                first_divergence_round: round,
                detail: Some(e.to_string()),
            }
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
