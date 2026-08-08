// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `xtask vhc-reconstruct-probe` — the archive-recovery forensic / audit command.
//!
//! Answers, over an assembled archive directory (the `vhc-replay` layout), the questions a
//! recovery post-mortem asks BEFORE anyone touches a live box:
//!
//! 1. **Chain topology**: every verified journal chain in `heads.cbor` (role, base identity,
//!    chain instance, head count, predecessor link) and the coordinator lineage the production
//!    reconstruction would select — a lineage that begins mid-history (its founding span
//!    instantiated `reason 2`, migrating from a capture the archive does not carry) is visible
//!    here as such.
//! 2. **Span structure**: each chain's record stream split at incarnation seams — instantiation
//!    reasons, read-back kinds (inline vs sidecar-referenced, with the content addresses a
//!    replay must resolve), publish/terminal counts.
//! 3. **Reconstructability**: the REAL `reconstruct_coordinator` (the §8.8 production path the
//!    worker runs at a recovery join) over the archive alone — no local journal home, no
//!    sidecar key — reporting the exported capture's section content addresses on success and
//!    the typed refusal on failure. `--through` truncates the lineage to a prefix, which
//!    isolates WHICH span stops resolving (e.g. defect 16's missing migration-capture bytes).
//!
//! Read-only over the archive: segments are staged into a throwaway flat content store
//! (`$TMPDIR/…`, removed on exit) because the production reconstruction fetches attested
//! segments content-addressed.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use daemon_vhc_journal::{scan_bytes, Body, Record};
use daemon_vhc_net::{ContentStore, FsContentStore};
use daemon_vhc_proto::det_state::CkptDocSection;
use daemon_vhc_proto::genesis::GenesisEnvelope;
use daemon_vhc_proto::{
    blake3_hash, coordinator_lineage, envelope_trusted_bases, from_canonical_slice,
    to_canonical_vec, verify_chains, ArchiveHeadRecord, VerifiedChain,
};
use daemon_vhc_session::reconstruct::{reconstruct_coordinator, ReconstructSpec};

/// The command's arguments (mirrors the clap surface in `main.rs`).
pub struct Args {
    /// The archive directory (the `vhc-replay` layout).
    pub archive: PathBuf,
    /// The run id (genesis hash hex) the archive belongs to.
    pub run: String,
    /// Reconstruct only the lineage prefix up to and including this chain instance.
    pub through: Option<u64>,
    /// Skip the reconstruction (topology + span dump only).
    pub no_reconstruct: bool,
}

pub fn run(args: Args) -> Result<()> {
    let dir = &args.archive;
    anyhow::ensure!(dir.is_dir(), "archive {} is not a directory", dir.display());

    // -- envelope: run id, trusted bases, the pinned coordinator module ---------------------------
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
    let trusted = envelope_trusted_bases(&envelope);
    anyhow::ensure!(
        !trusted.is_empty(),
        "envelope names no genesis-trusted base identities"
    );
    let coordinator_role = envelope
        .roles
        .keys()
        .find(|r| r.contains("coordinator"))
        .cloned()
        .context("envelope names no coordinator role")?;
    let coord_wasm = std::fs::read(dir.join("coordinator.wasm"))
        .with_context(|| format!("read {}/coordinator.wasm", dir.display()))?;

    // -- heads: verify every chain, print the full topology --------------------------------------
    let heads_bytes = std::fs::read(dir.join("heads.cbor"))
        .with_context(|| format!("read {}/heads.cbor", dir.display()))?;
    let records: Vec<ArchiveHeadRecord> = from_canonical_slice(&heads_bytes)
        .map_err(|e| anyhow::anyhow!("decode heads.cbor: {e}"))?;
    println!("run {}", run_id.to_hex());
    println!("head records: {}", records.len());
    let chains = verify_chains(&run_id, &trusted, records).context("verify head records")?;
    println!("verified chains: {}", chains.len());
    for chain in &chains {
        println!(
            "  role={} base={} instance={} heads={} predecessor={} terminal={}",
            chain.role,
            hex8(&chain.base.0),
            chain.chain_instance,
            chain.heads.len(),
            chain
                .predecessor
                .map_or_else(|| "none (founding)".into(), |h| hex8(&h.0)),
            hex8(&chain.terminal_address.0),
        );
    }

    let lineage =
        coordinator_lineage(&chains, &coordinator_role).context("order coordinator lineage")?;
    println!(
        "\ncoordinator lineage ({}): {}",
        coordinator_role,
        lineage
            .iter()
            .map(|c| c.chain_instance.to_string())
            .collect::<Vec<_>>()
            .join(" -> ")
    );

    // -- span dump: every coordinator-lineage chain's record structure ---------------------------
    let segments_dir = dir.join("segments");
    for chain in &lineage {
        println!(
            "\nchain instance {} ({} heads):",
            chain.chain_instance,
            chain.heads.len()
        );
        let mut all: Vec<Record> = Vec::new();
        for head in &chain.heads {
            let path = segments_dir.join(format!("{}.seg", head.body.segment_hash.to_hex()));
            let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            anyhow::ensure!(
                blake3_hash(&bytes) == head.body.segment_hash,
                "segment {} does not hash-match its attested head",
                path.display()
            );
            let scan = scan_bytes(&bytes)
                .map_err(|e| anyhow::anyhow!("scan segment {}: {e}", head.body.segment))?;
            all.extend(scan.records);
        }
        dump_spans(&all);
    }

    if args.no_reconstruct {
        return Ok(());
    }

    // -- reconstruct: the production §8.8 path over the archive alone ----------------------------
    // The lineage prefix under probe: all chains, or `--through`'s prefix.
    let probe: Vec<&&VerifiedChain> = match args.through {
        None => lineage.iter().collect(),
        Some(through) => {
            let mut prefix: Vec<&&VerifiedChain> = Vec::new();
            for chain in &lineage {
                prefix.push(chain);
                if chain.chain_instance == through {
                    break;
                }
            }
            anyhow::ensure!(
                prefix.last().is_some_and(|c| c.chain_instance == through),
                "--through {through} is not in the lineage ({})",
                lineage
                    .iter()
                    .map(|c| c.chain_instance.to_string())
                    .collect::<Vec<_>>()
                    .join(" -> ")
            );
            prefix
        }
    };
    let probe_heads: Vec<ArchiveHeadRecord> =
        probe.iter().flat_map(|c| c.heads.iter().cloned()).collect();
    let incarnation = probe.iter().map(|c| c.chain_instance).max().unwrap_or(0) + 1;

    // Stage the segments into a throwaway FLAT content store (the reconstruction fetches
    // attested segments content-addressed; the archive names them `<hex>.seg`).
    let staged = tempdir_for_probe()?;
    for entry in std::fs::read_dir(&segments_dir)
        .with_context(|| format!("read dir {}", segments_dir.display()))?
    {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(stem) = name.strip_suffix(".seg") {
            std::fs::copy(&path, staged.join(stem))
                .with_context(|| format!("stage {}", path.display()))?;
        }
    }
    let store: Arc<dyn ContentStore> =
        Arc::new(FsContentStore::open(&staged).map_err(|e| anyhow::anyhow!("open store: {e}"))?);

    let role_entry = &envelope.roles[&coordinator_role];
    let config =
        to_canonical_vec(&role_entry.config).map_err(|e| anyhow::anyhow!("role config: {e}"))?;
    let engine = daemon_vhc_host::Worker::new(daemon_vhc_host::EngineConfig::default())
        .map_err(|e| anyhow::anyhow!("engine: {e}"))?;
    let linked = daemon_vhc_host::linked_worlds(&engine, &coord_wasm)
        .map_err(|e| anyhow::anyhow!("linked worlds: {e}"))?;
    let grants =
        daemon_vhc_proto::GrantsDoc::author(&linked, &role_entry.grants).to_canonical_bytes();

    println!(
        "\nreconstructing lineage prefix [{}] (archive-only: no journal home, no sidecar key)…",
        probe
            .iter()
            .map(|c| c.chain_instance.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let rt = tokio::runtime::Runtime::new().context("tokio runtime")?;
    let outcome = rt.block_on(reconstruct_coordinator(
        ReconstructSpec {
            heads: probe_heads,
            run_id,
            trusted,
            role: coordinator_role,
            run_label: run_id.to_hex(),
            journal_root: None,
            module: coord_wasm,
            config,
            grants,
            incarnation,
            restore: None,
            sidecar_key: None,
            deadline_ms: 120_000,
        },
        store,
    ));
    let _ = std::fs::remove_dir_all(&staged);

    match outcome {
        Ok(capture) => {
            println!("RECONSTRUCTED. exported capture:");
            println!(
                "  manifest: {} B (blake3 {})",
                capture.manifest.len(),
                hex8(&blake3_hash(&capture.manifest).0)
            );
            for section in &capture.sections {
                match section {
                    CkptDocSection::Inline(name, bytes) => println!(
                        "  section {name}: inline {} B blake3 {}",
                        bytes.len(),
                        blake3_hash(bytes).to_hex()
                    ),
                    CkptDocSection::ByRef(name, fref) => {
                        println!("  section {name}: by-ref {fref:?}");
                    }
                }
            }
            Ok(())
        }
        Err(e) => {
            println!("REFUSED (typed): {e}");
            bail!("reconstruction refused: {e}");
        }
    }
}

/// Split a chain's records at tag-0 seams and print each span's structure.
fn dump_spans(records: &[Record]) {
    let mut span = -1i64;
    let mut counts: BTreeMap<&'static str, u64> = BTreeMap::new();
    let flush = |span: i64, counts: &mut BTreeMap<&'static str, u64>| {
        if span >= 0 && !counts.is_empty() {
            let line = counts
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!("      {line}");
            counts.clear();
        }
    };
    for record in records {
        match &record.body {
            Body::RunHeader(h) => {
                flush(span, &mut counts);
                span += 1;
                println!(
                    "    span {span}: header ord={} instance={} epoch={} module={}",
                    record.ord,
                    h.instance,
                    h.epoch,
                    hex8(&h.module.0)
                );
            }
            Body::Instantiation(inst) => {
                println!(
                    "      instantiation ord={} counter={} reason={} ({})",
                    record.ord,
                    inst.counter,
                    inst.reason,
                    match inst.reason {
                        0 => "initial",
                        1 => "trap-restart",
                        2 => "migration",
                        _ => "?",
                    }
                );
            }
            Body::ReadBack(r) => match (&r.value, &r.sidecar) {
                (Some(v), _) => println!(
                    "      read-back ord={} kind={} inline {} B",
                    record.ord,
                    r.kind,
                    v.len()
                ),
                (None, Some(sref)) => println!(
                    "      read-back ord={} kind={} SIDECAR {} ({} B, seg {})",
                    record.ord,
                    r.kind,
                    sref.hash.to_hex(),
                    sref.size,
                    sref.seg
                ),
                (None, None) => {
                    println!("      read-back ord={} kind={} empty", record.ord, r.kind)
                }
            },
            Body::Terminal(t) => println!(
                "      terminal ord={} kind={} outcome={:?} trap={:?}",
                record.ord, t.kind, t.outcome, t.trap
            ),
            Body::Snapshot(_) => println!("      snapshot ord={}", record.ord),
            Body::Event(_) => *counts.entry("events").or_default() += 1,
            Body::Publish(_) => *counts.entry("publishes").or_default() += 1,
            Body::SignedFrame(_) => *counts.entry("signed-frames").or_default() += 1,
            Body::Clock(_) => *counts.entry("clocks").or_default() += 1,
            Body::TimerArm(_) => *counts.entry("timer-arms").or_default() += 1,
            _ => *counts.entry("other").or_default() += 1,
        }
    }
    flush(span, &mut counts);
}

fn hex8(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(4)
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

/// A throwaway staging dir under the system temp root (removed by the caller).
fn tempdir_for_probe() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("vhc-reconstruct-probe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir)
}
