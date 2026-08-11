// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `xtask vhc-archive-pull` — assemble a run's PRODUCT archive (ABI §8.8) from the registry +
//! content plane into the §3.4 replay layout `vhc-replay` consumes.
//!
//! The pull is a pure reader with the same trust posture as any third party:
//!
//! 1. the run descriptor + frozen envelope come from the registry (`GET /runs/:id`, presigned
//!    envelope GET, blake3-verified against the descriptor's `envelope_hash`);
//! 2. the published archive-head records come from the untrusted archive slots
//!    (`GET /runs/:id/archive/heads`);
//! 3. every content object (sealed segments, committed payloads, the pinned coordinator module)
//!    is fetched by content address and re-hashed on arrival.
//!
//! All verification (head authorization against the genesis-trusted bases, the structural chain
//! fold, lineage ordering, payload enumeration, per-peer digest extraction) lives in the shared
//! core [`daemon_vhc_observe::assemble_archive`] — this command only moves bytes.
//!
//! xtask is maintainer dev tooling (not the shipped node); its egress rides the SSRF-safe
//! [`daemon_egress::EgressClient`] and fs writes go through the assembler.

use std::path::PathBuf;

use anyhow::{Context, Result};
use daemon_egress::{EgressClient, EgressConfig, Redirects};
use daemon_vhc_net::transport::ContentStore;
use daemon_vhc_net::{
    classify_egress, status_is_transient, ArchiveHeadStore, HttpArchiveHeadStore,
    HttpPresignClient, PresignClient, PresignOp, PresignRequest, PublishedArtifact, R2Store,
    RegistryClient, RunId, VhcNetError,
};
use daemon_vhc_observe::assemble_archive;
use daemon_vhc_proto::Hash;

/// The parsed `vhc-archive-pull` inputs (auth mirrors `publish-module`).
pub struct Args {
    /// The run id (genesis hash hex).
    pub run: String,
    /// The coordinator/gateway base, e.g. `https://…/api/v1/vhc`.
    pub base: String,
    /// `vhc:*`-scoped bearer token (gateway path), if any.
    pub bearer: Option<String>,
    /// Internal identity headers (direct-to-`apps/vhc` dev path): `(org_id, actor)`.
    pub internal: Option<(String, String)>,
    /// The output archive directory (the §3.4 layout).
    pub out: PathBuf,
}

fn egress() -> Result<EgressClient> {
    EgressClient::new(EgressConfig::default()).context("egress client")
}

/// One content-plane GET: the production `ContentStore` get (`payload/<hex>` — segments and
/// committed payloads), with the run-artifact key (`modules/<hex>.wasm`) as the fallback for the
/// genesis-pinned coordinator module. The assembler re-hashes every byte; this only moves them.
///
/// Reliability posture (REL-2a, reliability spec §3.1 — supersedes the C2 field patch): the
/// content leg inherits the store's bounded transient retry (`R2Store::get_content`, REL-2);
/// the module-fallback leg carries its own equivalent bounded loop below; and on a double
/// failure the CONTENT leg's error is reported as primary with the fallback outcome as an
/// annotation — the fallback's (expected) miss must never bury the real fault.
async fn fetch_object(
    r2: &R2Store<HttpPresignClient>,
    presign: &HttpPresignClient,
    egress: &EgressClient,
    run: &RunId,
    hash: &Hash,
) -> Result<Vec<u8>, String> {
    let first = match r2.get_content(hash).await {
        Ok(bytes) => return Ok(bytes),
        Err(e) => e.to_string(),
    };
    let path = PublishedArtifact::Module(*hash).object_path();
    match fetch_module_fallback(presign, egress, run, &path).await {
        Ok(bytes) => Ok(bytes),
        Err(fallback) => Err(format!("{first} (module fallback {path}: {fallback})")),
    }
}

/// How many times the module-fallback GET is attempted before its fault becomes the caller's.
const FALLBACK_ATTEMPTS: u32 = 4;

/// The fallback leg of [`fetch_object`] under the same REL-2 retry contract as the store path:
/// bounded attempts with doubling backoff on transient shapes (connect/timeout/reset via the
/// egress classifier; 5xx/408/429 by status), immediate return on semantic refusals (a 404 here
/// is the authoritative "not a module object" answer the caller folds into its report).
async fn fetch_module_fallback(
    presign: &HttpPresignClient,
    egress: &EgressClient,
    run: &RunId,
    path: &str,
) -> Result<Vec<u8>, String> {
    let mut delay = std::time::Duration::from_secs(1);
    let mut last = String::new();
    for attempt in 1..=FALLBACK_ATTEMPTS {
        let req = PresignRequest::artifact(PresignOp::Get, path);
        let resp = presign
            .presign(run, &req)
            .await
            .map_err(|e| format!("presign GET: {e}"))?;
        let transient = match egress.get(&resp.url, Redirects::None).await {
            Ok(r) if r.status().is_success() => {
                return r
                    .bytes()
                    .await
                    .map(|b| b.to_vec())
                    .map_err(|e| format!("read body: {e}"));
            }
            Ok(r) if status_is_transient(r.status()) => format!("GET returned {}", r.status()),
            Ok(r) => return Err(format!("GET returned {}", r.status())),
            Err(e) => match classify_egress(&e, "module fallback GET") {
                VhcNetError::Transient { detail, .. } => detail,
                other => return Err(other.to_string()),
            },
        };
        last = transient;
        if attempt < FALLBACK_ATTEMPTS {
            eprintln!(
                "  transient on module fallback {path} (attempt {attempt}/{FALLBACK_ATTEMPTS}): \
                 {last}; retrying in {delay:?}"
            );
            tokio::time::sleep(delay).await;
            delay = delay.saturating_mul(2);
        }
    }
    Err(format!("exhausted {FALLBACK_ATTEMPTS} attempts: {last}"))
}

/// The `vhc-archive-pull` entry point.
pub fn run(args: Args) -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context("tokio runtime")?;
    let run = RunId::new(&args.run);

    // -- registry reads: descriptor + envelope + head records ------------------------------------
    let (envelope_bytes, head_records) = rt.block_on(async {
        let mut registry = RegistryClient::new(egress()?, args.base.clone());
        if let Some(bearer) = &args.bearer {
            registry = registry.with_bearer(bearer.clone());
        }
        if let Some((org, actor)) = &args.internal {
            registry = registry.with_internal(org.clone(), actor.clone());
        }
        let descriptor = registry
            .get_run(&args.run)
            .await
            .context("read run descriptor")?
            .with_context(|| format!("run {} is not registered", args.run))?;
        let envelope_bytes = registry
            .fetch_envelope(&run, &descriptor)
            .await
            .context("fetch + verify envelope")?;

        let mut heads = HttpArchiveHeadStore::new(egress()?, args.base.clone(), run.clone());
        if let Some(bearer) = &args.bearer {
            heads = heads.with_bearer(bearer.clone());
        }
        if let Some((org, actor)) = &args.internal {
            heads = heads.with_internal(org.clone(), actor.clone());
        }
        let head_records = heads.fetch_heads().await.context("fetch archive heads")?;
        anyhow::Ok((envelope_bytes, head_records))
    })?;
    anyhow::ensure!(
        !head_records.is_empty(),
        "run {} has no published archive heads",
        args.run
    );

    // -- assemble: the shared verifying core, fetching content through the presigned plane -------
    let presign = |base: &str| -> Result<HttpPresignClient> {
        let mut client = HttpPresignClient::new(egress()?, base.to_string());
        if let Some(bearer) = &args.bearer {
            client = client.with_bearer(bearer.clone());
        }
        if let Some((org, actor)) = &args.internal {
            client = client.with_internal(org.clone(), actor.clone());
        }
        Ok(client)
    };
    let r2 = R2Store::new(presign(&args.base)?, egress()?, run.clone());
    let presign_client = presign(&args.base)?;
    let egress_client = egress()?;
    let mut fetch = |hash: &Hash| -> Result<Vec<u8>, String> {
        // Resume lives in the assembler itself now (`fetch_verified_at`, REL-2a): a prior
        // pull's verified content-addressed file satisfies the fetch before this closure is
        // ever consulted. This closure only moves network bytes.
        rt.block_on(fetch_object(
            &r2,
            &presign_client,
            &egress_client,
            &run,
            hash,
        ))
    };
    let report = assemble_archive(&args.out, &envelope_bytes, head_records, &mut fetch)
        .map_err(|e| anyhow::anyhow!("assemble: {e}"))?;

    println!(
        "assembled run {} into {}",
        report.run_id.to_hex(),
        args.out.display()
    );
    println!("  chains verified    : {}", report.chains_verified);
    println!("  head records       : {}", report.heads_written);
    println!(
        "  sealed segments    : {} fetched, {} reused (resumed)",
        report.segments_written, report.segments_reused
    );
    println!(
        "  payload objects    : {} fetched, {} reused (resumed)",
        report.payloads_written, report.payloads_reused
    );
    if report.module_reused {
        println!("  coordinator.wasm   : reused (resumed)");
    }
    println!("  peer transcripts   : {}", report.peer_transcripts);
    println!(
        "  coordinator lineage: {:?} (chain instances, founding first)",
        report.coordinator_lineage
    );
    println!(
        "verify with: cargo run -p xtask -- vhc-replay --archive {} --run {}",
        args.out.display(),
        report.run_id.to_hex()
    );
    Ok(())
}
