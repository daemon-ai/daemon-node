// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `xtask publish-module` / `publish-corpus` — the P3 lane-S authoring/upload path (spec §8).
//!
//! The P2 gate **pre-staged** the experiment `.wasm` on every fleet box and used a synthetic corpus.
//! Lane S publishes both to the payload store at **content-addressed** keys, so the fleet fetches
//! them by hash (presigned GET, blake3-verified) — no pre-staging:
//!
//! - `publish-module` uploads a module to `modules/<blake3>.wasm` (a presigned PUT direct to R2 on
//!   the SigV4 plane) and prints its `blake3` + `size` + the `r2://modules/<blake3>.wasm` URL to drop
//!   into the envelope's `[artifacts]` (the worker's `resolve_module` fetches it by hash).
//! - `publish-corpus` uploads each `tokenize-corpus` shard to `corpus/<shard_blake3>.bin` and the
//!   `manifest.json` to `corpus/<manifest_blake3>.json`, then prints the `manifest_blake3` + total
//!   sequences for the `CorpusRef` (`EngineParams.corpus`) the run author declares.
//!
//! xtask is maintainer dev tooling (not the shipped node); its egress rides the SSRF-safe
//! [`daemon_egress::EgressClient`] and the fs reads are covered by main.rs's crate-level
//! `#![allow(clippy::disallowed_methods)]`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_swarm_net::{HttpPresignClient, PresignClient, PresignOp, PresignRequest, RunId};
use daemon_swarm_run::data::Manifest;

/// The presign coordinator target + auth (mirrors the worker's `JoinCredentials` auth choices).
pub struct Target {
    /// The coordinator presign base, e.g. `https://daemon-swarm-dev.me-dc6.workers.dev/api/v1/swarm`.
    pub presign_base: String,
    /// The run id whose prefix the objects live under (`runs/<run>/…`).
    pub run: String,
    /// `swarm:*`-scoped bearer token (gateway path), if any.
    pub bearer: Option<String>,
    /// Internal identity headers (direct-to-`apps/swarm` dev path): `(org_id, actor)`.
    pub internal: Option<(String, String)>,
}

impl Target {
    fn presign_client(&self) -> Result<HttpPresignClient> {
        let egress = EgressClient::new(EgressConfig::default()).context("presign egress client")?;
        let mut client = HttpPresignClient::new(egress, self.presign_base.clone());
        if let Some(bearer) = &self.bearer {
            client = client.with_bearer(bearer.clone());
        }
        if let Some((org, actor)) = &self.internal {
            client = client.with_internal(org.clone(), actor.clone());
        }
        Ok(client)
    }
}

/// Upload `bytes` to the run-relative artifact `path` via a presigned PUT (idempotent — a
/// content-addressed key is safe to re-PUT). Returns nothing; errors carry the object status.
async fn put_artifact(
    presign: &HttpPresignClient,
    egress: &EgressClient,
    run: &RunId,
    path: &str,
    bytes: &[u8],
) -> Result<()> {
    let req = PresignRequest::artifact(PresignOp::Put, path);
    let resp = presign
        .presign(run, &req)
        .await
        .with_context(|| format!("presign PUT {path}"))?;
    let egress_resp = if resp.headers.is_empty() {
        egress
            .put(&resp.url, bytes.to_vec(), Redirects::None)
            .await
            .with_context(|| format!("PUT {path}"))?
    } else {
        let mut ereq = EgressRequest::put(&resp.url, bytes.to_vec());
        for (name, value) in &resp.headers {
            ereq = ereq.header(name, value);
        }
        egress
            .execute(ereq, Redirects::None)
            .await
            .with_context(|| format!("PUT {path}"))?
    };
    let status = egress_resp.status();
    anyhow::ensure!(status.is_success(), "PUT {path} returned {status}");
    Ok(())
}

/// `publish-module`: upload `module` to `modules/<blake3>.wasm` and print its hash + size + URL.
pub fn publish_module(module: PathBuf, target: Target) -> Result<()> {
    let bytes =
        std::fs::read(&module).with_context(|| format!("read module {}", module.display()))?;
    let hash = blake3::hash(&bytes).to_hex().to_string();
    let path = format!("modules/{hash}.wasm");
    let rt = tokio::runtime::Runtime::new().context("tokio runtime")?;
    rt.block_on(async {
        let presign = target.presign_client()?;
        let egress = EgressClient::new(EgressConfig::default()).context("egress client")?;
        let run = RunId::new(&target.run);
        put_artifact(&presign, &egress, &run, &path, &bytes).await
    })?;
    println!("published module ({} bytes) to r2://{path}", bytes.len());
    println!("  blake3 = {hash}");
    println!("  size   = {}", bytes.len());
    println!("  envelope [artifacts] url: r2://{path}");
    Ok(())
}

/// `publish-corpus`: upload every shard listed in `<dir>/manifest.json` to `corpus/<shard_blake3>.bin`
/// and the manifest itself to `corpus/<manifest_blake3>.json`. Prints the `CorpusRef` fields.
pub fn publish_corpus(manifest_path: PathBuf, target: Target) -> Result<()> {
    let dir = manifest_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest = Manifest::from_json(std::str::from_utf8(&manifest_bytes)?)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    let manifest_hash = blake3::hash(&manifest_bytes).to_hex().to_string();

    let rt = tokio::runtime::Runtime::new().context("tokio runtime")?;
    rt.block_on(async {
        let presign = target.presign_client()?;
        let egress = EgressClient::new(EgressConfig::default()).context("egress client")?;
        let run = RunId::new(&target.run);

        for shard in &manifest.shards {
            let bytes = std::fs::read(dir.join(&shard.name))
                .with_context(|| format!("read shard {}", shard.name))?;
            let actual = blake3::hash(&bytes).to_hex().to_string();
            anyhow::ensure!(
                actual == shard.blake3,
                "shard {} content blake3 {actual} does not match manifest {}",
                shard.name,
                shard.blake3
            );
            let path = format!("corpus/{}.bin", shard.blake3);
            put_artifact(&presign, &egress, &run, &path, &bytes).await?;
            println!(
                "  shard {} -> r2://{path} ({} bytes)",
                shard.name,
                bytes.len()
            );
        }
        // Upload the manifest LAST (so a partial corpus never has a resolvable manifest).
        let manifest_key = format!("corpus/{manifest_hash}.json");
        put_artifact(&presign, &egress, &run, &manifest_key, &manifest_bytes).await?;
        anyhow::Ok(())
    })?;

    println!("published corpus manifest to r2://corpus/{manifest_hash}.json");
    println!("  CorpusRef.manifest_blake3 (hex) = {manifest_hash}");
    println!(
        "  CorpusRef.manifest_size         = {}",
        manifest_bytes.len()
    );
    println!(
        "  total sequences = {} (set window_sequences = min(rounds*global_batch, total), or 0 = whole corpus)",
        manifest.total_sequences()
    );
    Ok(())
}
