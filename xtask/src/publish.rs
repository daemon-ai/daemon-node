// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `xtask publish-module` / `publish-corpus` — the fleet artifact authoring/upload path (spec §8).
//!
//! Earlier gate stages **pre-staged** the experiment `.wasm` on every fleet box and used a synthetic corpus.
//! This path publishes both to the payload store at **content-addressed** keys, so the fleet fetches
//! them by hash (presigned GET, blake3-verified) — no pre-staging:
//!
//! - `publish-module` uploads a module to `modules/<blake3>.wasm` (a presigned PUT direct to R2 on
//!   the SigV4 plane) and prints its `blake3` + `size` + the `r2://modules/<blake3>.wasm` URL to drop
//!   into the envelope's `[artifacts]` (the worker's `resolve_module` fetches it by hash).
//! - `publish-corpus` uploads the chunk-addressed corpus a `tokenize-corpus` run authored: each
//!   shard to `corpus/<shard_hash>.bin` (the shard's **fold** identity — the chunk-addressed
//!   artifact id, never a plain content hash), the tokenizer artifact to
//!   `corpus/<tokenizer_blake3>.json`, and the canonical-CBOR manifest LAST to
//!   `corpus/<manifest_blake3>.cbor`. Prints the genesis `corpus_manifest` pin + the complete
//!   artifact list a trainer role's grants must carry.
//!
//! xtask is maintainer dev tooling (not the shipped node); its egress rides the SSRF-safe
//! [`daemon_egress::EgressClient`] and the fs reads are covered by main.rs's crate-level
//! `#![allow(clippy::disallowed_methods)]`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_vhc_net::{HttpPresignClient, PresignClient, PresignOp, PresignRequest, RunId};
use daemon_vhc_proto::corpus::CorpusManifest;

/// The presign coordinator target + auth (mirrors the worker's `JoinCredentials` auth choices).
pub struct Target {
    /// The coordinator presign base, e.g. `https://daemon-vhc-dev.me-dc6.workers.dev/api/v1/vhc`.
    pub presign_base: String,
    /// The run id whose prefix the objects live under (`runs/<run>/…`).
    pub run: String,
    /// `vhc:*`-scoped bearer token (gateway path), if any.
    pub bearer: Option<String>,
    /// Internal identity headers (direct-to-`apps/vhc` dev path): `(org_id, actor)`.
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

/// `publish-corpus`: upload the chunk-addressed corpus beside `<dir>/corpus-manifest.cbor` —
/// every shard under its fold identity, the tokenizer artifact, and the manifest LAST (so a
/// partial corpus never has a resolvable manifest). Prints the genesis pin + the artifact list.
pub fn publish_corpus(manifest_path: PathBuf, target: Target) -> Result<()> {
    let dir = manifest_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest = CorpusManifest::from_canonical_bytes(&manifest_bytes)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", manifest_path.display()))?;
    let manifest_hash = daemon_vhc_proto::blake3_hash(&manifest_bytes).to_hex();

    let tokenizer_bytes = std::fs::read(dir.join("tokenizer.json"))
        .context("read tokenizer.json beside the manifest")?;
    let tokenizer_hash = daemon_vhc_proto::blake3_hash(&tokenizer_bytes);
    anyhow::ensure!(
        tokenizer_hash == manifest.tokenizer.hash,
        "tokenizer.json blake3 {} does not match the manifest's tokenizer identity {}",
        tokenizer_hash.to_hex(),
        manifest.tokenizer.hash.to_hex()
    );

    let rt = tokio::runtime::Runtime::new().context("tokio runtime")?;
    rt.block_on(async {
        let presign = target.presign_client()?;
        let egress = EgressClient::new(EgressConfig::default()).context("egress client")?;
        let run = RunId::new(&target.run);

        for (i, shard) in manifest.shards.iter().enumerate() {
            let name = format!("{}.bin", shard.shard_hash.to_hex());
            let bytes = std::fs::read(dir.join(&name))
                .with_context(|| format!("read shard {i} ({name})"))?;
            // Integrity before upload: the file's chunk fold must BE the manifest identity.
            let entry =
                CorpusManifest::author_shard(&bytes, shard.token_count, manifest.chunk_size)
                    .map_err(|e| anyhow::anyhow!("shard {i}: {e}"))?;
            anyhow::ensure!(
                entry.shard_hash == shard.shard_hash,
                "shard {i} bytes do not fold to the manifest identity {}",
                shard.shard_hash.to_hex()
            );
            let path = format!("corpus/{}.bin", shard.shard_hash.to_hex());
            put_artifact(&presign, &egress, &run, &path, &bytes).await?;
            println!("  shard {i} -> r2://{path} ({} bytes)", bytes.len());
        }
        let tokenizer_key = format!("corpus/{}.json", manifest.tokenizer.hash.to_hex());
        put_artifact(&presign, &egress, &run, &tokenizer_key, &tokenizer_bytes).await?;
        println!(
            "  tokenizer -> r2://{tokenizer_key} ({} bytes)",
            tokenizer_bytes.len()
        );
        // Upload the manifest LAST (so a partial corpus never has a resolvable manifest).
        let manifest_key = format!("corpus/{manifest_hash}.cbor");
        put_artifact(&presign, &egress, &run, &manifest_key, &manifest_bytes).await?;
        anyhow::Ok(())
    })?;

    println!("published corpus manifest to r2://corpus/{manifest_hash}.cbor");
    println!("  genesis `corpus_manifest` pin        = {manifest_hash}");
    println!(
        "  tokenizer artifact blake3            = {}",
        manifest.tokenizer.hash.to_hex()
    );
    println!(
        "  total sequences                      = {}",
        manifest.total_sequences()
    );
    println!("  trainer-role artifact grants (manifest + tokenizer + every shard):");
    println!("    {manifest_hash}");
    println!("    {}", manifest.tokenizer.hash.to_hex());
    for shard in &manifest.shards {
        println!("    {}", shard.shard_hash.to_hex());
    }
    Ok(())
}
