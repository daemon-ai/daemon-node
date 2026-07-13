// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `xtask tokenize-corpus` — offline corpus → fixed-width shards + `manifest.json` (spec §8).
//!
//! Mirrors `build-guests`: a maintainer-run dev tool (never the shipped node), so its egress goes
//! through the revision-pinned `hf-hub` client (no raw `reqwest::Client` — the clippy
//! `disallowed_types` egress ban is respected; the fs writes are covered by main.rs's crate-level
//! `#![allow(clippy::disallowed_methods)]`). It pulls a dataset text + a tokenizer (both by pinned
//! revision, or from local paths for hermetic/offline runs), tokenizes with the `tokenizers` crate,
//! and writes fixed-width `u16`/`u32` LE shards + a `manifest.json` carrying the additive tokenizer/
//! dataset provenance (`daemon_swarm_run::data::Manifest`). The shards are the exact format the
//! participant runtime reads (`data.rs`), so a vendored fixture needs no egress in CI (RUN-3).

use std::path::{Path, PathBuf};

use anyhow::Context;
use daemon_swarm_run::data::{Manifest, ShardDesc, TokenWidth};

/// Arguments for `tokenize-corpus` (the frozen CLI seam — see `swarm-ledger-m1.md`).
pub struct Args {
    /// HF **dataset** repo id (e.g. `roneneldan/TinyStories`); ignored when `text` is set.
    pub dataset: Option<String>,
    /// The file within the dataset repo to pull (e.g. `TinyStories-valid.txt`).
    pub dataset_file: Option<String>,
    /// The pinned dataset revision (commit SHA / tag). Recorded in the manifest.
    pub revision: String,
    /// A local corpus text file — bypasses the HF dataset pull entirely (offline / synthetic).
    pub text: Option<PathBuf>,
    /// HF **model** repo id for the tokenizer (e.g. `gpt2`) OR a local `tokenizer.json` path.
    pub tokenizer: String,
    /// The pinned tokenizer revision (defaults to `main` when pulling from HF).
    pub tokenizer_revision: Option<String>,
    /// The output directory for shards + `manifest.json`.
    pub out_dir: PathBuf,
    /// Tokens per shard (rounded down to a whole multiple of `seq_len`).
    pub shard_tokens: u64,
    /// Sequence length (tokens per training sequence).
    pub seq_len: u32,
    /// Token element width: `u16` (vocab ≤ 65 536) or `u32`.
    pub token_width: TokenWidth,
    /// Optional cap on the total tokens emitted (keeps a vendored fixture small).
    pub max_tokens: Option<u64>,
}

/// Run `tokenize-corpus`.
pub fn run(args: Args) -> anyhow::Result<()> {
    anyhow::ensure!(args.seq_len > 0, "--seq-len must be > 0");
    anyhow::ensure!(args.shard_tokens > 0, "--shard-tokens must be > 0");
    let seq_len = u64::from(args.seq_len);

    // 1. Resolve the tokenizer (local file or pinned HF model download).
    let tokenizer_path = resolve_tokenizer(&args.tokenizer, args.tokenizer_revision.as_deref())?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;

    // 2. Resolve the corpus text (local file or pinned HF dataset download).
    let text = resolve_text(&args)?;

    // 3. Tokenize. Encode line-by-line so one gigantic `encode` never has to buffer the whole file.
    let mut ids: Vec<u32> = Vec::new();
    let cap = args.max_tokens.unwrap_or(u64::MAX);
    for line in text.split_inclusive('\n') {
        if ids.len() as u64 >= cap {
            break;
        }
        let enc = tokenizer
            .encode(line, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        ids.extend_from_slice(enc.get_ids());
    }
    if ids.len() as u64 > cap {
        ids.truncate(cap as usize);
    }

    // 4. Align down to whole sequences, then to whole shards.
    let per_shard = (args.shard_tokens / seq_len) * seq_len;
    anyhow::ensure!(per_shard > 0, "--shard-tokens must be >= --seq-len");
    let usable = (ids.len() as u64 / seq_len) * seq_len;
    anyhow::ensure!(
        usable >= seq_len,
        "corpus produced only {} tokens (< seq_len {})",
        ids.len(),
        args.seq_len
    );
    ids.truncate(usable as usize);

    // 5. Validate the token range fits the declared width.
    let max_id = ids.iter().copied().max().unwrap_or(0);
    if matches!(args.token_width, TokenWidth::U16) {
        anyhow::ensure!(
            max_id < 65_536,
            "token id {max_id} exceeds u16; use --token-width u32"
        );
    }

    // 6. Write shards + manifest.
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("create {}", args.out_dir.display()))?;
    let mut shards: Vec<ShardDesc> = Vec::new();
    let mut cursor = 0u64;
    let mut shard_idx = 0u32;
    while cursor < usable {
        let end = (cursor + per_shard).min(usable);
        let chunk = &ids[cursor as usize..end as usize];
        let bytes = encode_shard(chunk, args.token_width);
        let name = format!("shard-{shard_idx:04}.bin");
        std::fs::write(args.out_dir.join(&name), &bytes)
            .with_context(|| format!("write {name}"))?;
        shards.push(ShardDesc {
            name,
            bytes: bytes.len() as u64,
            tokens: chunk.len() as u64,
            blake3: blake3::hash(&bytes).to_hex().to_string(),
        });
        cursor = end;
        shard_idx += 1;
    }

    let manifest = Manifest {
        token_width: args.token_width,
        seq_len: args.seq_len,
        shards,
        tokenizer: Some(args.tokenizer.clone()),
        tokenizer_revision: args.tokenizer_revision.clone(),
        dataset: args.dataset.clone(),
        dataset_revision: args.dataset.as_ref().map(|_| args.revision.clone()),
    };
    manifest
        .validate()
        .context("generated manifest failed validation")?;
    let json = manifest.to_json().context("serialize manifest.json")?;
    std::fs::write(args.out_dir.join("manifest.json"), json).context("write manifest.json")?;

    println!(
        "wrote {} shards ({} tokens, seq_len {}) + manifest.json to {}",
        manifest.shards.len(),
        manifest.total_tokens(),
        args.seq_len,
        args.out_dir.display()
    );
    Ok(())
}

/// Encode a token slice to fixed-width little-endian bytes.
fn encode_shard(ids: &[u32], width: TokenWidth) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * width.bytes() as usize);
    match width {
        TokenWidth::U16 => {
            for &id in ids {
                out.extend_from_slice(&(id as u16).to_le_bytes());
            }
        }
        TokenWidth::U32 => {
            for &id in ids {
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
    }
    out
}

/// Resolve the tokenizer to a local `tokenizer.json` path: a local file as-is, else a pinned HF
/// model download.
fn resolve_tokenizer(spec: &str, revision: Option<&str>) -> anyhow::Result<PathBuf> {
    let local = Path::new(spec);
    if local.is_file() {
        return Ok(local.to_path_buf());
    }
    let rev = revision.unwrap_or("main").to_string();
    hf_get(
        spec.to_string(),
        hf_hub::RepoType::Model,
        rev,
        "tokenizer.json",
    )
    .with_context(|| format!("download tokenizer.json from HF model {spec}"))
}

/// Resolve the corpus text: `--text` local file, else a pinned HF dataset file download.
fn resolve_text(args: &Args) -> anyhow::Result<String> {
    if let Some(path) = &args.text {
        return std::fs::read_to_string(path)
            .with_context(|| format!("read corpus text {}", path.display()));
    }
    let dataset = args
        .dataset
        .as_deref()
        .context("provide --text <file> or --dataset <hf-id> --dataset-file <name>")?;
    let file = args
        .dataset_file
        .as_deref()
        .context("--dataset requires --dataset-file <name>")?;
    let path = hf_get(
        dataset.to_string(),
        hf_hub::RepoType::Dataset,
        args.revision.clone(),
        file,
    )
    .with_context(|| {
        format!(
            "download {file} from HF dataset {dataset}@{}",
            args.revision
        )
    })?;
    std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))
}

/// Download one file from a pinned HF repo via `hf-hub` (its cache), returning the local path. Uses
/// a short-lived tokio runtime so the sync xtask can drive the async hub client.
fn hf_get(
    repo: String,
    kind: hf_hub::RepoType,
    revision: String,
    file: &str,
) -> anyhow::Result<PathBuf> {
    let rt = tokio::runtime::Runtime::new().context("tokio runtime")?;
    rt.block_on(async move {
        let api = hf_hub::api::tokio::Api::new().context("hf-hub api")?;
        let repo = api.repo(hf_hub::Repo::with_revision(repo, kind, revision));
        repo.get(file)
            .await
            .map_err(|e| anyhow::anyhow!("hf-hub get {file}: {e}"))
    })
}
