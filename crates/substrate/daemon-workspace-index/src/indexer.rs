// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The walk + reconcile sweep: enumerate the workspace, (re)chunk + (re)embed what changed, and
//! prune rows for files that disappeared. The initial pass and every periodic reconcile share this
//! one idempotent routine — a fresh index is just a reconcile against an empty store.

use std::collections::HashSet;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_core::EmbeddingProvider;
use daemon_tool_fs::read::{has_binary_extension, looks_binary};
use sha2::{Digest, Sha256};

use crate::chunk::chunk_text;
use crate::store::{ChunkRow, Store};
use crate::{IndexError, WorkspaceIndexConfig};

/// One full reconcile pass over `root`. Errors from a single file (embed failure, unreadable bytes)
/// are logged and skipped so one bad file never fails the whole sweep; only store-level errors
/// propagate.
pub(crate) async fn sweep(
    store: &Store,
    root: &Path,
    embedder: &dyn EmbeddingProvider,
    cfg: &WorkspaceIndexConfig,
) -> Result<(), IndexError> {
    let mut present: HashSet<String> = HashSet::new();

    for path in walk_files(root) {
        let Some(rel) = rel_path(root, &path) else {
            continue;
        };
        // Binary-by-extension: never index, and drop any stale row.
        if has_binary_extension(&rel) {
            store.delete_file(&rel)?;
            continue;
        }
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        let size = meta.len();
        // Oversized: skip and drop any stale row.
        if size > cfg.max_file_bytes {
            store.delete_file(&rel)?;
            continue;
        }
        let mtime_ms = mtime_ms(&meta);
        present.insert(rel.clone());

        // Cheap pre-filter: unchanged mtime AND size ⇒ assume unchanged, skip the read+hash.
        let prior = store.file_row(&rel)?;
        if let Some(prior) = &prior {
            if prior.mtime_ms == mtime_ms && prior.size == size as i64 {
                continue;
            }
        }

        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        // Binary-by-content (extension-less binaries): skip and drop any stale row.
        if looks_binary(&bytes) {
            store.delete_file(&rel)?;
            continue;
        }

        let hash = Sha256::digest(&bytes);
        // Content-hash guard: a touch that did not change bytes only refreshes the stamp.
        if let Some(prior) = &prior {
            if prior.content_hash == hash.as_slice() {
                store.touch_file(&rel, mtime_ms, size as i64, now_ms())?;
                continue;
            }
        }

        let text = String::from_utf8_lossy(&bytes);
        let chunks = chunk_text(&text, cfg.chunk_lines, cfg.chunk_overlap);
        if chunks.is_empty() {
            // An empty/whitespace-only file: record its identity so the pre-filter skips it next
            // time, but with no chunks.
            store.upsert_file(&rel, hash.as_slice(), mtime_ms, size as i64, now_ms(), &[])?;
            continue;
        }

        // Embed in `cfg.batch`-sized provider calls, yielding between batches so the shared local
        // embedding worker (a single serialized Mutex) is not starved from Mnemosyne recall.
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut embed_failed = false;
        for batch in texts.chunks(cfg.batch.max(1)) {
            match embedder.embed(batch).await {
                Ok(vecs) => embeddings.extend(vecs),
                Err(e) => {
                    tracing::warn!(path = %rel, error = %e, "embedding a workspace file failed; skipping");
                    embed_failed = true;
                    break;
                }
            }
            tokio::task::yield_now().await;
        }
        if embed_failed || embeddings.len() != chunks.len() {
            continue;
        }

        let rows: Vec<ChunkRow<'_>> = chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(c, emb)| ChunkRow {
                start_line: c.start_line,
                end_line: c.end_line,
                text: &c.text,
                embedding: emb,
            })
            .collect();
        store.upsert_file(
            &rel,
            hash.as_slice(),
            mtime_ms,
            size as i64,
            now_ms(),
            &rows,
        )?;
    }

    // Remove rows for files that no longer exist (or became binary/oversized and were dropped above,
    // which already left them out of `present`).
    store.delete_absent(&present)?;
    Ok(())
}

/// Build the gitignore-respecting, deterministic file list. Derived inline (the fs tool's walker is
/// private): respect `.gitignore`/`.ignore` even without a `.git` dir, and sort for determinism.
fn walk_files(root: &Path) -> Vec<std::path::PathBuf> {
    ignore::WalkBuilder::new(root)
        .require_git(false)
        .sort_by_file_path(std::cmp::Ord::cmp)
        .build()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
        .map(|e| e.into_path())
        .collect()
}

/// The workspace-relative, forward-slash path for `path` under `root` (`None` if not under root).
fn rel_path(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    Some(
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/"),
    )
}

/// A file's mtime in unix-millis (`0` if unavailable).
fn mtime_ms(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// The current wall-clock time in unix-millis.
fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
