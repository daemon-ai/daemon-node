// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-workspace-index` — an embedding-based index over a workspace directory tree, backing the
//! `semantic_search` chat tool (Cursor `SemanticSearch` parity).
//!
//! It walks the workspace (respecting `.gitignore`), chunks each text file heuristically, embeds the
//! chunks through the shared [`EmbeddingProvider`] port, and persists chunk text + vectors in its own
//! SQLite database. A background task keeps the index fresh via a periodic reconcile sweep. Queries
//! embed the query string and brute-force cosine-rank every chunk (the design's explicit tradeoff —
//! no ANN index), returning the top hits with their file path + line span + text.
//!
//! The index is deliberately decoupled from the node: it is injected into the daemon as an
//! `extra_tools` handle (like `session_search`), never touching `NodeAssembly`/`daemon-core`.

#![forbid(unsafe_code)]

mod chunk;
mod indexer;
mod store;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use daemon_core::EmbeddingProvider;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::store::{Store, SCHEMA_VERSION};

/// Errors opening or migrating the index store. Query/index *content* failures never surface here —
/// [`WorkspaceIndex::query`] returns an empty result and the background sweep logs + skips.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// A SQLite error from the index store.
    #[error("workspace-index sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A migration-ladder error applying the index schema.
    #[error("workspace-index migration error: {0}")]
    Migrate(#[from] rusqlite_migration::Error),
}

/// Tuning for the workspace index.
#[derive(Clone, Debug)]
pub struct WorkspaceIndexConfig {
    /// Whether the index is active (the node skips wiring it when `false`).
    pub enable: bool,
    /// Files larger than this (bytes) are skipped (default 1 MiB).
    pub max_file_bytes: u64,
    /// The fixed-window height (lines) for regions without a semantic anchor (default 60).
    pub chunk_lines: usize,
    /// How many lines consecutive windows share (default 10).
    pub chunk_overlap: usize,
    /// The provider embed-call batch size (default 64).
    pub batch: usize,
    /// The reconcile-sweep cadence (default 30s).
    pub reconcile_interval: Duration,
}

impl Default for WorkspaceIndexConfig {
    fn default() -> Self {
        Self {
            enable: true,
            max_file_bytes: 1024 * 1024,
            chunk_lines: 60,
            chunk_overlap: 10,
            batch: 64,
            reconcile_interval: Duration::from_secs(30),
        }
    }
}

/// One similarity hit: a file path (root-relative), an inclusive 1-based line span, the cosine
/// score, and the chunk's text.
#[derive(Clone, Debug)]
pub struct IndexHit {
    /// The matching file's workspace-root-relative path (forward slashes).
    pub path: String,
    /// The 1-based first line of the chunk (inclusive).
    pub start_line: usize,
    /// The 1-based last line of the chunk (inclusive).
    pub end_line: usize,
    /// The cosine similarity of the chunk to the query, in `[-1, 1]`.
    pub score: f32,
    /// The chunk's verbatim text.
    pub snippet: String,
}

/// The shared state behind [`WorkspaceIndex`] (held in an `Arc` so a query can offload the cosine
/// scan onto `spawn_blocking` and the background task can own a clone).
struct Inner {
    store: Store,
    workspace_root: PathBuf,
    embedder: Arc<dyn EmbeddingProvider>,
    cfg: WorkspaceIndexConfig,
    ready: AtomicBool,
}

/// A handle to the workspace embedding index (cheaply cloneable via its inner `Arc`).
pub struct WorkspaceIndex {
    inner: Arc<Inner>,
}

impl WorkspaceIndex {
    /// Open (creating if absent) the index at `db_path` over `workspace_root`, embedding through
    /// `embedder`. The `meta` table records the root + model + dims + schema; any mismatch triggers a
    /// full rebuild (the stored vectors are only comparable within one model/dimensionality).
    pub fn open(
        db_path: &Path,
        workspace_root: PathBuf,
        embedder: Arc<dyn EmbeddingProvider>,
        cfg: WorkspaceIndexConfig,
    ) -> Result<Arc<Self>, IndexError> {
        let store = Store::open(db_path)?;

        let root_str = workspace_root.to_string_lossy().into_owned();
        let model = embedder.model().to_string();
        let dims = embedder.dimensions().to_string();
        let mismatch = store.meta_get("workspace_root")?.as_deref() != Some(root_str.as_str())
            || store.meta_get("embed_model")?.as_deref() != Some(model.as_str())
            || store.meta_get("embed_dims")?.as_deref() != Some(dims.as_str())
            || store.meta_get("schema")?.as_deref() != Some(SCHEMA_VERSION);
        if mismatch {
            store.clear()?;
            store.meta_set("workspace_root", &root_str)?;
            store.meta_set("embed_model", &model)?;
            store.meta_set("embed_dims", &dims)?;
            store.meta_set("schema", SCHEMA_VERSION)?;
        }

        Ok(Arc::new(Self {
            inner: Arc::new(Inner {
                store,
                workspace_root,
                embedder,
                cfg,
                ready: AtomicBool::new(false),
            }),
        }))
    }

    /// The workspace root the index is rooted at (its containment reference).
    pub fn workspace_root(&self) -> &Path {
        &self.inner.workspace_root
    }

    /// Whether the initial full pass has completed (queries return `[]` until it flips).
    pub fn ready(&self) -> bool {
        self.inner.ready.load(Ordering::Acquire)
    }

    /// Embed `query` and return the top-`k` chunks by cosine similarity, restricted to `dir_filters`
    /// (root-relative directory prefixes; empty = the whole index). Returns `[]` when the index is
    /// not ready, the query is empty, or embedding fails — never an error.
    pub async fn query(&self, query: &str, k: usize, dir_filters: &[String]) -> Vec<IndexHit> {
        if !self.ready() || query.trim().is_empty() || k == 0 {
            return Vec::new();
        }
        let qvec = match self.inner.embedder.embed(&[query.to_string()]).await {
            Ok(vecs) if !vecs.is_empty() => vecs.into_iter().next().unwrap(),
            _ => return Vec::new(),
        };
        let inner = self.inner.clone();
        let filters = dir_filters.to_vec();
        match tokio::task::spawn_blocking(move || inner.store.topk(&qvec, k, &filters)).await {
            Ok(Ok(hits)) => hits,
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "workspace-index query failed");
                Vec::new()
            }
            Err(e) => {
                tracing::warn!(error = %e, "workspace-index query task panicked");
                Vec::new()
            }
        }
    }

    /// Spawn the background indexer: an initial full pass (which flips [`ready`](Self::ready)), then a
    /// reconcile sweep every `reconcile_interval` until `cancel` fires. The returned handle lets the
    /// caller await/abort the task on shutdown.
    pub fn spawn(self: &Arc<Self>, cancel: CancellationToken) -> JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            if let Err(e) = this.run_sweep().await {
                tracing::warn!(error = %e, "workspace-index initial pass failed");
            }
            this.inner.ready.store(true, Ordering::Release);
            tracing::info!(root = %this.inner.workspace_root.display(), "workspace index ready");
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = tokio::time::sleep(this.inner.cfg.reconcile_interval) => {
                        // DEFERRED FOLLOW-UP: there is no fs-write push-hook here to invalidate a
                        // single edited file incrementally — the node's `WorkspaceFs` is a pull
                        // cursor with no clean event seam, so freshness rides this periodic
                        // mtime+size reconcile instead. A future FsTool push-hook would let an edit
                        // re-embed just the touched file immediately.
                        if let Err(e) = this.run_sweep().await {
                            tracing::warn!(error = %e, "workspace-index reconcile failed");
                        }
                    }
                }
            }
        })
    }

    /// Run one reconcile sweep against the store.
    async fn run_sweep(&self) -> Result<(), IndexError> {
        indexer::sweep(
            &self.inner.store,
            &self.inner.workspace_root,
            self.inner.embedder.as_ref(),
            &self.inner.cfg,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::MockEmbedder;
    use std::io::Write;

    /// Materialize a workspace fixture with the given `(relpath, contents)` files (dirs created).
    fn fixture(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (rel, contents) in files {
            let path = dir.path().join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(contents.as_bytes()).unwrap();
        }
        dir
    }

    fn open_index(root: &Path, embedder: Arc<dyn EmbeddingProvider>) -> Arc<WorkspaceIndex> {
        WorkspaceIndex::open(
            &root.join("index.sqlite"),
            root.to_path_buf(),
            embedder,
            WorkspaceIndexConfig {
                reconcile_interval: Duration::from_millis(50),
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn initial_pass_indexes_text_excludes_gitignored_and_binary() {
        let dir = fixture(&[
            ("src/auth.rs", "fn authenticate_user_with_jwt() {}\n"),
            ("src/ignoreme.rs", "fn secret() {}\n"),
            (".gitignore", "ignoreme.rs\n"),
            ("data.bin", "\0\0binary\0content"),
            ("notes.md", "# Notes\nsome prose about tokens\n"),
        ]);
        let idx = open_index(dir.path(), Arc::new(MockEmbedder::new(64)));
        let cancel = CancellationToken::new();
        let handle = idx.spawn(cancel.clone());

        // Wait for the initial pass to flip ready.
        wait_ready(&idx).await;

        // A query embeds and returns hits from the indexed text files.
        let hits = idx.query("authenticate jwt", 10, &[]).await;
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(paths.contains(&"src/auth.rs"), "auth.rs indexed: {paths:?}");
        assert!(paths.contains(&"notes.md"), "markdown indexed: {paths:?}");
        // The gitignored file and the binary file are excluded.
        assert!(
            !paths.contains(&"src/ignoreme.rs"),
            "gitignored excluded: {paths:?}"
        );
        assert!(!paths.contains(&"data.bin"), "binary excluded: {paths:?}");

        cancel.cancel();
        handle.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_reembeds_changed_deletes_removed() {
        let dir = fixture(&[("a.rs", "fn alpha() {}\n"), ("b.rs", "fn beta() {}\n")]);
        let idx = open_index(dir.path(), Arc::new(MockEmbedder::new(64)));
        let cancel = CancellationToken::new();
        let handle = idx.spawn(cancel.clone());
        wait_ready(&idx).await;

        assert!(!idx.query("alpha", 10, &[]).await.is_empty());
        assert!(!idx.query("beta", 10, &[]).await.is_empty());

        // Remove b.rs, change a.rs (new content the pre-filter must catch via mtime/size change).
        std::fs::remove_file(dir.path().join("b.rs")).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        std::fs::write(
            dir.path().join("a.rs"),
            "fn alpha_renamed_much_longer_body() {}\n",
        )
        .unwrap();

        // Wait for a reconcile to pick up the change: b.rs gone, a.rs still present.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let b_gone = idx
                .query("beta", 10, &[])
                .await
                .iter()
                .all(|h| h.path != "b.rs");
            if b_gone {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "reconcile never removed b.rs"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        // a.rs is still queryable after re-embed.
        assert!(idx
            .query("alpha", 10, &[])
            .await
            .iter()
            .any(|h| h.path == "a.rs"));

        cancel.cancel();
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn query_returns_empty_before_ready() {
        let dir = fixture(&[("a.rs", "fn alpha() {}\n")]);
        let idx = open_index(dir.path(), Arc::new(MockEmbedder::new(64)));
        // No spawn ⇒ never ready ⇒ empty.
        assert!(!idx.ready());
        assert!(idx.query("alpha", 10, &[]).await.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn model_change_triggers_rebuild() {
        let dir = fixture(&[("a.rs", "fn alpha() {}\n")]);
        let db = dir.path().join("index.sqlite");
        // First open with a 32-dim mock, index, close.
        {
            let idx = WorkspaceIndex::open(
                &db,
                dir.path().to_path_buf(),
                Arc::new(MockEmbedder::new(32)),
                WorkspaceIndexConfig::default(),
            )
            .unwrap();
            let cancel = CancellationToken::new();
            let handle = idx.spawn(cancel.clone());
            wait_ready(&idx).await;
            assert!(!idx.query("alpha", 10, &[]).await.is_empty());
            cancel.cancel();
            handle.await.unwrap();
        }
        // Reopen with a DIFFERENT dimensionality: the meta mismatch rebuilds, so the stale 32-dim
        // vectors are cleared. A query works again once the new pass runs (proving no dim panic).
        let idx = WorkspaceIndex::open(
            &db,
            dir.path().to_path_buf(),
            Arc::new(MockEmbedder::new(128)),
            WorkspaceIndexConfig::default(),
        )
        .unwrap();
        let cancel = CancellationToken::new();
        let handle = idx.spawn(cancel.clone());
        wait_ready(&idx).await;
        assert!(!idx.query("alpha", 10, &[]).await.is_empty());
        cancel.cancel();
        handle.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_stops_the_reconcile_loop() {
        let dir = fixture(&[("a.rs", "fn alpha() {}\n")]);
        let idx = open_index(dir.path(), Arc::new(MockEmbedder::new(64)));
        let cancel = CancellationToken::new();
        let handle = idx.spawn(cancel.clone());
        wait_ready(&idx).await;
        cancel.cancel();
        // The task exits promptly (well within the test deadline).
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("cancel should stop the loop")
            .unwrap();
    }

    async fn wait_ready(idx: &Arc<WorkspaceIndex>) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !idx.ready() {
            assert!(
                std::time::Instant::now() < deadline,
                "index never became ready"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
