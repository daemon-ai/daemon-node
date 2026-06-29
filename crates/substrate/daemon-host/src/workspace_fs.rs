// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `workspace_fs` — the node's filesystem / workspace surface (daemon-fs-surface-spec.md).
//!
//! Two pieces:
//! - [`WorkspaceRoots`]: resolves an [`FsRootId`](daemon_api::FsRootId) (or a bare session id) to a
//!   directory on the node. It is shared by the engine's exec-env builder (so agents root here) and
//!   by the [`NodeApi`](daemon_api::NodeApi) filesystem ops (so operator and agent see one
//!   filesystem). A session's root is either an isolated per-session sandbox under the node
//!   `workspace_root`, or a directory the operator bound in place (recorded when the engine is
//!   built). It also owns the **host browse** policy (home + operator allowlist) used for discovery
//!   before binding.
//! - [`WorkspaceFs`]: the actual list/stat/read/write/search operations, served via
//!   [`daemon_core::exec::contain`] + `tokio::fs` (the engine is never involved), reusing the
//!   agents' containment so a path can never escape its root.

use daemon_common::cursored::CursoredRing;
use dashmap::DashMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use daemon_api::{
    ApiError, FsChange, FsChangeKind, FsContent, FsEntry, FsEntryKind, FsRevision, FsRootId,
    FsSearchHit, FsSearchPage, FsSearchQuery, FsWatchPageView,
};
use daemon_core::exec::contain;

/// Default `fs_read` cap when the caller passes `max_bytes == 0`.
const DEFAULT_MAX_READ: u64 = 1024 * 1024;
/// Per-file cap when scanning for `fs_search` (skip larger / binary-ish files).
const SEARCH_FILE_CAP: u64 = 1024 * 1024;
/// Hard cap on files visited by one `fs_search` (latency guard).
const SEARCH_FILE_BUDGET: usize = 20_000;
/// Bounded change-event ring kept per watched directory.
const WATCH_RING_CAP: usize = 4096;

/// Directory / file names treated as ignored (build artifacts + VCS). Marked on listings rather
/// than hidden; the client chooses whether to show them. (A full `.gitignore` evaluation is a
/// future refinement; this matches the artifact set agents/tools already skip.)
const IGNORED_NAMES: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    ".cache",
    "__pycache__",
    ".venv",
    "dist",
    "build",
    ".next",
    ".turbo",
];

fn is_ignored_name(name: &str) -> bool {
    IGNORED_NAMES.contains(&name)
}

fn mtime_ms(meta: &std::fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn entry_kind(meta: &std::fs::Metadata) -> FsEntryKind {
    if meta.file_type().is_symlink() {
        FsEntryKind::Symlink
    } else if meta.is_dir() {
        FsEntryKind::Dir
    } else {
        FsEntryKind::File
    }
}

/// Join a root-relative dir + a child name into a POSIX root-relative path.
fn join_rel(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        name.to_string()
    } else {
        format!("{}/{}", dir.trim_end_matches('/'), name)
    }
}

/// Sanitize a session id into a single safe path segment (mirrors
/// [`daemon_core::exec::LocalEnvironment::sandbox`]'s rule), so a path-namespaced child id like
/// `parent/c1` becomes one flat directory and can never introduce traversal.
fn sanitize_segment(session: &str) -> String {
    session
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Resolves filesystem roots for the node. Shared (as an `Arc`) between the engine exec-env builder
/// and the NodeApi filesystem surface so both resolve a session to the *same* directory.
#[derive(Debug)]
pub struct WorkspaceRoots {
    /// Parent directory of isolated per-session sandboxes; also the `FsRootId::Workspace` root.
    base: PathBuf,
    /// Session id -> the root an engine actually rooted at, recorded at engine construction. A
    /// recorded entry overrides the isolated default (this is how a `Bound` session is reflected to
    /// the FS surface).
    bindings: DashMap<String, PathBuf>,
    /// The host browse allowlist: advertised root id -> absolute directory (home + operator
    /// allowlist). Read-only discovery roots; empty disables host browsing.
    browse: Vec<(String, PathBuf)>,
}

impl WorkspaceRoots {
    /// A resolver rooted at `base` (the node `workspace_root`) with no host browse roots.
    pub fn new(base: PathBuf) -> Self {
        Self {
            base,
            bindings: DashMap::new(),
            browse: Vec::new(),
        }
    }

    /// Set the host browse allowlist (advertised id -> directory). Order is preserved; the first
    /// entry is treated as the default browse root by clients.
    pub fn with_browse_roots(mut self, roots: Vec<(String, PathBuf)>) -> Self {
        self.browse = roots;
        self
    }

    /// The default isolated sandbox for a session: `<base>/<sanitized session id>`.
    pub fn isolated_root(&self, session_id: &str) -> PathBuf {
        self.base.join(sanitize_segment(session_id))
    }

    /// Record the root an engine actually rooted at, so the FS surface resolves the same directory
    /// (called from the exec-env builder when the engine is constructed).
    pub fn record(&self, session_id: &str, root: PathBuf) {
        self.bindings.insert(session_id.to_string(), root);
    }

    /// The resolved root for a session: a recorded binding if present, else the isolated sandbox.
    pub fn session_root(&self, session_id: &str) -> PathBuf {
        self.bindings
            .get(session_id)
            .map(|p| p.clone())
            .unwrap_or_else(|| self.isolated_root(session_id))
    }

    /// The node workspace root (`FsRootId::Workspace`).
    pub fn workspace_root(&self) -> PathBuf {
        self.base.clone()
    }

    /// Resolve a host browse root id to its allowed directory (`None` if not advertised).
    pub fn host_root(&self, id: &str) -> Option<PathBuf> {
        self.browse
            .iter()
            .find(|(k, _)| k == id)
            .map(|(_, p)| p.clone())
    }

    /// The advertised host browse roots (id, dir).
    pub fn browse_roots(&self) -> &[(String, PathBuf)] {
        &self.browse
    }
}

/// Per-watched-directory change state: the last directory snapshot (name -> (mtime_ms, size)) and a
/// shared [`CursoredRing`] of `(seq, change)` for cursor reads. The ring's floor flags a reader that
/// aged out (the fs analogue of the merged log's lossy-lag `Reset`); re-list to reconcile.
struct WatchState {
    snapshot: HashMap<String, (u64, u64)>,
    ring: CursoredRing<FsChange>,
    primed: bool,
}

impl WatchState {
    fn new(cap: usize) -> Self {
        Self {
            snapshot: HashMap::new(),
            ring: CursoredRing::new(cap),
            primed: false,
        }
    }
}

/// The filesystem operations behind the node's `fs_*` surface. Resolves an [`FsRootId`] to a
/// directory via [`WorkspaceRoots`], then serves list/stat/read/write/search through
/// [`daemon_core::exec::contain`] + `std`/`tokio` fs, so a path can never escape its root — the
/// same containment the agent `fs` tool is bound by. The engine is never involved.
pub struct WorkspaceFs {
    roots: std::sync::Arc<WorkspaceRoots>,
    /// (root-key, dir) -> change state, for the `fs_watch_after` cursor (on-demand diff).
    watches: DashMap<(String, String), WatchState>,
    /// The per-dir watch-ring capacity (default [`WATCH_RING_CAP`]); overridable in tests to exercise
    /// the overflow -> `reset` path without synthesizing thousands of changes.
    watch_ring_cap: usize,
}

impl WorkspaceFs {
    /// Build the surface over a shared [`WorkspaceRoots`].
    pub fn new(roots: std::sync::Arc<WorkspaceRoots>) -> Self {
        Self {
            roots,
            watches: DashMap::new(),
            watch_ring_cap: WATCH_RING_CAP,
        }
    }

    /// Test-only: shrink the watch ring so an overflow -> `reset` is cheap to drive.
    #[cfg(test)]
    fn with_watch_ring_cap(mut self, cap: usize) -> Self {
        self.watch_ring_cap = cap;
        self
    }

    /// The shared root resolver (so callers can advertise roots / record bindings).
    pub fn roots(&self) -> &std::sync::Arc<WorkspaceRoots> {
        &self.roots
    }

    /// Resolve a root id to its base directory and whether it is writable (`Host` is read-only).
    fn resolve(&self, root: &FsRootId) -> Result<(PathBuf, bool), ApiError> {
        match root {
            FsRootId::Host(id) => self
                .roots
                .host_root(id)
                .map(|p| (p, false))
                .ok_or_else(|| ApiError::Other(format!("unknown host browse root: {id}"))),
            FsRootId::Workspace => Ok((self.roots.workspace_root(), true)),
            FsRootId::Session(s) => Ok((self.roots.session_root(s.as_str()), true)),
        }
    }

    /// A stable string key for a root id (for the watch map).
    fn root_key(root: &FsRootId) -> String {
        match root {
            FsRootId::Host(id) => format!("host:{id}"),
            FsRootId::Workspace => "workspace".to_string(),
            FsRootId::Session(s) => format!("session:{}", s.as_str()),
        }
    }

    /// Contain a root-relative path against the resolved base (rejects `..`/absolute escapes).
    fn contained(base: &Path, rel: &str) -> Result<PathBuf, ApiError> {
        contain(base, Path::new(rel)).map_err(|e| ApiError::Other(format!("path not allowed: {e}")))
    }

    /// List one directory's children (root-relative `dir`, "" = the root). Entries matching the
    /// ignore set are marked `ignored`; when `show_ignored` is false they are dropped.
    pub async fn list(
        &self,
        root: &FsRootId,
        dir: &str,
        show_ignored: bool,
    ) -> Result<Vec<FsEntry>, ApiError> {
        let (base, _) = self.resolve(root)?;
        let abs = Self::contained(&base, dir)?;
        let mut rd = tokio::fs::read_dir(&abs)
            .await
            .map_err(|e| ApiError::Other(format!("read_dir {dir:?}: {e}")))?;
        let mut entries = Vec::new();
        while let Some(item) = rd
            .next_entry()
            .await
            .map_err(|e| ApiError::Other(format!("read_dir: {e}")))?
        {
            let name = item.file_name().to_string_lossy().to_string();
            let ignored = is_ignored_name(&name);
            if ignored && !show_ignored {
                continue;
            }
            let meta = match item.metadata().await {
                Ok(m) => m,
                Err(_) => continue,
            };
            entries.push(FsEntry {
                path: join_rel(dir, &name),
                name,
                kind: entry_kind(&meta),
                size: if meta.is_dir() { 0 } else { meta.len() },
                mtime_ms: mtime_ms(&meta),
                ignored,
            });
        }
        // Directories first, then case-insensitive name (a stable, IDE-like ordering).
        entries.sort_by(|a, b| {
            let ad = matches!(a.kind, FsEntryKind::Dir);
            let bd = matches!(b.kind, FsEntryKind::Dir);
            bd.cmp(&ad)
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });
        Ok(entries)
    }

    /// One entry's metadata.
    pub async fn stat(&self, root: &FsRootId, path: &str) -> Result<FsEntry, ApiError> {
        let (base, _) = self.resolve(root)?;
        let abs = Self::contained(&base, path)?;
        let meta = tokio::fs::metadata(&abs)
            .await
            .map_err(|e| ApiError::Other(format!("stat {path:?}: {e}")))?;
        let name = Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        Ok(FsEntry {
            ignored: is_ignored_name(&name),
            name,
            path: path.to_string(),
            kind: entry_kind(&meta),
            size: if meta.is_dir() { 0 } else { meta.len() },
            mtime_ms: mtime_ms(&meta),
        })
    }

    /// Read up to `max_bytes` (`0` = [`DEFAULT_MAX_READ`]) of a file + its etag + truncation flag.
    pub async fn read(
        &self,
        root: &FsRootId,
        path: &str,
        max_bytes: u64,
    ) -> Result<FsContent, ApiError> {
        let (base, _) = self.resolve(root)?;
        let abs = Self::contained(&base, path)?;
        let meta = tokio::fs::metadata(&abs)
            .await
            .map_err(|e| ApiError::Other(format!("stat {path:?}: {e}")))?;
        let cap = if max_bytes == 0 {
            DEFAULT_MAX_READ
        } else {
            max_bytes
        };
        let full = tokio::fs::read(&abs)
            .await
            .map_err(|e| ApiError::Other(format!("read {path:?}: {e}")))?;
        let truncated = full.len() as u64 > cap;
        let bytes = if truncated {
            full[..cap as usize].to_vec()
        } else {
            full
        };
        Ok(FsContent {
            bytes,
            revision: FsRevision {
                mtime_ms: mtime_ms(&meta),
                size: meta.len(),
            },
            truncated,
            // Populated by the node layer (NodeApiImpl::fs_read) when a content store is bound.
            blob_ref: None,
        })
    }

    /// The current etag of a file (or `None` if it does not exist).
    pub async fn revision(
        &self,
        root: &FsRootId,
        path: &str,
    ) -> Result<Option<FsRevision>, ApiError> {
        let (base, _) = self.resolve(root)?;
        let abs = Self::contained(&base, path)?;
        match tokio::fs::metadata(&abs).await {
            Ok(meta) => Ok(Some(FsRevision {
                mtime_ms: mtime_ms(&meta),
                size: meta.len(),
            })),
            Err(_) => Ok(None),
        }
    }

    /// Whether a root is writable (`Host` browse roots are read-only).
    pub fn writable(&self, root: &FsRootId) -> Result<bool, ApiError> {
        self.resolve(root).map(|(_, w)| w)
    }

    /// Write bytes to a contained path with optimistic concurrency: when `base_revision` is
    /// supplied it must match the file's current etag (else [`ApiError::Conflict`]). Returns the new
    /// etag. (Sensitive-path / approval / checkpoint gating is applied by the caller, which has the
    /// session/approval context.)
    pub async fn write(
        &self,
        root: &FsRootId,
        path: &str,
        bytes: &[u8],
        base_revision: Option<FsRevision>,
    ) -> Result<FsRevision, ApiError> {
        let (base, writable) = self.resolve(root)?;
        if !writable {
            return Err(ApiError::Unsupported(
                "host browse roots are read-only".into(),
            ));
        }
        let abs = Self::contained(&base, path)?;
        if let Some(expected) = base_revision {
            if let Ok(meta) = tokio::fs::metadata(&abs).await {
                let current = FsRevision {
                    mtime_ms: mtime_ms(&meta),
                    size: meta.len(),
                };
                if current != expected {
                    return Err(ApiError::Conflict(format!(
                        "stale base revision for {path:?} (file changed since read)"
                    )));
                }
            }
        }
        if let Some(parent) = abs.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ApiError::Other(format!("mkdir for {path:?}: {e}")))?;
        }
        tokio::fs::write(&abs, bytes)
            .await
            .map_err(|e| ApiError::Other(format!("write {path:?}: {e}")))?;
        let meta = tokio::fs::metadata(&abs)
            .await
            .map_err(|e| ApiError::Other(format!("stat after write {path:?}: {e}")))?;
        Ok(FsRevision {
            mtime_ms: mtime_ms(&meta),
            size: meta.len(),
        })
    }

    /// Server-side project search over a root (plain substring or regex), paginated. Walks the tree
    /// skipping ignored directories, scanning up to [`SEARCH_FILE_BUDGET`] files.
    pub async fn search(
        &self,
        root: &FsRootId,
        query: &FsSearchQuery,
    ) -> Result<FsSearchPage, ApiError> {
        let (base, _) = self.resolve(root)?;
        let needle = query.query.clone();
        if needle.is_empty() {
            return Ok(FsSearchPage::default());
        }
        let regex = if query.regex {
            let pat = if query.case_sensitive {
                needle.clone()
            } else {
                format!("(?i){needle}")
            };
            Some(regex::Regex::new(&pat).map_err(|e| ApiError::Other(format!("bad regex: {e}")))?)
        } else {
            None
        };
        let limit = if query.max_results == 0 {
            200
        } else {
            query.max_results as usize
        };
        let skip = (query.page as usize).saturating_mul(limit);
        let case_sensitive = query.case_sensitive;
        // Synchronous walk in a blocking task (small workspaces; bounded by SEARCH_FILE_BUDGET).
        let result = tokio::task::spawn_blocking(move || {
            let mut hits: Vec<FsSearchHit> = Vec::new();
            let mut visited = 0usize;
            let mut stack: Vec<PathBuf> = vec![base.clone()];
            let mut overflow = false;
            'walk: while let Some(dir) = stack.pop() {
                let rd = match std::fs::read_dir(&dir) {
                    Ok(rd) => rd,
                    Err(_) => continue,
                };
                for item in rd.flatten() {
                    let name = item.file_name().to_string_lossy().to_string();
                    if is_ignored_name(&name) {
                        continue;
                    }
                    let meta = match item.metadata() {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    let abs = item.path();
                    if meta.is_dir() {
                        stack.push(abs);
                        continue;
                    }
                    if meta.len() > SEARCH_FILE_CAP {
                        continue;
                    }
                    visited += 1;
                    if visited > SEARCH_FILE_BUDGET {
                        overflow = true;
                        break 'walk;
                    }
                    let text = match std::fs::read_to_string(&abs) {
                        Ok(t) => t,
                        Err(_) => continue, // binary / non-utf8
                    };
                    let rel = abs
                        .strip_prefix(&base)
                        .ok()
                        .map(|p| p.to_string_lossy().replace('\\', "/"))
                        .unwrap_or_default();
                    for (lineno, line) in text.lines().enumerate() {
                        let col = match &regex {
                            Some(re) => re.find(line).map(|m| m.start()),
                            None if case_sensitive => line.find(&needle),
                            None => line.to_lowercase().find(&needle.to_lowercase()),
                        };
                        if let Some(col) = col {
                            hits.push(FsSearchHit {
                                path: rel.clone(),
                                line: lineno as u32 + 1,
                                col: col as u32 + 1,
                                preview: line.trim().chars().take(200).collect(),
                            });
                        }
                    }
                }
            }
            (hits, overflow)
        })
        .await
        .map_err(|e| ApiError::Other(format!("search task: {e}")))?;
        let (all_hits, overflow) = result;
        let total = all_hits.len();
        let page: Vec<FsSearchHit> = all_hits.into_iter().skip(skip).take(limit).collect();
        let has_more = overflow || skip + page.len() < total;
        Ok(FsSearchPage {
            hits: page,
            has_more,
        })
    }

    /// Drain change events under a watched directory since `after_seq` (the cursor / long-poll form
    /// of the change stream). Implemented as an on-demand diff: each call re-scans the directory's
    /// top level and folds created/modified/removed events into a bounded per-dir ring keyed by a
    /// monotonic `seq`, so a client polls with the returned `next_seq`. The first call primes the
    /// snapshot and reports no changes.
    pub async fn watch_after(
        &self,
        root: &FsRootId,
        dir: &str,
        after_seq: u64,
        max: u32,
    ) -> Result<FsWatchPageView, ApiError> {
        let (base, _) = self.resolve(root)?;
        let abs = Self::contained(&base, dir)?;
        // Snapshot the directory's current entries (name -> (mtime, size)).
        let mut current: HashMap<String, (u64, u64)> = HashMap::new();
        if let Ok(mut rd) = tokio::fs::read_dir(&abs).await {
            while let Ok(Some(item)) = rd.next_entry().await {
                let name = item.file_name().to_string_lossy().to_string();
                if is_ignored_name(&name) {
                    continue;
                }
                if let Ok(meta) = item.metadata().await {
                    current.insert(
                        name,
                        (mtime_ms(&meta), if meta.is_dir() { 0 } else { meta.len() }),
                    );
                }
            }
        }
        let key = (Self::root_key(root), dir.to_string());
        let cap = self.watch_ring_cap;
        let mut state = self
            .watches
            .entry(key)
            .or_insert_with(|| WatchState::new(cap));
        if !state.primed {
            state.snapshot = current;
            state.primed = true;
            return Ok(FsWatchPageView {
                events: Vec::new(),
                next_seq: state.ring.head(),
                head_seq: state.ring.head(),
                reset: false,
            });
        }
        // Diff previous snapshot vs current to synthesize change events.
        let mut changes: Vec<FsChange> = Vec::new();
        for (name, sig) in &current {
            match state.snapshot.get(name) {
                None => changes.push(FsChange {
                    path: join_rel(dir, name),
                    kind: FsChangeKind::Created,
                }),
                Some(prev) if prev != sig => changes.push(FsChange {
                    path: join_rel(dir, name),
                    kind: FsChangeKind::Modified,
                }),
                _ => {}
            }
        }
        for name in state.snapshot.keys() {
            if !current.contains_key(name) {
                changes.push(FsChange {
                    path: join_rel(dir, name),
                    kind: FsChangeKind::Removed,
                });
            }
        }
        state.snapshot = current;
        for change in changes {
            // push handles the cap eviction + raising the resync floor past evicted events.
            state.ring.push(change);
        }
        // The reader's cursor aged out of the ring (events evicted past it): this page can't be a
        // complete delta, so flag `reset` and let the client re-list to reconcile.
        let reset = state.ring.lagged(after_seq);
        let events: Vec<FsChange> = state
            .ring
            .page(after_seq, max as usize)
            .into_iter()
            .map(|(_, c)| c)
            .collect();
        let head_seq = state.ring.head();
        Ok(FsWatchPageView {
            events,
            next_seq: head_seq,
            head_seq,
            reset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_root_sanitizes_and_joins() {
        let roots = WorkspaceRoots::new(PathBuf::from("/ws"));
        assert_eq!(roots.isolated_root("abc"), PathBuf::from("/ws/abc"));
        // Path-namespaced child ids collapse to one safe segment (no traversal).
        assert_eq!(
            roots.isolated_root("parent/c1"),
            PathBuf::from("/ws/parent_c1")
        );
    }

    #[test]
    fn recorded_binding_overrides_isolated() {
        let roots = WorkspaceRoots::new(PathBuf::from("/ws"));
        assert_eq!(roots.session_root("s1"), PathBuf::from("/ws/s1"));
        roots.record("s1", PathBuf::from("/srv/projects/foo"));
        assert_eq!(roots.session_root("s1"), PathBuf::from("/srv/projects/foo"));
    }

    #[test]
    fn host_root_resolves_allowlist() {
        let roots = WorkspaceRoots::new(PathBuf::from("/ws"))
            .with_browse_roots(vec![("home".into(), PathBuf::from("/home/me"))]);
        assert_eq!(roots.host_root("home"), Some(PathBuf::from("/home/me")));
        assert_eq!(roots.host_root("nope"), None);
    }

    fn temp_base(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("wfs-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    #[tokio::test]
    async fn write_read_list_round_trip_and_revision_conflict() {
        let base = temp_base("rw");
        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let root = FsRootId::Workspace;

        let rev = fs.write(&root, "a/b.txt", b"hello", None).await.unwrap();
        assert_eq!(rev.size, 5);

        let content = fs.read(&root, "a/b.txt", 0).await.unwrap();
        assert_eq!(content.bytes, b"hello");
        assert!(!content.truncated);

        // A stale base revision is rejected with Conflict.
        let stale = FsRevision {
            mtime_ms: 1,
            size: 1,
        };
        let err = fs
            .write(&root, "a/b.txt", b"x", Some(stale))
            .await
            .unwrap_err();
        assert!(matches!(err, ApiError::Conflict(_)));

        let entries = fs.list(&root, "a", false).await.unwrap();
        assert!(entries.iter().any(|e| e.name == "b.txt"));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn contain_rejects_escape_and_host_is_read_only() {
        let base = temp_base("escape");
        let fs = WorkspaceFs::new(std::sync::Arc::new(
            WorkspaceRoots::new(base.clone())
                .with_browse_roots(vec![("home".into(), base.clone())]),
        ));
        // Escapes above the root are rejected.
        assert!(fs.read(&FsRootId::Workspace, "../secret", 0).await.is_err());
        // Host browse roots are read-only.
        let err = fs
            .write(&FsRootId::Host("home".into()), "x.txt", b"y", None)
            .await
            .unwrap_err();
        assert!(matches!(err, ApiError::Unsupported(_)));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn bound_session_shares_dir_with_fs_surface() {
        // The engine rooted session "s1" at an operator-specified directory and recorded it; the FS
        // surface must resolve FsRootId::Session("s1") to that same in-place directory, so operator
        // (fs_*) and agent (fs/shell tools) see one filesystem.
        let bound = temp_base("bound");
        let wsbase = temp_base("wsbase");
        let roots = std::sync::Arc::new(WorkspaceRoots::new(wsbase.clone()));
        roots.record("s1", bound.clone());
        let fs = WorkspaceFs::new(roots.clone());
        let root = FsRootId::Session(daemon_common::SessionId::new("s1"));

        fs.write(&root, "note.md", b"shared", None).await.unwrap();
        // The bytes land in the bound directory in place.
        let on_disk = std::fs::read(bound.join("note.md")).unwrap();
        assert_eq!(on_disk, b"shared");
        // An unbound session falls back to the isolated sandbox under the workspace base.
        assert_eq!(roots.session_root("s2"), wsbase.join("s2"));

        let _ = std::fs::remove_dir_all(&bound);
        let _ = std::fs::remove_dir_all(&wsbase);
    }

    #[tokio::test]
    async fn watch_after_diffs_changes() {
        let base = temp_base("watch");
        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let root = FsRootId::Workspace;
        // First poll primes the snapshot (no events).
        let first = fs.watch_after(&root, "", 0, 0).await.unwrap();
        assert!(first.events.is_empty());
        // A new file shows up as a Created change on the next poll.
        fs.write(&root, "new.txt", b"hi", None).await.unwrap();
        let next = fs.watch_after(&root, "", first.next_seq, 0).await.unwrap();
        assert!(next
            .events
            .iter()
            .any(|c| c.path == "new.txt" && matches!(c.kind, FsChangeKind::Created)));
        let _ = std::fs::remove_dir_all(&base);
    }

    // A bounded ring that overflows past a stale cursor flags `reset` (the fs Lagged->Reset), so the
    // client knows to re-list rather than trust an incomplete delta. head_seq tracks the live edge.
    #[tokio::test]
    async fn watch_after_overflow_sets_reset() {
        let base = temp_base("watch-reset");
        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())))
            .with_watch_ring_cap(2);
        let root = FsRootId::Workspace;
        // Prime, then capture the cursor at seq 0 (a reader that will fall behind).
        let primed = fs.watch_after(&root, "", 0, 0).await.unwrap();
        assert!(!primed.reset);
        let stale_cursor = primed.next_seq; // 0

        // Generate > cap (2) changes across polls so the ring evicts the earliest seqs.
        for n in 0..5 {
            std::fs::write(base.join(format!("f{n}.txt")), b"x").unwrap();
            fs.watch_after(&root, "", 999, 0).await.unwrap(); // advance the ring (no reset; ahead)
        }

        // Polling from the stale cursor (0) now sees the floor risen past it -> reset.
        let page = fs.watch_after(&root, "", stale_cursor, 0).await.unwrap();
        assert!(page.reset, "an aged-out cursor must flag reset");
        assert!(
            page.head_seq >= 5,
            "head_seq tracks the live edge, got {}",
            page.head_seq
        );

        // A reader at the live edge is NOT reset.
        let fresh = fs.watch_after(&root, "", page.head_seq, 0).await.unwrap();
        assert!(!fresh.reset);
        let _ = std::fs::remove_dir_all(&base);
    }
}
