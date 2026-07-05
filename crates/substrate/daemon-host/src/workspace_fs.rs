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

use daemon_api::{
    clamp_page_max, ApiError, FsChange, FsChangeKind, FsContent, FsEntry, FsEntryKind, FsListPage,
    FsRevision, FsRootId, FsSearchHit, FsSearchPage, FsSearchQuery, FsWatchPageView, WIRE_PAGE_MAX,
};
use daemon_core::exec::{ContainedRoot, Meta};

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

/// The wire entry-kind for a [`ContainedRoot`] entry's non-following metadata.
fn entry_kind(meta: &Meta) -> FsEntryKind {
    if meta.is_symlink {
        FsEntryKind::Symlink
    } else if meta.is_dir {
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

    /// Open the resolved base as a [`ContainedRoot`] — every subsequent path op resolves via
    /// `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` from that root fd, so no relative path (at any
    /// component) can escape the root via a symlink or `..`, with no check-then-open TOCTOU.
    fn open_contained(base: &Path) -> Result<ContainedRoot, ApiError> {
        ContainedRoot::open(base).map_err(|e| ApiError::Other(format!("open root: {e}")))
    }

    /// List one directory's children (root-relative `dir`, "" = the root), paged at
    /// [`WIRE_PAGE_MAX`] entries per call. Entries matching the ignore set are marked `ignored`;
    /// when `show_ignored` is false they are dropped BEFORE pagination (the cursor walks the
    /// filtered + sorted listing). `after` is the previous page's `next` cursor (the last served
    /// entry's `path`); the returned `next` is `Some` iff more entries remain.
    pub async fn list(
        &self,
        root: &FsRootId,
        dir: &str,
        show_ignored: bool,
        after: Option<&str>,
    ) -> Result<FsListPage, ApiError> {
        let (base, _) = self.resolve(root)?;
        let cr = Self::open_contained(&base)?;
        // Contained, no-follow directory read: the directory is opened openat2-relative (a symlinked
        // component anywhere in `dir` is rejected) and each child's metadata is a non-following lstat.
        let children = cr
            .read_dir(Path::new(dir))
            .await
            .map_err(|e| ApiError::Other(format!("read_dir {dir:?}: {e}")))?;
        let mut entries = Vec::new();
        for child in children {
            let name = child.name;
            let ignored = is_ignored_name(&name);
            if ignored && !show_ignored {
                continue;
            }
            let meta = child.meta;
            entries.push(FsEntry {
                path: join_rel(dir, &name),
                name,
                kind: entry_kind(&meta),
                size: if meta.is_dir { 0 } else { meta.size },
                mtime_ms: meta.mtime_ms,
                ignored,
            });
        }
        // Directories first, then case-insensitive name (a stable, IDE-like ordering). This order
        // DEFINES the cursor: `next`/`after` is the last served entry's `path`.
        entries.sort_by(|a, b| {
            let ad = matches!(a.kind, FsEntryKind::Dir);
            let bd = matches!(b.kind, FsEntryKind::Dir);
            bd.cmp(&ad)
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });
        // Resume PAST the cursor entry: exact path match first (the normal case). A cursor whose
        // entry vanished between pages falls back to the first entry sorting after it under the
        // same order, treating the lost entry as a file (files sort last, so this never re-serves
        // entries; a deleted *dir* cursor can skip until the next re-list, and the deletion itself
        // fires the watch that triggers one).
        let start = match after {
            None => 0,
            Some(after) => match entries.iter().position(|e| e.path == after) {
                Some(idx) => idx + 1,
                None => {
                    let cursor_name = after.rsplit('/').next().unwrap_or(after).to_lowercase();
                    entries.partition_point(|e| {
                        matches!(e.kind, FsEntryKind::Dir) || e.name.to_lowercase() <= cursor_name
                    })
                }
            },
        };
        let mut page: Vec<FsEntry> = entries.split_off(start.min(entries.len()));
        let next = if page.len() > WIRE_PAGE_MAX {
            page.truncate(WIRE_PAGE_MAX);
            page.last().map(|e| e.path.clone())
        } else {
            None
        };
        Ok(FsListPage { items: page, next })
    }

    /// One entry's metadata.
    pub async fn stat(&self, root: &FsRootId, path: &str) -> Result<FsEntry, ApiError> {
        let (base, _) = self.resolve(root)?;
        // Non-following lstat on the final component, with the parent chain proven symlink-free
        // (openat2): a symlinked final component is reported as a link, never followed out of root.
        let meta = Self::open_contained(&base)?
            .symlink_metadata(Path::new(path))
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
            size: if meta.is_dir { 0 } else { meta.size },
            mtime_ms: meta.mtime_ms,
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
        let cap = if max_bytes == 0 {
            DEFAULT_MAX_READ
        } else {
            max_bytes
        };
        // openat2-relative read (a symlinked component anywhere is rejected); the revision metadata is
        // taken from the opened fd, so there is no stat-then-read TOCTOU. `meta.size` is the full file
        // size (the etag basis), independent of the returned/truncated byte count.
        let (bytes, meta, truncated) = Self::open_contained(&base)?
            .read_capped(Path::new(path), cap)
            .await
            .map_err(|e| ApiError::Other(format!("read {path:?}: {e}")))?;
        Ok(FsContent {
            bytes,
            revision: FsRevision {
                mtime_ms: meta.mtime_ms,
                size: meta.size,
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
        // Non-following lstat with a symlink-free parent chain; any escape/not-found yields None.
        match Self::open_contained(&base)?
            .symlink_metadata(Path::new(path))
            .await
        {
            Ok(meta) => Ok(Some(FsRevision {
                mtime_ms: meta.mtime_ms,
                size: meta.size,
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
        let cr = Self::open_contained(&base)?;
        if let Some(expected) = base_revision {
            // Non-following lstat (a symlinked final component is rejected on the write open anyway).
            if let Ok(meta) = cr.symlink_metadata(Path::new(path)).await {
                let current = FsRevision {
                    mtime_ms: meta.mtime_ms,
                    size: meta.size,
                };
                if current != expected {
                    return Err(ApiError::Conflict(format!(
                        "stale base revision for {path:?} (file changed since read)"
                    )));
                }
            }
        }
        // Parent dirs are created contained (each component O_NOFOLLOW) and the file is opened
        // openat2-relative, so a write can never clobber a file outside the root via a symlink; the
        // post-write revision is taken from the opened fd.
        let meta = cr
            .write(Path::new(path), bytes)
            .await
            .map_err(|e| ApiError::Other(format!("write {path:?}: {e}")))?;
        Ok(FsRevision {
            mtime_ms: meta.mtime_ms,
            size: meta.size,
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
        // Default AND clamp the page size to the wire bound: a >64-hit page is un-decodable by the
        // fixed-buffer client codec, so `0` ("server default", previously 200) and any larger
        // request both resolve to WIRE_PAGE_MAX per page; `page` serves the rest.
        let limit = clamp_page_max(query.max_results) as usize;
        let skip = (query.page as usize).saturating_mul(limit);
        let case_sensitive = query.case_sensitive;
        let cr = Self::open_contained(&base)?;
        // Synchronous fd-contained walk in a blocking task (small workspaces; bounded by
        // SEARCH_FILE_BUDGET). Every directory read and file read resolves openat2-relative from the
        // root fd, so a symlinked component anywhere is rejected; symlink entries are additionally
        // skipped so the walk never descends into or reads through a link out of the root.
        let result = tokio::task::spawn_blocking(move || {
            let mut hits: Vec<FsSearchHit> = Vec::new();
            let mut visited = 0usize;
            let mut stack: Vec<String> = vec![String::new()]; // "" = the root itself
            let mut overflow = false;
            'walk: while let Some(dir_rel) = stack.pop() {
                let children = match cr.read_dir_sync(Path::new(&dir_rel)) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                for child in children {
                    let name = child.name;
                    if is_ignored_name(&name) {
                        continue;
                    }
                    let meta = child.meta;
                    // Skip symlink entries entirely — never recurse into a symlinked directory nor
                    // read through a symlinked file out of the root.
                    if meta.is_symlink {
                        continue;
                    }
                    let child_rel = if dir_rel.is_empty() {
                        name.clone()
                    } else {
                        format!("{dir_rel}/{name}")
                    };
                    if meta.is_dir {
                        stack.push(child_rel);
                        continue;
                    }
                    if meta.size > SEARCH_FILE_CAP {
                        continue;
                    }
                    visited += 1;
                    if visited > SEARCH_FILE_BUDGET {
                        overflow = true;
                        break 'walk;
                    }
                    let text = match cr.read_sync(Path::new(&child_rel)) {
                        Ok(bytes) => match String::from_utf8(bytes) {
                            Ok(t) => t,
                            Err(_) => continue, // binary / non-utf8
                        },
                        Err(_) => continue,
                    };
                    let rel = child_rel.replace('\\', "/");
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
        // Contained, no-follow snapshot of the watched directory's current entries (a symlinked
        // component anywhere in `dir` is rejected; per-entry metadata is a non-following lstat).
        let mut current: HashMap<String, (u64, u64)> = HashMap::new();
        if let Ok(children) = Self::open_contained(&base)?.read_dir(Path::new(dir)).await {
            for child in children {
                if is_ignored_name(&child.name) {
                    continue;
                }
                let meta = child.meta;
                current.insert(
                    child.name,
                    (meta.mtime_ms, if meta.is_dir { 0 } else { meta.size }),
                );
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

        let page = fs.list(&root, "a", false, None).await.unwrap();
        assert!(page.items.iter().any(|e| e.name == "b.txt"));
        assert!(page.next.is_none(), "a small dir fits one page");

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

    // Wire pagination (v24): a 150-entry directory lists to completion across 3 pages (64/64/22)
    // in the stable dirs-first order with no duplicates or misses, and only the last page carries
    // next = None. Mixed kinds prove the cursor respects the dirs-before-files ordering.
    #[tokio::test]
    async fn list_pages_large_dir_to_completion() {
        let base = temp_base("paginate");
        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let root = FsRootId::Workspace;
        for n in 0..10 {
            std::fs::create_dir(base.join(format!("d{n:02}"))).unwrap();
        }
        for n in 0..140 {
            std::fs::write(base.join(format!("f{n:03}.txt")), b"x").unwrap();
        }

        let mut sizes = Vec::new();
        let mut all: Vec<String> = Vec::new();
        let mut after: Option<String> = None;
        loop {
            let page = fs.list(&root, "", false, after.as_deref()).await.unwrap();
            assert!(page.items.len() <= WIRE_PAGE_MAX, "page exceeds the bound");
            sizes.push(page.items.len());
            all.extend(page.items.iter().map(|e| e.path.clone()));
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
            assert!(sizes.len() <= 10, "runaway pagination");
        }

        assert_eq!(sizes, vec![64, 64, 22]);
        let expected: Vec<String> = (0..10)
            .map(|n| format!("d{n:02}"))
            .chain((0..140).map(|n| format!("f{n:03}.txt")))
            .collect();
        assert_eq!(all, expected, "stable order, no duplicates, no misses");
        let _ = std::fs::remove_dir_all(&base);
    }

    // A cursor whose entry was deleted between pages still resumes at the right position (the
    // first entry sorting after it), with no duplicates.
    #[tokio::test]
    async fn list_cursor_survives_deleted_entry() {
        let base = temp_base("cursor-del");
        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let root = FsRootId::Workspace;
        for n in 0..100 {
            std::fs::write(base.join(format!("f{n:03}.txt")), b"x").unwrap();
        }

        let first = fs.list(&root, "", false, None).await.unwrap();
        assert_eq!(first.items.len(), WIRE_PAGE_MAX);
        let cursor = first.next.clone().expect("more pages remain");
        assert_eq!(cursor, "f063.txt", "the cursor is the last served path");

        // The cursor entry vanishes between pages; the resume continues at the next entry.
        std::fs::remove_file(base.join("f063.txt")).unwrap();
        let second = fs.list(&root, "", false, Some(&cursor)).await.unwrap();
        assert_eq!(
            second.items.first().map(|e| e.name.as_str()),
            Some("f064.txt"),
            "resume lands on the entry after the deleted cursor"
        );
        assert_eq!(second.items.len(), 36, "the remaining tail in one page");
        assert!(second.next.is_none());
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

    // Cluster C interim guard: a symlink inside the workspace pointing at a file OUTSIDE it is
    // lexically contained, but the read must be refused (O_NOFOLLOW), not followed to the secret.
    #[cfg(unix)]
    #[tokio::test]
    async fn read_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;
        let base = temp_base("sym-read");
        let outside = temp_base("sym-read-secret");
        let secret = outside.join("secret.txt");
        std::fs::write(&secret, b"TOP SECRET").unwrap();
        symlink(&secret, base.join("link.txt")).unwrap();

        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let res = fs.read(&FsRootId::Workspace, "link.txt", 0).await;
        assert!(
            res.is_err(),
            "reading through an escaping symlink must be rejected, got {res:?}"
        );
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // The write must not clobber a file outside the root through a symlinked final component.
    #[cfg(unix)]
    #[tokio::test]
    async fn write_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;
        let base = temp_base("sym-write");
        let outside = temp_base("sym-write-target");
        let target = outside.join("target.txt");
        std::fs::write(&target, b"ORIGINAL").unwrap();
        symlink(&target, base.join("link.txt")).unwrap();

        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let res = fs
            .write(&FsRootId::Workspace, "link.txt", b"OVERWRITTEN", None)
            .await;
        assert!(
            res.is_err(),
            "writing through an escaping symlink must be rejected, got {res:?}"
        );
        assert_eq!(
            std::fs::read(&target).unwrap(),
            b"ORIGINAL",
            "the outside target must not be written through the symlink"
        );
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // `search` must not read through a symlink to a file outside the root (no hit from the target).
    #[cfg(unix)]
    #[tokio::test]
    async fn search_skips_symlinked_target() {
        use std::os::unix::fs::symlink;
        let base = temp_base("sym-search");
        let outside = temp_base("sym-search-secret");
        std::fs::write(outside.join("secret.txt"), b"NEEDLE_abc123").unwrap();
        symlink(outside.join("secret.txt"), base.join("link.txt")).unwrap();
        // A real in-root file WITHOUT the needle, to prove search itself works.
        std::fs::write(base.join("real.txt"), b"nothing here").unwrap();

        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let query = FsSearchQuery {
            query: "NEEDLE_abc123".into(),
            regex: false,
            case_sensitive: true,
            max_results: 0,
            page: 0,
        };
        let page = fs.search(&FsRootId::Workspace, &query).await.unwrap();
        assert!(
            page.hits.is_empty(),
            "search must not read through a symlink to an outside file, got {:?}",
            page.hits
        );
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // Listing a symlinked directory final component is refused, so it cannot enumerate an outside
    // directory through a link.
    #[cfg(unix)]
    #[tokio::test]
    async fn list_rejects_symlinked_subdir() {
        use std::os::unix::fs::symlink;
        let base = temp_base("sym-list");
        let outside = temp_base("sym-list-target");
        std::fs::write(outside.join("hidden.txt"), b"x").unwrap();
        symlink(&outside, base.join("sublink")).unwrap();

        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let res = fs.list(&FsRootId::Workspace, "sublink", true, None).await;
        assert!(
            res.is_err(),
            "listing a symlinked subdir must be rejected, got {res:?}"
        );
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // No-regression: the guard exempts the trusted root, so a session bound onto a directory that is
    // itself a symlink still lists (only components strictly BELOW the root are guarded).
    #[cfg(unix)]
    #[tokio::test]
    async fn symlinked_bound_root_still_lists() {
        use std::os::unix::fs::symlink;
        let real = temp_base("sym-boundroot-real");
        std::fs::write(real.join("inside.txt"), b"hi").unwrap();
        let wsbase = temp_base("sym-boundroot-ws");
        // A symlink that will be recorded as session s1's root.
        let link = wsbase.join("s1link");
        symlink(&real, &link).unwrap();

        let roots = std::sync::Arc::new(WorkspaceRoots::new(wsbase.clone()));
        roots.record("s1", link.clone());
        let fs = WorkspaceFs::new(roots);
        let root = FsRootId::Session(daemon_common::SessionId::new("s1"));
        let page = fs.list(&root, "", true, None).await.unwrap();
        assert!(
            page.items.iter().any(|e| e.name == "inside.txt"),
            "a symlinked bound root must still list its contents"
        );
        let _ = std::fs::remove_dir_all(&real);
        let _ = std::fs::remove_dir_all(&wsbase);
    }

    // Phase 3 (ContainedRoot): the case the interim guard did NOT close — an INTERMEDIATE path
    // component is a symlink out of the root. The interim only re-verified the FINAL component, so a
    // read/list through a symlinked PARENT dir followed the link out of the root. `openat2`
    // (RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS) rejects a symlink at any component. FAILS on the
    // interim tree (escape succeeds), passes after the fix.
    #[cfg(unix)]
    #[tokio::test]
    async fn read_rejects_intermediate_symlink_escape() {
        use std::os::unix::fs::symlink;
        let base = temp_base("isym-read");
        let outside = temp_base("isym-read-secret");
        std::fs::write(outside.join("secret.txt"), b"TOP SECRET").unwrap();
        symlink(&outside, base.join("sub")).unwrap();

        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let res = fs.read(&FsRootId::Workspace, "sub/secret.txt", 0).await;
        assert!(
            res.is_err(),
            "reading through an intermediate symlinked dir must be rejected, got {res:?}"
        );
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn list_rejects_intermediate_symlink() {
        use std::os::unix::fs::symlink;
        let base = temp_base("isym-list");
        let outside = temp_base("isym-list-target");
        std::fs::create_dir_all(outside.join("inner")).unwrap();
        std::fs::write(outside.join("inner").join("hidden.txt"), b"x").unwrap();
        symlink(&outside, base.join("sub")).unwrap();

        let fs = WorkspaceFs::new(std::sync::Arc::new(WorkspaceRoots::new(base.clone())));
        let res = fs.list(&FsRootId::Workspace, "sub/inner", true, None).await;
        assert!(
            res.is_err(),
            "listing through an intermediate symlinked dir must be rejected, got {res:?}"
        );
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_dir_all(&outside);
    }
}
