// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`ContainedRoot`] — the Phase 3 (Cluster C) filesystem containment capability.
//!
//! A validated location under a workspace root is representable **only** as an operation on an open
//! root directory fd, never as a re-resolvable `PathBuf`. Every attacker-influenced *relative* path
//! is resolved via `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS)` from that
//! fd (Linux) — the kernel evaluates the whole path atomically and fails the instant *any* component
//! (intermediate or final) is a symlink or would climb above the root. This eliminates, as a class,
//! both symlink-escape and the check-then-open TOCTOU that the lexical [`super::contain`] +
//! [`super`]'s (now-removed) interim `O_NOFOLLOW`/`lstat` guards only partially closed (they guarded
//! the *final* component; an intermediate symlinked directory was still followed).
//!
//! Backend: `rustix::fs::openat2` **directly** (not `cap-std`) — it is already in the dependency graph
//! transitively, gives exact `ResolveFlags` control, and its API is safe, so `daemon-core`'s
//! `#![forbid(unsafe_code)]` is preserved (there is no `unsafe` in this module).
//!
//! Platform matrix (honest residuals):
//! - **Linux:** `openat2` with the three RESOLVE flags — atomic, no residual on the guarded ops.
//!   Falls back to the unix component walk on `ENOSYS`/`EPERM` (kernel < 5.6 or a seccomp filter).
//! - **Other unix (macOS):** a fd-relative component walk — each segment is `openat`'d from its
//!   parent fd with `O_NOFOLLOW`, so a symlink at *any* component fails. Still fd-relative (never
//!   re-resolves a path string), so it closes the intermediate-symlink class too; the only residual
//!   is a directory-rename race *between* opening component i and i+1 (far weaker than a full re-walk;
//!   `openat2` has none).
//! - **Non-unix (Windows v1 stub-worker lane, ships no engine fs tools):** best-effort lexical
//!   [`super::contain`] + a `symlink_metadata` reject on the final component. Documented as
//!   best-effort; kept only so the crate builds on the Windows cross target.
//!
//! ## Consumers
//! Constructed from a workspace/session root (`LocalEnvironment`, the node `WorkspaceFs` surface, the
//! `fs`/`shell` tools, cron, the fleet job worker). **The Phase 3 `exec-os-sandbox` track also
//! consumes this type** to contain `execute_code`'s host-side staging directory
//! (`<ws_root>/.execute_code/<run_id>`), which a child-process kernel sandbox cannot cover — build a
//! `ContainedRoot::open(ws_root)` and use [`ContainedRoot::create_dir_all`] / [`ContainedRoot::write`].

use std::path::PathBuf;

/// Lightweight, cross-platform metadata for a contained entry (a subset of [`std::fs::Metadata`] that
/// the fs surfaces actually consume — kind + size + mtime), so callers never need the raw fd.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Meta {
    /// The entry is a directory.
    pub is_dir: bool,
    /// The entry is a regular file.
    pub is_file: bool,
    /// The entry is a symbolic link (reported by the non-following lstat, never traversed).
    pub is_symlink: bool,
    /// File size in bytes (`0` for directories).
    pub size: u64,
    /// Last-modified time in milliseconds since the unix epoch (`0` if unavailable).
    pub mtime_ms: u64,
    /// The unix permission bits (`st_mode & 0o7777`); `0` on non-unix or when unavailable. Used to
    /// preserve an existing file's mode across the `fs` tool's atomic replace.
    pub mode: u32,
}

/// One directory child from [`ContainedRoot::read_dir`]: its name plus non-following metadata.
#[derive(Clone, Debug)]
pub struct DirEntryLite {
    /// The entry's file name (single path segment).
    pub name: String,
    /// The entry's non-following (lstat) metadata.
    pub meta: Meta,
}

/// A spawn-safe working directory produced by [`ContainedRoot::child_cwd`], holding the verifying dir
/// fd alive for the duration a child is spawned. On Linux `path` is `/proc/self/fd/<n>` — the kernel
/// resolves the fd's target at spawn with no path re-resolution (closing the verify→spawn window);
/// on other platforms `path` is the verified absolute path (documented residual: a dir→symlink swap
/// between verify and spawn, which the `exec-os-sandbox` track owns hardening).
pub struct ChildCwd {
    /// The path to hand to `Command::current_dir`.
    pub path: PathBuf,
    #[cfg(unix)]
    _keepalive: Option<std::os::fd::OwnedFd>,
}

// =================================================================================================
// unix implementation (Linux openat2 fast path + cross-unix openat component-walk fallback)
// =================================================================================================
#[cfg(unix)]
mod imp {
    use super::{ChildCwd, DirEntryLite, Meta};
    use std::io;
    use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
    use std::path::{Component, Path, PathBuf};
    use std::sync::Arc;

    use rustix::fs::{Mode, OFlags};

    const DIR_MODE: Mode = Mode::from_bits_retain(0o755);
    const FILE_MODE: Mode = Mode::from_bits_retain(0o644);

    /// A validated filesystem containment boundary: an open root directory fd whose methods resolve
    /// every relative path via `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS)`
    /// (Linux) or an `openat(O_NOFOLLOW)` component walk (other unix). See the module docs.
    #[derive(Clone)]
    pub struct ContainedRoot {
        root_fd: Arc<OwnedFd>,
        /// The root's absolute path — for policy checks, display, spawn cwd, and the grep/glob walker
        /// entry point ONLY. Never used to open a relative target (that is always fd-relative).
        root_abs: PathBuf,
    }

    impl ContainedRoot {
        /// Open `root` as a containment boundary, creating it if missing (matches the historical
        /// lazy `ensure_root`). The root itself is opened by path — following symlinks ONCE, since an
        /// operator may legitimately bind a workspace onto a symlinked directory (e.g. macOS
        /// `/tmp -> /private/tmp`); only RELATIVE lookups from the returned fd are symlink-hardened.
        pub fn open(root: &Path) -> io::Result<Self> {
            std::fs::create_dir_all(root)?;
            let root_fd = rustix::fs::open(
                root,
                OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::RDONLY,
                Mode::empty(),
            )
            .map_err(to_io)?;
            Ok(Self {
                root_fd: Arc::new(root_fd),
                root_abs: root.to_path_buf(),
            })
        }

        /// The root's absolute path (policy / display only — never used to open a relative target).
        pub fn root(&self) -> &Path {
            &self.root_abs
        }

        /// A **policy/display-only** absolute path for `rel` (lexical join + normalize). It performs
        /// NO filesystem access and MUST NOT be used to open anything — it exists solely so callers
        /// that check a path against an allow/deny prefix (e.g. the `fs` tool's protected-path guard)
        /// keep working. Actual opens always go through the fd-relative methods.
        pub fn resolve_display(&self, rel: &Path) -> io::Result<PathBuf> {
            super::super::contain(&self.root_abs, rel)
        }

        // -- async surface (byte I/O + resolution off the reactor via spawn_blocking) --------------

        /// Read a contained file in full.
        pub async fn read(&self, rel: &Path) -> io::Result<Vec<u8>> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.read_sync(&rel)).await
        }

        /// Read up to `cap` bytes of a contained file; returns `(bytes, meta, truncated)` with the
        /// metadata taken from the opened fd (no stat↔read race).
        pub async fn read_capped(&self, rel: &Path, cap: u64) -> io::Result<(Vec<u8>, Meta, bool)> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.read_capped_sync(&rel, cap)).await
        }

        /// Write bytes to a contained file (create + truncate), creating parent dirs; returns the
        /// post-write metadata from the opened fd.
        pub async fn write(&self, rel: &Path, bytes: &[u8]) -> io::Result<Meta> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            let bytes = bytes.to_vec();
            blocking(move || this.write_sync(&rel, &bytes)).await
        }

        /// List a contained directory's children with non-following metadata (sorted by name).
        pub async fn read_dir(&self, rel: &Path) -> io::Result<Vec<DirEntryLite>> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.read_dir_sync(&rel)).await
        }

        /// Non-following metadata of a contained entry (lstat semantics on the final component; the
        /// parent chain is proven symlink-free).
        pub async fn symlink_metadata(&self, rel: &Path) -> io::Result<Meta> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.symlink_metadata_sync(&rel)).await
        }

        /// Create a directory and all missing parents, contained (each component `O_NOFOLLOW`).
        pub async fn create_dir_all(&self, rel: &Path) -> io::Result<()> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.create_dir_all_sync(&rel)).await
        }

        /// Remove a contained regular file.
        pub async fn remove_file(&self, rel: &Path) -> io::Result<()> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.remove_file_sync(&rel)).await
        }

        /// Remove a contained (empty) directory.
        pub async fn remove_dir(&self, rel: &Path) -> io::Result<()> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.remove_dir_sync(&rel)).await
        }

        /// Rename `from` to `to`, both contained beneath the root.
        pub async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
            let this = self.clone();
            let from = from.to_path_buf();
            let to = to.to_path_buf();
            blocking(move || this.rename_sync(&from, &to)).await
        }

        /// Set the unix permission bits of a contained file (opened openat2-relative, so no symlink is
        /// followed) — used to preserve an existing file's mode across the `fs` tool's atomic replace.
        pub async fn set_mode(&self, rel: &Path, mode: u32) -> io::Result<()> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.set_mode_sync(&rel, mode)).await
        }

        fn set_mode_sync(&self, rel: &Path, mode: u32) -> io::Result<()> {
            let fd = self.open_at(rel, OFlags::RDONLY, Mode::empty())?;
            rustix::fs::fchmod(&fd, Mode::from_bits_retain(mode)).map_err(to_io)
        }

        /// Prove `rel` names a directory reachable with no symlink escape and return its verified
        /// absolute path — for path-based subsystems (grep/glob walker, process cwd) that cannot take
        /// an fd. The walker must be configured `follow_links(false)`; the returned path is the entry
        /// point only.
        pub async fn verify_dir(&self, rel: &Path) -> io::Result<PathBuf> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.verify_dir_sync(&rel)).await
        }

        /// A spawn-safe, contained working directory for a child process (creates it if missing). See
        /// [`ChildCwd`].
        pub async fn child_cwd(&self, rel: &Path) -> io::Result<ChildCwd> {
            let this = self.clone();
            let rel = rel.to_path_buf();
            blocking(move || this.child_cwd_sync(&rel)).await
        }

        // -- sync core (also called directly by blocking walkers like fs_search) --------------------

        /// Read a contained file in full (sync; safe to call from within a blocking task).
        pub fn read_sync(&self, rel: &Path) -> io::Result<Vec<u8>> {
            use std::io::Read as _;
            let fd = self.open_at(rel, OFlags::RDONLY, Mode::empty())?;
            let mut file = std::fs::File::from(fd);
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            Ok(buf)
        }

        fn read_capped_sync(&self, rel: &Path, cap: u64) -> io::Result<(Vec<u8>, Meta, bool)> {
            use std::io::Read as _;
            let fd = self.open_at(rel, OFlags::RDONLY, Mode::empty())?;
            let mut file = std::fs::File::from(fd);
            let meta = meta_from_std(&file.metadata()?);
            let mut full = Vec::new();
            file.read_to_end(&mut full)?;
            let truncated = full.len() as u64 > cap;
            let bytes = if truncated {
                full[..cap as usize].to_vec()
            } else {
                full
            };
            Ok((bytes, meta, truncated))
        }

        fn write_sync(&self, rel: &Path, bytes: &[u8]) -> io::Result<Meta> {
            use std::io::Write as _;
            if let Some(parent) = rel.parent() {
                if !parent.as_os_str().is_empty() {
                    self.create_dir_all_sync(parent)?;
                }
            }
            let fd = self.open_at(
                rel,
                OFlags::WRONLY | OFlags::CREATE | OFlags::TRUNC,
                FILE_MODE,
            )?;
            let mut file = std::fs::File::from(fd);
            file.write_all(bytes)?;
            let meta = meta_from_std(&file.metadata()?);
            Ok(meta)
        }

        /// Open a contained file for writing (create + truncate) and hand back the raw fd as a
        /// [`std::fs::File`] — for callers that manage their own write/rename dance (the `fs` tool's
        /// atomic temp write). Creates parent dirs.
        pub fn open_write_file(&self, rel: &Path) -> io::Result<std::fs::File> {
            if let Some(parent) = rel.parent() {
                if !parent.as_os_str().is_empty() {
                    self.create_dir_all_sync(parent)?;
                }
            }
            let fd = self.open_at(
                rel,
                OFlags::WRONLY | OFlags::CREATE | OFlags::TRUNC,
                FILE_MODE,
            )?;
            Ok(std::fs::File::from(fd))
        }

        /// List a contained directory's children with non-following metadata (sync).
        pub fn read_dir_sync(&self, rel: &Path) -> io::Result<Vec<DirEntryLite>> {
            let dir_fd = self.open_at(rel, OFlags::RDONLY | OFlags::DIRECTORY, Mode::empty())?;
            let mut out = Vec::new();
            // Iterate names/d_type via a Dir over a clone, then lstat each child from the dir fd.
            let iter_fd = dir_fd.try_clone()?;
            let dir = rustix::fs::Dir::read_from(&iter_fd).map_err(to_io)?;
            for entry in dir {
                let entry = entry.map_err(to_io)?;
                let raw = entry.file_name().to_bytes();
                if raw == b"." || raw == b".." {
                    continue;
                }
                let name = String::from_utf8_lossy(raw).into_owned();
                // Non-following lstat of just this child, from the (contained) directory fd.
                let meta = match rustix::fs::statat(
                    &dir_fd,
                    entry.file_name(),
                    rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
                ) {
                    Ok(st) => meta_from_stat(&st),
                    Err(_) => continue,
                };
                out.push(DirEntryLite { name, meta });
            }
            out.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(out)
        }

        fn symlink_metadata_sync(&self, rel: &Path) -> io::Result<Meta> {
            // Resolve the PARENT dir contained (no symlink escape in the chain), then lstat only the
            // final component from that fd — reporting a final-component symlink as a link, never
            // following it out of the root.
            let rel = self.norm(rel)?;
            match split_parent(&rel) {
                None => {
                    // The root itself.
                    let st = rustix::fs::fstat(&*self.root_fd).map_err(to_io)?;
                    Ok(meta_from_stat(&st))
                }
                Some((parent, name)) => {
                    let parent_fd =
                        self.open_at(parent, OFlags::RDONLY | OFlags::DIRECTORY, Mode::empty())?;
                    let st = rustix::fs::statat(
                        &parent_fd,
                        name.as_os_str(),
                        rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
                    )
                    .map_err(to_io)?;
                    Ok(meta_from_stat(&st))
                }
            }
        }

        /// Create `rel` and all missing parents, each component opened `O_NOFOLLOW` from its parent fd
        /// (so a symlinked component is rejected — no escape via a symlinked intermediate dir).
        pub fn create_dir_all_sync(&self, rel: &Path) -> io::Result<()> {
            let rel = self.norm(rel)?;
            let mut cur: OwnedFd = self.root_fd.try_clone()?;
            for seg in normal_segments(&rel)? {
                // Create if missing (ignore EEXIST), then descend into it with O_NOFOLLOW.
                match rustix::fs::mkdirat(&cur, seg.as_os_str(), DIR_MODE) {
                    Ok(()) => {}
                    Err(rustix::io::Errno::EXIST) => {
                        // The component already exists: reject it outright if it is a symlink, so a
                        // symlinked intermediate/final directory can never be traversed (surfaced as
                        // PermissionDenied, matching openat2's ELOOP mapping — an O_NOFOLLOW openat
                        // would otherwise fail with the less-precise ENOTDIR).
                        let st = rustix::fs::statat(
                            &cur,
                            seg.as_os_str(),
                            rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
                        )
                        .map_err(to_io)?;
                        if rustix::fs::FileType::from_raw_mode(st.st_mode as rustix::fs::RawMode)
                            == rustix::fs::FileType::Symlink
                        {
                            return Err(escape());
                        }
                    }
                    Err(e) => return Err(to_io(e)),
                }
                let next = rustix::fs::openat(
                    &cur,
                    seg.as_os_str(),
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(to_io)?;
                cur = next;
            }
            Ok(())
        }

        fn remove_file_sync(&self, rel: &Path) -> io::Result<()> {
            let (parent_fd, name) = self.parent_fd_and_name(rel)?;
            rustix::fs::unlinkat(&parent_fd, name.as_os_str(), rustix::fs::AtFlags::empty())
                .map_err(to_io)
        }

        fn remove_dir_sync(&self, rel: &Path) -> io::Result<()> {
            let (parent_fd, name) = self.parent_fd_and_name(rel)?;
            rustix::fs::unlinkat(&parent_fd, name.as_os_str(), rustix::fs::AtFlags::REMOVEDIR)
                .map_err(to_io)
        }

        fn rename_sync(&self, from: &Path, to: &Path) -> io::Result<()> {
            let (from_parent, from_name) = self.parent_fd_and_name(from)?;
            let (to_parent, to_name) = self.parent_fd_and_name(to)?;
            rustix::fs::renameat(
                &from_parent,
                from_name.as_os_str(),
                &to_parent,
                to_name.as_os_str(),
            )
            .map_err(to_io)
        }

        /// Prove `rel` is a contained directory and return its verified absolute path (sync).
        pub fn verify_dir_sync(&self, rel: &Path) -> io::Result<PathBuf> {
            let _fd = self.open_at(rel, OFlags::RDONLY | OFlags::DIRECTORY, Mode::empty())?;
            // openat2 (or the O_NOFOLLOW walk) proved no symlink escape; the lexical join is now a
            // faithful absolute path for the (follow_links=false) walker / process cwd.
            super::super::contain(&self.root_abs, rel)
        }

        fn child_cwd_sync(&self, rel: &Path) -> io::Result<ChildCwd> {
            self.create_dir_all_sync(rel)?;
            let fd = self.open_at(rel, OFlags::RDONLY | OFlags::DIRECTORY, Mode::empty())?;
            #[cfg(target_os = "linux")]
            {
                // The kernel resolves the fd's target at chdir (pre-exec), with no path re-resolution;
                // the fd is inherited across fork and closed at exec (CLOEXEC), so no leak.
                let path = PathBuf::from(format!("/proc/self/fd/{}", fd.as_raw_fd()));
                Ok(ChildCwd {
                    path,
                    _keepalive: Some(fd),
                })
            }
            #[cfg(not(target_os = "linux"))]
            {
                drop(fd);
                Ok(ChildCwd {
                    path: super::super::contain(&self.root_abs, rel)?,
                    _keepalive: None,
                })
            }
        }

        // -- resolution core (the ONE place the RESOLVE flags live) --------------------------------

        /// Normalize an incoming path to a clean root-relative path (no `.`/`..`/absolute prefix)
        /// using the lexical [`super::super::contain`] floor, which also accepts an absolute path that
        /// is *within* the root (callers sometimes pass the already-contained absolute path, e.g. the
        /// shell tool's sticky cwd) and rejects a lexical escape. The returned path feeds openat2,
        /// which then enforces the symlink/TOCTOU containment the lexical floor cannot.
        fn norm(&self, path: &Path) -> io::Result<PathBuf> {
            let abs = super::super::contain(&self.root_abs, path)?;
            Ok(abs
                .strip_prefix(&self.root_abs)
                .unwrap_or(Path::new(""))
                .to_path_buf())
        }

        /// Resolve `rel` from the root fd, rejecting any symlinked or out-of-root component.
        fn open_at(&self, rel: &Path, oflags: OFlags, mode: Mode) -> io::Result<OwnedFd> {
            let rel = self.norm(rel)?;
            resolve_open(self.root_fd.as_fd(), &rel, oflags, mode)
        }

        fn parent_fd_and_name(&self, rel: &Path) -> io::Result<(OwnedFd, PathBuf)> {
            let rel = self.norm(rel)?;
            match split_parent(&rel) {
                None => Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "operation requires a name below the root",
                )),
                Some((parent, name)) => {
                    let parent_fd =
                        self.open_at(parent, OFlags::RDONLY | OFlags::DIRECTORY, Mode::empty())?;
                    Ok((parent_fd, name.to_path_buf()))
                }
            }
        }
    }

    // -- resolution backends --------------------------------------------------------------------

    #[cfg(target_os = "linux")]
    fn resolve_open(
        root: BorrowedFd,
        rel: &Path,
        oflags: OFlags,
        mode: Mode,
    ) -> io::Result<OwnedFd> {
        use rustix::fs::{openat2, ResolveFlags};
        let path = if rel.as_os_str().is_empty() {
            Path::new(".")
        } else {
            rel
        };
        // Lexically reject climbs up front so a bare `..` returns PermissionDenied uniformly.
        reject_escape(rel)?;
        let resolve =
            ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS;
        match openat2(root, path, oflags | OFlags::CLOEXEC, mode, resolve) {
            Ok(fd) => Ok(fd),
            // openat2 landed in Linux 5.6; a seccomp'd/older kernel returns ENOSYS/EPERM -> walk.
            Err(rustix::io::Errno::NOSYS)
            | Err(rustix::io::Errno::PERM)
            | Err(rustix::io::Errno::OPNOTSUPP) => walk_openat(root, rel, oflags, mode),
            Err(e) => Err(to_io(e)),
        }
    }

    #[cfg(all(unix, not(target_os = "linux")))]
    fn resolve_open(
        root: BorrowedFd,
        rel: &Path,
        oflags: OFlags,
        mode: Mode,
    ) -> io::Result<OwnedFd> {
        walk_openat(root, rel, oflags, mode)
    }

    /// Cross-unix fallback (macOS; and Linux < 5.6 / seccomp): walk `rel` component by component,
    /// opening each from its parent fd with `O_NOFOLLOW`, so a symlink at ANY component fails
    /// (`ELOOP`). Never re-resolves a full path string, so the intermediate-symlink class is closed;
    /// residual is a per-component rename race (documented in the module header).
    fn walk_openat(
        root: BorrowedFd,
        rel: &Path,
        oflags: OFlags,
        mode: Mode,
    ) -> io::Result<OwnedFd> {
        reject_escape(rel)?;
        let segs = normal_segments(rel)?;
        if segs.is_empty() {
            // The root itself: dup with the requested flags via `.` (root is the trust anchor).
            return rustix::fs::openat(
                root,
                ".",
                oflags | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                mode,
            )
            .map_err(to_io);
        }
        let mut cur: Option<OwnedFd> = None;
        let last = segs.len() - 1;
        for (i, seg) in segs.iter().enumerate() {
            let parent = cur.as_ref().map(|f| f.as_fd()).unwrap_or(root);
            let fd = if i == last {
                rustix::fs::openat(
                    parent,
                    seg.as_os_str(),
                    oflags | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    mode,
                )
                .map_err(to_io)?
            } else {
                rustix::fs::openat(
                    parent,
                    seg.as_os_str(),
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(to_io)?
            };
            cur = Some(fd);
        }
        Ok(cur.expect("non-empty segments yield a fd"))
    }

    // -- helpers --------------------------------------------------------------------------------

    /// Reject absolute paths and any `..` that could climb, before any syscall (uniform errors +
    /// keeps the fallback walk honest).
    fn reject_escape(rel: &Path) -> io::Result<()> {
        for c in rel.components() {
            match c {
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(escape())
                }
                Component::CurDir | Component::Normal(_) => {}
            }
        }
        Ok(())
    }

    /// The `Normal` segments of `rel` (drops `.`), erroring on any climb/absolute component.
    fn normal_segments(rel: &Path) -> io::Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        for c in rel.components() {
            match c {
                Component::Normal(seg) => out.push(PathBuf::from(seg)),
                Component::CurDir => {}
                _ => return Err(escape()),
            }
        }
        Ok(out)
    }

    /// Split `rel` into `(parent, final_name)`, or `None` when there is no name (root / empty / `.`).
    fn split_parent(rel: &Path) -> Option<(&Path, &Path)> {
        let name = rel.file_name()?;
        let parent = rel.parent().unwrap_or(Path::new(""));
        Some((parent, Path::new(name)))
    }

    fn meta_from_std(m: &std::fs::Metadata) -> Meta {
        use std::os::unix::fs::PermissionsExt as _;
        let mtime_ms = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Meta {
            is_dir: m.is_dir(),
            is_file: m.is_file(),
            is_symlink: false, // an opened fd never refers to a symlink (NO_SYMLINKS / O_NOFOLLOW)
            size: m.len(),
            mtime_ms,
            mode: m.permissions().mode() & 0o7777,
        }
    }

    // `st_mode`/`st_size`/`st_mtime*` field widths are platform-dependent (u32/i64 on the linux-raw
    // backend, u16/off_t/time_t on the libc backend), so the widening casts are necessary on some
    // targets even where they are no-ops on Linux.
    #[allow(clippy::unnecessary_cast)]
    fn meta_from_stat(st: &rustix::fs::Stat) -> Meta {
        let ft = rustix::fs::FileType::from_raw_mode(st.st_mode as rustix::fs::RawMode);
        let mtime_ms =
            (st.st_mtime as i128 * 1000 + (st.st_mtime_nsec as i128) / 1_000_000).max(0) as u64;
        Meta {
            is_dir: ft == rustix::fs::FileType::Directory,
            is_file: ft == rustix::fs::FileType::RegularFile,
            is_symlink: ft == rustix::fs::FileType::Symlink,
            size: st.st_size as u64,
            mtime_ms,
            mode: (st.st_mode as u32) & 0o7777,
        }
    }

    fn escape() -> io::Error {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "path escapes the workspace sandbox",
        )
    }

    /// Map a rustix errno to `io::Error`, surfacing openat2's containment/symlink violations
    /// (`EXDEV` from RESOLVE_BENEATH, `ELOOP` from RESOLVE_NO_SYMLINKS) as `PermissionDenied` — the
    /// same shape the lexical guard used, so callers/tests see a stable "escape" error.
    fn to_io(e: rustix::io::Errno) -> io::Error {
        match e {
            rustix::io::Errno::XDEV | rustix::io::Errno::LOOP => escape(),
            other => io::Error::from_raw_os_error(other.raw_os_error()),
        }
    }

    /// Run a blocking fs op off the async reactor (mirrors `tokio::fs`'s threadpool).
    async fn blocking<T, F>(f: F) -> io::Result<T>
    where
        F: FnOnce() -> io::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        tokio::task::spawn_blocking(f)
            .await
            .map_err(|e| io::Error::other(format!("blocking fs task: {e}")))?
    }
}

// =================================================================================================
// non-unix stub (Windows v1 stub-worker lane; ships no engine fs tools). Best-effort: lexical
// containment + a final-component symlink reject. Kept only so the crate builds on the cross target.
// =================================================================================================
#[cfg(not(unix))]
mod imp {
    use super::{ChildCwd, DirEntryLite, Meta};
    use std::io;
    use std::path::{Path, PathBuf};

    #[derive(Clone)]
    pub struct ContainedRoot {
        root_abs: PathBuf,
    }

    impl ContainedRoot {
        pub fn open(root: &Path) -> io::Result<Self> {
            std::fs::create_dir_all(root)?;
            Ok(Self {
                root_abs: root.to_path_buf(),
            })
        }

        pub fn root(&self) -> &Path {
            &self.root_abs
        }

        pub fn resolve_display(&self, rel: &Path) -> io::Result<PathBuf> {
            super::super::contain(&self.root_abs, rel)
        }

        fn contained(&self, rel: &Path) -> io::Result<PathBuf> {
            let abs = super::super::contain(&self.root_abs, rel)?;
            reject_symlink_final(&abs)?;
            Ok(abs)
        }

        pub async fn read(&self, rel: &Path) -> io::Result<Vec<u8>> {
            tokio::fs::read(self.contained(rel)?).await
        }

        pub async fn read_capped(&self, rel: &Path, cap: u64) -> io::Result<(Vec<u8>, Meta, bool)> {
            let abs = self.contained(rel)?;
            let full = tokio::fs::read(&abs).await?;
            let meta = meta_of(&tokio::fs::metadata(&abs).await?);
            let truncated = full.len() as u64 > cap;
            let bytes = if truncated {
                full[..cap as usize].to_vec()
            } else {
                full
            };
            Ok((bytes, meta, truncated))
        }

        pub async fn write(&self, rel: &Path, bytes: &[u8]) -> io::Result<Meta> {
            let abs = super::super::contain(&self.root_abs, rel)?;
            if let Some(parent) = abs.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            reject_symlink_final(&abs)?;
            tokio::fs::write(&abs, bytes).await?;
            Ok(meta_of(&tokio::fs::metadata(&abs).await?))
        }

        pub fn open_write_file(&self, rel: &Path) -> io::Result<std::fs::File> {
            let abs = super::super::contain(&self.root_abs, rel)?;
            if let Some(parent) = abs.parent() {
                std::fs::create_dir_all(parent)?;
            }
            reject_symlink_final(&abs)?;
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&abs)
        }

        pub async fn read_dir(&self, rel: &Path) -> io::Result<Vec<DirEntryLite>> {
            self.read_dir_sync(rel)
        }

        pub fn read_dir_sync(&self, rel: &Path) -> io::Result<Vec<DirEntryLite>> {
            let abs = self.contained(rel)?;
            let mut out = Vec::new();
            for item in std::fs::read_dir(&abs)? {
                let item = item?;
                let meta = match item.metadata() {
                    Ok(m) => meta_of(&m),
                    Err(_) => continue,
                };
                out.push(DirEntryLite {
                    name: item.file_name().to_string_lossy().into_owned(),
                    meta,
                });
            }
            out.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(out)
        }

        pub async fn symlink_metadata(&self, rel: &Path) -> io::Result<Meta> {
            let abs = super::super::contain(&self.root_abs, rel)?;
            Ok(meta_of(&tokio::fs::symlink_metadata(&abs).await?))
        }

        pub fn read_sync(&self, rel: &Path) -> io::Result<Vec<u8>> {
            std::fs::read(self.contained(rel)?)
        }

        pub async fn create_dir_all(&self, rel: &Path) -> io::Result<()> {
            tokio::fs::create_dir_all(super::super::contain(&self.root_abs, rel)?).await
        }

        pub fn create_dir_all_sync(&self, rel: &Path) -> io::Result<()> {
            std::fs::create_dir_all(super::super::contain(&self.root_abs, rel)?)
        }

        pub async fn remove_file(&self, rel: &Path) -> io::Result<()> {
            tokio::fs::remove_file(self.contained(rel)?).await
        }

        pub async fn remove_dir(&self, rel: &Path) -> io::Result<()> {
            tokio::fs::remove_dir(self.contained(rel)?).await
        }

        pub async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
            let from = self.contained(from)?;
            let to = super::super::contain(&self.root_abs, to)?;
            tokio::fs::rename(from, to).await
        }

        pub async fn set_mode(&self, _rel: &Path, _mode: u32) -> io::Result<()> {
            // No unix permission bits on this lane; best-effort no-op.
            Ok(())
        }

        pub async fn verify_dir(&self, rel: &Path) -> io::Result<PathBuf> {
            self.contained(rel)
        }

        pub fn verify_dir_sync(&self, rel: &Path) -> io::Result<PathBuf> {
            self.contained(rel)
        }

        pub async fn child_cwd(&self, rel: &Path) -> io::Result<ChildCwd> {
            let abs = super::super::contain(&self.root_abs, rel)?;
            std::fs::create_dir_all(&abs)?;
            reject_symlink_final(&abs)?;
            Ok(ChildCwd { path: abs })
        }
    }

    fn reject_symlink_final(abs: &Path) -> io::Result<()> {
        match std::fs::symlink_metadata(abs) {
            Ok(m) if m.file_type().is_symlink() => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "refusing to follow symlink at workspace path",
            )),
            _ => Ok(()),
        }
    }

    fn meta_of(m: &std::fs::Metadata) -> Meta {
        let mtime_ms = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Meta {
            is_dir: m.is_dir(),
            is_file: m.is_file(),
            is_symlink: m.file_type().is_symlink(),
            size: m.len(),
            mtime_ms,
            mode: 0,
        }
    }
}

pub use imp::ContainedRoot;
