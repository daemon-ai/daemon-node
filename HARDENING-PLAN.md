# HARDENING-PLAN — Cluster C interim symlink/TOCTOU guard

Track: `hardening/fs-symlink-interim` (Phase 1, Cluster C).
Status: **PLAN ONLY — no source modified, nothing committed.** Awaiting coordinator approval before Phase 2.

This is an **interim stopgap** before the Phase 3 `ContainedRoot` migration (cap-std / `openat2`,
owned by the `contained-root-type` track). Scope is intentionally minimal: re-verify the open that
FOLLOWS `contain()` so a symlinked final component (or a swapped final path) is rejected rather than
followed. It does **not** replace `contain()` and does **not** close intermediate-component symlink
swaps — Phase 3 does.

---

## 1. The gap

`contain(root, requested)` in `crates/engine/daemon-core/src/exec/mod.rs:118` is **purely lexical**:
it normalizes `.`/`..` and asserts `normalized.starts_with(root)`. It never touches the filesystem
(by design — it must work for not-yet-existing write targets, and canonicalize would follow
symlinks). Two consequences:

1. **Symlink-present-at-check-time follow.** If the final path component is a symlink that points
   outside `root`, `contain()` still returns the lexical path (which *starts_with* root), and the
   subsequent `tokio::fs::read`/`write`/`read_dir` **follows the symlink** to the outside target.
2. **Check-then-open TOCTOU.** Between `contain()` returning and the actual open, an attacker (a
   concurrent tool call in the same workspace) can swap the final component for a symlink, redirecting
   the open outside `root`.

`contain()` returns a re-resolvable `PathBuf`; every caller then does a fresh path-based open that
re-walks (and re-follows) the path. That second walk is the exploitable seam.

---

## 2. Containment call + open inventory (file:line)

### In-scope for this track (the plan lists exactly these two files)

**`crates/engine/daemon-core/src/exec/local.rs`** (`LocalEnvironment`, the v1 exec backend):

| Op | Line | contain() | Open that trusts it (follows symlinks today) | Severity |
|----|------|-----------|----------------------------------------------|----------|
| `run` cwd | 49 | `contain(&self.root, requested)` | `tokio::fs::create_dir_all(&resolved)` (48–50) then child `current_dir(&dir)` (59) — a symlinked cwd runs the child outside root | MED |
| `read` | 104 | `contain(&self.root, path)` | `tokio::fs::read(resolved)` (105) — follows a symlinked file out of root (exfiltration) | **HIGH** |
| `write` | 109 | `contain(&self.root, path)` | `tokio::fs::write(resolved, bytes)` (113) — follows a symlinked file out of root (overwrite) | **HIGH** |
| `list` | 117 | `contain(&self.root, path)` | `tokio::fs::read_dir(resolved)` (119) — follows a symlinked subdir (name enumeration outside root) | MED |

**`crates/substrate/daemon-host/src/workspace_fs.rs`** (`WorkspaceFs`, the node's `fs_*` surface;
`Self::contained` at line 245 wraps `contain`):

| Op | contain() (line) | Open that trusts it (line) | Behavior today | Severity |
|----|------------------|----------------------------|----------------|----------|
| `list` | `contained(&base,dir)` 262 | `tokio::fs::read_dir(&abs)` 263 | follows symlinked subdir | MED |
| `stat` | `contained(&base,path)` 328 | `tokio::fs::metadata(&abs)` 329 (**follows**) | leaks target size/kind outside root | LOW-MED |
| `read` | `contained(&base,path)` 354 | `tokio::fs::metadata` 355 + `tokio::fs::read(&abs)` 363 | follows symlinked file out of root | **HIGH** |
| `revision` | `contained(&base,path)` 391 | `tokio::fs::metadata(&abs)` 392 (**follows**) | leaks target revision | LOW |
| `write` | `contained(&base,path)` 423 | base-rev `metadata` 425 + `tokio::fs::write(&abs)` 442 + `metadata` 445 | follows symlinked file out of root (overwrite) | **HIGH** |
| `search` | (walks `base`) 461 | blocking walk 483–541: `DirEntry::metadata()` is `lstat` (no follow) at 498, **but** a symlink-to-file is treated as a file and `std::fs::read_to_string(&abs)` at 515 **follows** it → content exfiltration | **HIGH** |
| `watch_after` | `contained(&base,dir)` 568 | `tokio::fs::read_dir(&abs)` 571 | follows symlinked watched dir | LOW-MED |

Note: `list`/`watch` entry loops already use `DirEntry::metadata()` (an `lstat` that does **not**
follow), so `entry_kind` (mod.rs helper `workspace_fs.rs:69`) correctly reports `Symlink` for direct
children — the residual leak there is only the *directory being opened* (`abs`), addressed below.

### Related callers of `contain()` that are OUT OF SCOPE for this track (flagged for the coordinator)

These also trust `contain()` then open by path, but the plan scopes the interim to `local.rs` +
`workspace_fs.rs`. They are **not** guarded here; the Phase 3 `ContainedRoot` type (which replaces
`contain()` itself) covers all of them uniformly:

- `tools/daemon-tool-fs/src/lib.rs:313,484,563,632,810,840` (`atomic_write`, `op_write`, reads, edits).
- `tools/daemon-tool-fs/src/lint.rs:235`.
- `tools/daemon-tool-shell/src/lib.rs:134,285` (`cd` containment).
- `crates/node/daemon-node/src/fleet/job_worker.rs:91` (`std::fs::read` of attachment).
- `crates/node/daemon-node/src/cron/seed.rs:99`.

(Recommend the coordinator either fold these into Phase 3 or open a follow-up interim task; guarding
them now would exceed this track's file scope and risk conflicts with Wave-2 tracks that touch the
tool crates.)

---

## 3. Guard mechanism (with the unix / non-unix split)

Add **additive** helpers next to `contain()` in `crates/engine/daemon-core/src/exec/mod.rs` (this is
additive only — it does not modify `contain()`, so it does not conflict with the Phase 3 rewrite).
`daemon-host` already depends on `daemon-core` and imports `daemon_core::exec::contain`, so both
in-scope files share one implementation (DRY; only `daemon-core` needs the `libc` constant).

```rust
/// Interim symlink/TOCTOU guard (Cluster C stopgap; superseded by the Phase-3 cap-std/openat2
/// `ContainedRoot`). Complements the lexical `contain()`: on the open that follows containment,
/// refuse to traverse a symlink at the FINAL path component.
///
/// Covered: final-component symlink follow on file read/write is closed ATOMICALLY on unix
/// (`O_NOFOLLOW`) — no check-then-open window. NOT covered: intermediate-component symlinks
/// (a symlink at a parent dir) are still followed; Phase 3 (`openat2 RESOLVE_BENEATH |
/// RESOLVE_NO_SYMLINKS`) closes that class.

/// Open a contained file for reading, refusing a symlinked final component.
pub async fn open_read_guarded(path: &Path) -> std::io::Result<tokio::fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        tokio::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW) // ELOOP if final component is a symlink
            .open(path)
            .await
    }
    #[cfg(not(unix))]
    {
        reject_symlink_final(path).await?; // best-effort lstat pre-check (residual TOCTOU)
        tokio::fs::File::open(path).await
    }
}

/// Open a contained file for writing (create + truncate), refusing a symlinked final component.
/// With `O_NOFOLLOW | O_CREAT`: an existing symlink final component ⇒ ELOOP (rejected); an existing
/// regular file ⇒ truncated (as `tokio::fs::write`); a missing file ⇒ created as a regular file.
pub async fn open_write_guarded(path: &Path) -> std::io::Result<tokio::fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        tokio::fs::OpenOptions::new()
            .write(true).create(true).truncate(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .await
    }
    #[cfg(not(unix))]
    {
        reject_symlink_final(path).await?;
        tokio::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path).await
    }
}

/// Reject a path whose FINAL component is a symlink — for directory / metadata opens where an
/// `O_NOFOLLOW` file open does not apply. `Ok(())` when the path does not exist (create case).
/// Check-then-use: a residual TOCTOU window remains (Phase 3 closes it).
pub async fn reject_symlink_final(path: &Path) -> std::io::Result<()> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(m) if m.file_type().is_symlink() => Err(symlink_escape_error(path)),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// As `reject_symlink_final`, but never rejects the trusted `root` itself — an operator may legitimately
/// bind a workspace onto a symlinked directory (and macOS `/tmp` → `/private/tmp`); only components
/// strictly below `root` are guarded. Used by directory ops (list/watch).
pub async fn reject_symlink_final_below(root: &Path, path: &Path) -> std::io::Result<()> {
    if path == root { return Ok(()); }
    reject_symlink_final(path).await
}

fn symlink_escape_error(path: &Path) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!("refusing to follow symlink at workspace path: {}", path.display()),
    )
}
```

Rationale for the split:
- **File read/write → `O_NOFOLLOW`** (atomic; the open itself enforces, so there is *no* check-then-open
  window on the final component — the strongest thing an interim guard can do without `openat2`).
- **Directory / metadata opens (`read_dir`, `metadata`) → `lstat`-based** (`reject_symlink_final*` or
  switch to `symlink_metadata`), because those APIs take a path, not open flags. These retain a small
  check-then-use window (documented).
- **`reject_symlink_final` is exercised on unix** (list/watch/cwd), so it is not dead code under the
  workspace `unused`-deny lint.

### Per-site wiring

**`local.rs`:**
- `read` (105): `let mut f = open_read_guarded(&resolved).await?; let mut buf = Vec::new(); f.read_to_end(&mut buf).await?; Ok(buf)` (`AsyncReadExt` already imported).
- `write` (108–114): keep `create_dir_all(parent)`; then `let mut f = open_write_guarded(&resolved).await?; f.write_all(bytes).await?; Ok(())` (add `use tokio::io::AsyncWriteExt`).
- `run` cwd (47–54): `reject_symlink_final_below(&self.root, &resolved).await?;` before `create_dir_all`.
- `list` (117–119): `reject_symlink_final_below(&self.root, &resolved).await?;` before `read_dir`.

**`workspace_fs.rs`** (extend the `use daemon_core::exec::…` import; add `AsyncReadExt`/`AsyncWriteExt`):
- `list` (263): `reject_symlink_final_below(&base, &abs).await?;` before `read_dir`.
- `stat` (329): `tokio::fs::metadata` → `tokio::fs::symlink_metadata` (report the link, never follow).
- `read` (355/363): `let mut f = open_read_guarded(&abs).await?; let meta = f.metadata().await?;` (fstat on the open fd — also removes the stat↔read TOCTOU), then `read_to_end` + existing cap/truncate logic.
- `revision` (392): `tokio::fs::metadata` → `tokio::fs::symlink_metadata`.
- `write` (425/442/445): base-revision check uses `symlink_metadata`; keep `create_dir_all(parent)`; `let mut f = open_write_guarded(&abs).await?; f.write_all(bytes).await?; let meta = f.metadata().await?;` for the returned revision.
- `search` (498–515): after `let meta = item.metadata()`, add `if meta.file_type().is_symlink() { continue; }` — skips symlink entries entirely (no recursion into symlinked dirs, no `read_to_string` of symlinked files). The trusted seed `base` is still walked.
- `watch_after` (571): `reject_symlink_final_below(&base, &abs).await?;` before `read_dir`.

### Dependency change
Add `libc = "0.2"` to `[workspace.dependencies]` (root `Cargo.toml`) and, in
`crates/engine/daemon-core/Cargo.toml` only:
```toml
[target.'cfg(unix)'.dependencies]
libc = { workspace = true }
```
`libc 0.2.186` and `rustix 1.1.4` are already in `Cargo.lock` (transitive). `daemon-host` needs no new
dep — it calls the `daemon-core` helpers, never `libc` directly. No wire types change, so no CDDL /
`arbitrary`-conformance impact.

---

## 4. Residual TOCTOU coverage (honest statement)

**Closed by this interim guard:**
- Final-component symlink **follow** on file `read`/`write` — closed atomically on unix via `O_NOFOLLOW`
  (the check-then-open window on the final component is eliminated for reads/writes; the open itself
  enforces).
- `workspace_fs` `read` stat↔read race — removed by taking metadata from the opened fd (`file.metadata()`).
- `stat`/`revision` no longer follow a symlinked final component (switched to `lstat`).
- `search` no longer reads through symlinks (symlink entries skipped in the walk).
- Symlinked final-component directories in `list`/`watch`/`run`-cwd are rejected (below-root).

**NOT closed (explicitly out of interim scope — Phase 3 `ContainedRoot` closes these):**
- **Intermediate-component symlink swaps.** A symlink at a *parent* directory in the path is still
  followed. A concurrent swap of an intermediate directory to a symlink between `contain()` and the
  open remains exploitable. Only the *final* component is guarded.
- **Residual check-then-use window on directory/metadata ops and on all non-unix opens** (the
  `lstat`-based paths are not atomic). File read/write on unix are atomic; everything else keeps a
  small window.
- **The out-of-scope `contain()` callers** in §2 (tool-fs, tool-shell, job_worker, cron/seed) are
  unguarded by this track.

The class is only fully eliminated when a validated location is representable *solely* as an open
directory fd operated via `*at`/`openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` (Phase 3), which
removes the re-resolvable `PathBuf` seam entirely.

---

## 5. Tests to add (Phase 2 — bug-reproducing test FIRST)

Workflow per item: add the test, run it on the current tree to confirm it **fails** (symlink followed
= escape), then implement the guard, confirm it **passes**. All symlink tests `#[cfg(unix)]`
(`std::os::unix::fs::symlink`); the guards still compile on non-unix via the fallback branch.

**`daemon-core` (`exec/local.rs` tests):**
1. `read_rejects_symlinked_final_component` — create a workspace root and an *outside* secret file;
   `symlink(outside_secret, root/link)`; assert `env.read("link")` errors (`PermissionDenied`/ELOOP)
   and does **not** return the secret bytes. Also assert a real in-root file still reads fine.
2. `write_rejects_symlinked_final_component` — `symlink(outside_target, root/link)`; assert
   `env.write("link", …)` errors and the outside target is **unmodified**.
3. `run_cwd_rejects_symlinked_dir` — `symlink(outside_dir, root/cwdlink)`; assert `run(cmd.cwd("cwdlink"))` errors.

**`daemon-host` (`workspace_fs.rs` tests):**
4. `read_rejects_symlink_escape` — secret outside `base`; `symlink(secret, base/link)`; assert
   `fs.read(Workspace, "link", 0)` is `Err` and never yields the secret.
5. `write_rejects_symlink_escape` — assert `fs.write` through a symlink errors; outside target unchanged.
6. `search_skips_symlinked_target` — outside file contains the needle; `symlink` it into `base`; assert
   `fs.search` returns **no** hit from the symlink target.
7. `list_rejects_symlinked_subdir` — `symlink(outside_dir, base/sublink)`; assert `fs.list(Workspace, "sublink", …)` errors.
8. `symlinked_bound_root_still_lists` (honesty/no-regression) — bind a session to a directory that *is*
   itself a symlink; assert `fs.list(Session, "", …)` still succeeds (proves `reject_symlink_final_below`
   does not break legitimately-bound symlinked roots).

Existing tests (`contain_accepts_relative_and_rejects_escapes`, round-trip, pagination, watch) must
stay green — the guards only reject symlinks, not ordinary contained paths.

---

## 6. Risks / ambiguities

- **`tokio::fs::OpenOptions::custom_flags` availability.** Expected present under `#[cfg(unix)]` in
  tokio 1.x (mirrors std). If the pinned tokio lacks the inherent method, fallback: build a
  `std::fs::OpenOptions` with `custom_flags`, open in `tokio::task::spawn_blocking`, wrap via
  `tokio::fs::File::from_std`. Verify on the first Phase-2 build.
- **Behavior change: benign in-workspace final-component symlinks are now rejected on read/write.**
  The interim guard rejects *all* final-component symlinks, including ones whose target is still inside
  the root. This is the conservative safe default for a stopgap (workspaces rarely contain symlinks);
  Phase 3 `openat2(RESOLVE_BENEATH)` can re-permit in-root symlinks with proper containment. Documented
  in the helper doc comment.
- **`stat`/`revision` semantics change** to `lstat` (report the link, not the target). Safe and more
  correct for containment; note in commit message.
- **`path == root` lexical equality** in `reject_symlink_final_below`: relies on `contain()` producing
  `abs` from `base` by lexical join+normalize (it does), so equality holds for `dir == "" | "."`.
- **Cross-track conflicts:** none in Wave 1 (no other track touches `exec/*` or `workspace_fs.rs`).
  Phase 3 `contained-root-type` rewrites `contain()` and will absorb/remove these helpers — expected.
- **Scope honesty:** the out-of-scope `contain()` callers (§2) remain exploitable after this track;
  flagged for the coordinator to route into Phase 3 or a follow-up.

---

## 7. Gate (Phase 2, from worktree root)
- `nix develop --command cargo fmt`
- `nix develop --command cargo clippy --workspace --all-targets -- -D warnings`
- `nix develop --command cargo test --workspace`
- (`cargo deny check` — libc is MIT/Apache, already in-tree; no wire change ⇒ CDDL/`arbitrary` gate not required, but the full gate is run anyway.)
Commit on `hardening/fs-symlink-interim`. Do NOT merge; do NOT remove the worktree.
