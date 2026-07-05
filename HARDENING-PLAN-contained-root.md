# HARDENING-PLAN — Cluster C `ContainedRoot` (Phase 3, durable TOCTOU/symlink elimination)

Track: `hardening/contained-root-type` (Phase 3, Cluster C).
Worktree: `/home/j/experiments/daemon-worktrees/contained-root`, branch off `hardening/integration`
(which already carries all Phase 1+2 hardening, incl. the Phase 1 `fs-symlink-interim` guard this
track **supersedes and removes**).

Status: **PLAN ONLY — no source modified, nothing committed.** Awaiting coordinator approval before
implementation.

This track replaces the lexical `contain()` + interim `O_NOFOLLOW`/`lstat` guards with a
`ContainedRoot` capability that resolves every attacker-influenced *relative* path via
`openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` from an **open root directory fd** — eliminating the
check-then-open TOCTOU class (not just the final-component follow the interim closed) and the
intermediate-component symlink escape the interim explicitly left open.

---

## 0. TL;DR of the decision points (for fast review)

| Decision | Choice | One-line why |
|----------|--------|--------------|
| Backend crate | **`rustix::fs::openat2` directly** (not `cap-std`) | `rustix 1.1.4` is already in `Cargo.lock` (transitive) ⇒ **zero new crates** for `cargo deny`; gives exact `RESOLVE_BENEATH \| RESOLVE_NO_SYMLINKS` control; safe API preserves `#![forbid(unsafe_code)]`. |
| Symlink policy | `RESOLVE_NO_SYMLINKS` (reject **all** symlinks below root) | Matches the plan's literal flags and preserves the interim's conservative "no symlinks in workspace" posture ⇒ no behavior regression, existing tests stay green. |
| Root anchor | opened once per op on the **trusted** root; only the **relative** path is hardened | The root (node sandbox / operator-`Bound` dir) is the trust boundary; the exploitable seam was always the *relative* re-resolution, which is now `openat2`-only. |
| Non-Linux | macOS: fd-based component walk (`openat` + `O_NOFOLLOW`); Windows: lexical `contain()` + `lstat` stub | openat2 is Linux-only; keep building on all three flake targets (Windows v1 = stub-worker, no engine fs tools). |
| `Bound` | canonicalize at the single bind choke point (`apply_workspace_exec`) | Stable real root fd; do **not** force `Bound` under node root (would break the documented "work on my repo" case). |
| Wire types | **unchanged** | `WorkspaceBinding` shape is untouched (canonicalization is node-side on the value) ⇒ no CDDL/`arbitrary` change required. |

---

## 1. The gap this closes (vs. the interim)

`contain(root, requested)` ([`crates/engine/daemon-core/src/exec/mod.rs:131`](crates/engine/daemon-core/src/exec/mod.rs)) is **purely lexical**: it normalizes `.`/`..` and asserts `normalized.starts_with(root)`, then returns a **re-resolvable `PathBuf`**. Every caller does a fresh, path-based open that re-walks (and re-follows) that path. The interim `fs-symlink-interim` guard bolted `O_NOFOLLOW`/`lstat` onto the *final component* of that second walk.

Two residuals the interim explicitly did **not** close (see its own `HARDENING-PLAN-fs-symlink-interim.md` §4):

1. **Intermediate-component symlink escape.** A symlink at a *parent* directory (`root/sub -> /outside`, then open `sub/file`) is still followed — the interim only guarded the final component (`file`, a normal file).
2. **Check-then-open TOCTOU on directory/metadata ops and all non-unix paths** — the `lstat`-then-use paths are not atomic.

`ContainedRoot` eliminates both structurally: the relative path is **only ever** resolved by `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` from an open root fd, which the kernel evaluates atomically and which fails (`ELOOP`/`EXDEV`/`ENOTDIR`) the instant *any* component (intermediate or final) is a symlink or would climb above the root. There is no second path walk to race.

---

## 2. `ContainedRoot` type design

### 2.1 Location & shape

New module `crates/engine/daemon-core/src/exec/contained.rs`, re-exported as
`daemon_core::exec::ContainedRoot` and `daemon_core::ContainedRoot` (mirroring the existing `contain`
re-export used by tool-shell). All current `contain()` consumers already depend on `daemon-core`.

```rust
/// A validated filesystem containment boundary: an OPEN root directory fd whose methods resolve
/// every (attacker-influenced) relative path via openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)
/// from that fd. A location under the root is only ever representable as an open capability derived
/// here — never re-resolved from a PathBuf — so symlink-escape and check-then-open TOCTOU are
/// eliminated as a class (not just the final component the interim guard covered).
#[derive(Clone)]
pub struct ContainedRoot {
    root: Arc<OwnedFd>,   // the root directory fd (the trust anchor, opened once at construction)
    root_abs: PathBuf,    // for policy checks / display / spawn cwd ONLY (never for opening)
}
```

- `root: Arc<OwnedFd>` — the sole capability. `Arc` so `.try_clone()`'d borrows can move into
  `tokio::task::spawn_blocking` closures (`OwnedFd: Send`), keeping byte-I/O off the async reactor.
- `root_abs` — retained for the three unavoidable path-interop needs, each documented as
  **policy/spawn only, never an open**: (a) `write_denied` prefix checks in tool-fs, (b) child-process
  `current_dir`, (c) the `ignore` walker entry point for grep/glob. It is *derived from the trusted
  root*, not from the relative request, so it carries no attacker path.

### 2.2 Constructor

```rust
impl ContainedRoot {
    /// Open `root` as the containment boundary, creating it if missing (matches LocalEnvironment's
    /// lazy ensure_root). The root itself is opened by path (following symlinks ONCE — an operator
    /// may legitimately Bind onto a symlinked dir, e.g. macOS /tmp -> /private/tmp); only RELATIVE
    /// lookups from the returned fd are symlink-hardened.
    pub fn open(root: &Path) -> io::Result<Self>;
}
```

`open` = `create_dir_all(root)` + `rustix::fs::open(root, O_DIRECTORY | O_CLOEXEC, empty_mode)`.
Cheap (one syscall); called per-op by the async wrappers so `LocalEnvironment::new` stays infallible
(no public API churn).

### 2.3 Methods (the migration surface)

All async methods do the fd resolution + byte I/O inside `spawn_blocking`.

| Method | openat2 flags | Backs |
|--------|---------------|-------|
| `read(rel) -> Vec<u8>` | `O_RDONLY` | local.rs `read`, workspace_fs `read`, tool-fs op_read (via exec), job_worker attachment read |
| `read_capped(rel, cap) -> (Vec<u8>, FileMeta, bool)` | `O_RDONLY` | workspace_fs `read` (fstat from the opened fd + cap/truncate; kills the stat↔read race) |
| `write(rel, bytes) -> FileMeta` | `O_WRONLY\|O_CREAT\|O_TRUNC` | local.rs `write`, workspace_fs `write` |
| `open_write_fd(rel) -> OwnedFd` | `O_WRONLY\|O_CREAT\|O_TRUNC` | tool-fs `atomic_write` temp-file open |
| `read_dir(rel) -> Vec<DirEntryLite>` | `O_RDONLY\|O_DIRECTORY` | local.rs `list`, workspace_fs `list`/`watch_after` (name+lstat kind/size per child via `fstatat(AT_SYMLINK_NOFOLLOW)`) |
| `symlink_metadata(rel) -> FileMeta` | `fstatat(.., AT_SYMLINK_NOFOLLOW)` after `RESOLVE_BENEATH\|NO_SYMLINKS` parent open | workspace_fs `stat`/`revision`, tool-fs op_delete kind, atomic_write perm-preserve |
| `create_dir_all(rel)` | `mkdirat` walk (each segment `RESOLVE_BENEATH\|NO_SYMLINKS`) | local.rs `run` cwd + `write` parent, workspace_fs `write` parent, tool-fs `atomic_write` parent |
| `remove_file(rel)` / `remove_dir(rel)` | `unlinkat` (+`AT_REMOVEDIR`) | tool-fs op_delete |
| `rename(from_rel, to_rel)` | `renameat` (both beneath root) | tool-fs `atomic_write` swap |
| `verify_dir(rel) -> PathBuf` | `O_DIRECTORY`, resolve then return spawn/walk-safe path | grep/glob root, tool-shell cwd, cron/seed script, child cwd |
| `resolve_display(rel) -> io::Result<PathBuf>` | lexical join of `root_abs` + normalized rel; **no fs** | policy-only path for `write_denied` (documented: never used to open) |

`FileMeta` / `DirEntryLite` are tiny local structs (mtime_ms, size, kind) so callers do not need the
`OwnedFd` and the blocking closure stays `'static`.

### 2.4 Resolution core (the one place the flags live)

```rust
#[cfg(target_os = "linux")]
fn resolve_beneath(root: BorrowedFd, rel: &Path, oflags: OFlags, mode: Mode) -> io::Result<OwnedFd> {
    use rustix::fs::{openat2, OpenHow, ResolveFlags};
    let how = OpenHow::new(oflags, mode)
        .resolve(ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS);
    match openat2(root, rel, how) {
        Ok(fd) => Ok(fd),
        // openat2 landed in Linux 5.6; a seccomp'd/older kernel returns ENOSYS/EPERM -> component walk.
        Err(e) if matches!(e, Errno::NOSYS | Errno::PERM | Errno::OPNOTSUPP) => walk_openat(root, rel, oflags, mode),
        Err(e) => Err(e.into()),
    }
}

#[cfg(all(unix, not(target_os = "linux")))]     // macOS et al.: no openat2
fn resolve_beneath(root: BorrowedFd, rel: &Path, oflags: OFlags, mode: Mode) -> io::Result<OwnedFd> {
    walk_openat(root, rel, oflags, mode)
}
```

`walk_openat` (unix, non-Linux fallback + Linux old-kernel fallback): lexically reject any
`Component::ParentDir`/absolute/prefix (climb-out), then for each `Normal` segment `openat(parent,
seg, O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)` re-parenting the fd; the final segment opens with the
requested `oflags | O_NOFOLLOW`. Because every step opens **from the parent fd** with `O_NOFOLLOW`,
a symlink at *any* component (intermediate or final) fails — this closes the intermediate-symlink
class on macOS too, not just Linux. It is still fd-relative (never re-resolves a full path string),
so no PathBuf seam. Residual: a directory-rename race *between* opening component i and i+1 (far
weaker than today; openat2 has none). Documented.

```rust
#[cfg(not(unix))]                                // Windows v1 stub-worker lane (no engine fs tools)
fn resolve_beneath(root_abs: &Path, rel: &Path, ...) -> io::Result<...> {
    // Best-effort: lexical contain() + symlink_metadata reject on the final component.
    // Documented as best-effort; the Windows v1 lane ships no engine workers that exercise this.
}
```

On non-unix, `ContainedRoot` stores `root_abs` only (no `OwnedFd`) and delegates to the current
lexical `contain()` + an `lstat` reject — i.e. it keeps building and preserves the interim's
non-unix behavior, no more, no less.

### 2.5 `#![forbid(unsafe_code)]` stays intact — confirmed

`daemon-core/src/lib.rs:22` has `#![forbid(unsafe_code)]`. `rustix`'s public `fs` API (`openat2`,
`openat`, `fstatat`, `mkdirat`, `unlinkat`, `renameat`, `OwnedFd`, `BorrowedFd`) is **safe** — rustix
encapsulates the raw syscalls behind safe wrappers, so our module writes **zero `unsafe`**. Converting
`OwnedFd -> std::fs::File` (`File::from(owned_fd)`) and `-> tokio::fs::File::from_std` are both safe.
No `unsafe` block is introduced anywhere in this track; the forbid attribute is preserved verbatim.

---

## 3. Migration inventory (every open, file:line)

Line numbers are current-tree (post-interim). Each row: replace the lexical `contain()` + the
interim-guarded open with the corresponding `ContainedRoot` method.

### 3.1 `crates/engine/daemon-core/src/exec/local.rs` (`LocalEnvironment` — v1 exec backend)

| Op | Lines | Today (interim) | Becomes |
|----|-------|-----------------|---------|
| import | 10–13 | `use super::{contain, open_read_guarded, open_write_guarded, reject_symlink_final_below, ...}` | `use super::{ContainedRoot, ...}` (drop the interim helpers) |
| `run` cwd | 69–80 | `contain` + `reject_symlink_final_below` + `create_dir_all(&resolved)` + child `current_dir(&dir)` | `let cr = ContainedRoot::open(&self.root)?; let dir = cr.child_cwd(requested)?;` (see §3.6 child-cwd) |
| `read` | 129–137 | `contain` + `open_read_guarded` + `read_to_end` | `ContainedRoot::open(&self.root)?.read(path).await` |
| `write` | 139–148 | `contain` + `create_dir_all(parent)` + `open_write_guarded` + `write_all` | `ContainedRoot::open(&self.root)?.write(path, bytes).await` (creates parent internally) |
| `list` | 150–162 | `contain` + `reject_symlink_final_below` + `read_dir` | `ContainedRoot::open(&self.root)?.read_dir(path).await` (returns sorted names) |

### 3.2 `crates/substrate/daemon-host/src/workspace_fs.rs` (`WorkspaceFs` — node `fs_*` surface)

| Op | Lines | Today | Becomes |
|----|-------|-------|---------|
| import | 28–30 | `use daemon_core::exec::{contain, open_read_guarded, open_write_guarded, reject_symlink_final_below}` | `use daemon_core::exec::ContainedRoot` |
| `Self::contained` helper | 247–250 | wraps `contain` | remove (or repoint to `ContainedRoot::resolve_display` for the rare display need) |
| `list` | 264–297 | `contained` + `reject_symlink_final_below` + `read_dir` + per-entry `metadata` | `ContainedRoot::open(&base)?.read_dir(dir)` → `DirEntryLite` list; keep sort/paginate |
| `stat` | 334–354 | `contained` + `symlink_metadata` | `ContainedRoot::open(&base)?.symlink_metadata(path)` |
| `read` | 357–400 | `contained` + `open_read_guarded` + `file.metadata` + `read_to_end` + cap | `ContainedRoot::open(&base)?.read_capped(path, cap)` (fstat from fd) |
| `revision` | 403–419 | `contained` + `symlink_metadata` | `ContainedRoot::open(&base)?.symlink_metadata(path)` (map to `FsRevision`, `None` on NotFound) |
| `write` | 430–480 | `contained` + base-rev `symlink_metadata` + `create_dir_all(parent)` + `open_write_guarded` + `write_all` + `file.metadata` | base-rev via `symlink_metadata`; then `ContainedRoot::open(&base)?.write(path, bytes)` returning `FileMeta` |
| `search` | 484–587 | blocking walk from `base` via `std::fs::read_dir`, `is_symlink()` skip, `read_to_string` | fd-recursive walk from the root fd (§3.5) |
| `watch_after` | 594–684 | `contained` + `reject_symlink_final_below` + `read_dir` | `ContainedRoot::open(&base)?.read_dir(dir)` for the snapshot; ring logic unchanged |

### 3.3 `tools/daemon-tool-fs/src/lib.rs`

| Op | Lines | Today | Becomes |
|----|-------|-------|---------|
| import | 37 | `use daemon_core::exec::contain;` | `use daemon_core::exec::ContainedRoot;` |
| `atomic_write` | 312–341 | `contain` + `create_dir_all` + `tokio::fs::write(tmp)` + `set_permissions` + `rename` | `ContainedRoot::open(workspace)?`: `create_dir_all(parent)`, `open_write_fd(tmp_rel)`+write, perm-preserve via `symlink_metadata`, `rename(tmp_rel, rel)` (both beneath root) |
| `op_write` | 483–489 | `contain` for `write_denied` | `resolve_display(&path)?` for the deny check (policy only); the write goes through `atomic_write` |
| `op_edit` | 562–568 | `contain` for `write_denied` | `resolve_display(&path)?` (policy only) |
| `op_delete` | 631–658 | `contain` + `tokio::fs::metadata` + `remove_dir`/`remove_file` | `resolve_display` for deny check; `ContainedRoot` `symlink_metadata` + `remove_dir`/`remove_file` |
| `op_grep` root | 810–819 | `contain(&workspace, path)` | `ContainedRoot::open(&workspace)?.verify_dir(&path)?` → verified abs root for the walker |
| `op_glob` root | 840–849 | `contain(&workspace, path)` | `ContainedRoot::open(&workspace)?.verify_dir(&path)?` |

Note: `op_read`/`op_edit` read the file via `cx.exec.read(...)` (already routed through
`LocalEnvironment` = `ContainedRoot`), so those reads are covered by §3.1. Only their *write-side*
`contain()` and the standalone `atomic_write`/`op_delete`/grep/glob paths need direct migration.

### 3.4 `tools/daemon-tool-fs/src/lint.rs`

| Op | Line | Today | Becomes |
|----|------|-------|---------|
| `baseline_output` temp | 235–240 | `daemon_core::exec::contain(workspace, &tmp_rel)` + `tokio::fs::write` + `remove_file` | `ContainedRoot::open(workspace)?`: `write(tmp_rel, pre_content)` + `remove_file(tmp_rel)` |

### 3.5 `tools/daemon-tool-fs/src/search.rs`

- `walker` (112–119): add `.follow_links(false)` explicitly (the `ignore` default, made a *declared*
  choice) so the grep/glob walk never traverses a symlink after the `verify_dir` entry check. This is
  the "verified root + non-following walker" containment for grep/glob; a full fd-recursion of ripgrep
  is disproportionate and is documented as the one residual (read-only enumeration, no link follow).

### 3.6 Child-process cwd (`local.rs::run`, tool-shell, cron/seed)

Spawning needs a cwd/exec *path*, not an fd, and `#![forbid(unsafe_code)]` rules out
`pre_exec`+`fchdir`. Approach:

- **Linux:** `verify_dir(rel)` opens the dir fd via openat2 (proving containment atomically) and
  returns `"/proc/self/fd/<raw>"` — the kernel resolves the fd's target at spawn/exec with no path
  re-resolution. `ContainedRoot::child_cwd(rel)` returns this (keeping the fd alive for the spawn's
  lifetime).
- **macOS / non-Linux:** return the verified absolute path (residual dir→symlink swap between verify
  and spawn; the **exec-os-sandbox** track owns spawn hardening — clean seam, see §7).

### 3.7 `crates/node/daemon-node/src/fleet/job_worker.rs`

| Op | Lines | Today | Becomes |
|----|-------|-------|---------|
| attachment read | 85–107 | `contain(&parent_root, path)` + `std::fs::read(src)` + `std::fs::write(inbox.join(name))` | `ContainedRoot::open(&parent_root)?.read(path)` for the (attacker-influenced) source; `ContainedRoot::open(&child_root)?.write(Path::new("inbox").join(name), &out)` for the sink |

### 3.8 `crates/node/daemon-node/src/cron/seed.rs`

| Op | Lines | Today | Becomes |
|----|-------|-------|---------|
| `run_script` | 95–103 | `contain(dir, rel)` + `Command::new(&path)` | `ContainedRoot::open(dir)?.child_cwd`-style verify → spawn-safe exec path (`/proc/self/fd` on Linux, verified abs on macOS); flag the spawn seam (§7) |

---

## 4. Interim helpers to DELETE (superseded)

All in `crates/engine/daemon-core/src/exec/mod.rs` (doc-comment-marked interim), plus their imports:

| Item | Lines | Action |
|------|-------|--------|
| Interim module doc block ("Interim symlink / TOCTOU guard") | 170–185 | delete |
| `symlink_escape_error` | 187–196 | delete (folded into `ContainedRoot` errors) |
| `open_read_guarded` | 198–218 | delete |
| `open_write_guarded` | 220–248 | delete |
| `reject_symlink_final` | 250–261 | delete |
| `reject_symlink_final_below` | 263–271 | delete |
| interim imports in `local.rs` | 10–13 | drop `open_read_guarded`, `open_write_guarded`, `reject_symlink_final_below` |
| interim imports in `workspace_fs.rs` | 28–30 | drop the three interim helpers |
| interim `libc` dep in `daemon-core/Cargo.toml` | 29–33 (`[target.'cfg(unix)'.dependencies] libc`) | replace with `rustix` (see §6) |
| interim symlink tests (final-component) in `local.rs`/`workspace_fs.rs` | see §5 | **keep** (retarget assertions to `ContainedRoot`; they must stay green — NO_SYMLINKS still rejects final-component symlinks) |

**Kept, NOT deleted:** the lexical `contain()` function itself (131–158) stays — it is still used for
the non-unix fallback inside `ContainedRoot`, and removing the public symbol would be a wider breaking
change than this track warrants (Phase 4 `clippy-disallow` will fence direct `std::fs`/`tokio::fs`
around `ContainedRoot`; a follow-up can privatize `contain` once every caller is migrated). Its
`CommandFingerprint`/`resolve_program_abs` neighbors (Cluster B) are untouched.

---

## 5. Bind-time canonicalization of `WorkspaceBinding::Bound`

Single choke point: `crates/node/daemon-node/src/profiles/resolve.rs::apply_workspace_exec` (200–223),
line 213–214:

```rust
// before
let (root, trusted) = match &binding {
    Some(WorkspaceBinding::Bound(p)) => (p.clone(), false),
    _ => (roots.isolated_root(id.as_str()), true),
};
// after
let (root, trusted) = match &binding {
    Some(WorkspaceBinding::Bound(p)) => (canonicalize_bound(p), false),
    _ => (roots.isolated_root(id.as_str()), true),
};
```

`canonicalize_bound(p)`: `create_dir_all(p)` then `std::fs::canonicalize(p)` (resolve the root's own
symlinks/`.`/`..` to a stable real absolute path); on failure fall back to `p.clone()`. This makes the
`ContainedRoot` root fd open on a stable target and makes `RESOLVE_BENEATH` well-defined regardless of
symlinks in the *prefix* of the bound path.

- **Centralization:** `cron/worker.rs::overlay_from_spec` (118–132) and `cron/seed.rs` (130) both
  build `Bound` from a `workdir` string, but both flow through the overlay into `apply_workspace_exec`,
  so canonicalizing there covers cron too — no per-site change needed.
- **NOT added:** a hard "`Bound` must live under the node root" rejection. `Bound` is *by design* the
  external "work on my repo" directory (`daemon-common` doc, `WorkspaceBinding::Bound`), and Phase 2
  `policy-partition` already gates overlay `workspace` mutations behind an operator-tier capability, so
  the "explicitly allowed" condition in the plan is satisfied by that gate. Forcing it under the node
  root would break the feature. Flagged as a surfaced-policy option deferred (Phase 4).
- **Wire type unchanged:** only the node-side *value* is canonicalized; `WorkspaceBinding`'s shape is
  untouched ⇒ no CDDL/`arbitrary` impact.

---

## 6. Dependency changes (with `cargo deny` note)

- **Add** to root `Cargo.toml` `[workspace.dependencies]`:
  `rustix = { version = "1", default-features = false, features = ["fs", "std"] }`
  (`rustix 1.1.4` is **already in `Cargo.lock`** as a transitive dep — this promotes it to a direct
  dep, adding **no new crate** to the graph.)
- **In `crates/engine/daemon-core/Cargo.toml`**, replace the interim unix block:
  ```toml
  [target.'cfg(unix)'.dependencies]
  rustix = { workspace = true }
  ```
  (was `libc = { workspace = true }`.)
- **Remove** the interim `libc` direct dep from daemon-core. Leave the `libc = "0.2"` entry in
  `[workspace.dependencies]` (still pulled transitively by `nix`, `rustix`, etc.); verify no other
  crate names `libc` as a *direct* dep — if none does, remove the workspace entry too (else keep).
- **`cargo deny`:** `rustix` is MIT/Apache-2.0/BSD-style (permissive, on the allow-list), from
  crates.io (`unknown-registry`/`unknown-git` unaffected), and already vetted (transitive). No new
  advisory/license/source surface; `bans.multiple-versions = "warn"` is unaffected (single rustix
  version already resolved). The gate includes `cargo deny check` per the track contract.

**Why not `cap-std`:** it would add ~6 new crates (`cap-std`, `cap-primitives`, `io-lifetimes`,
`io-extras`, `fs-set-times`, `arf-strings`…) to the deny surface, and its resolution policy uses
`RESOLVE_BENEATH` (permits in-sandbox symlinks) rather than the plan's `RESOLVE_NO_SYMLINKS`, so it
would *change* behavior vs. the interim (in-root symlinks would newly be allowed) and not expose the
exact flags. `rustix` gives precise flag control, zero new crates, and the same safe-API/`forbid`
guarantee. cap-std's built-in Windows fallback is its one advantage, but the Windows v1 lane ships no
engine fs tools, so our lexical non-unix stub is sufficient.

---

## 7. Cross-track coordination (exec-os-sandbox — flag for deconfliction)

The sibling Phase 3 **exec-os-sandbox** track owns the *process-sandbox* path; I own the
*fs-containment* path. Overlap surfaces and the clean seam:

- **`crates/engine/daemon-core/src/exec/mod.rs`** — I edit: delete interim helpers (§4), add
  `pub mod contained;` + `pub use contained::ContainedRoot;`. I **do not** touch `Command`,
  `ExecResult`, `ExecCx`, the `ExecutionEnvironment` trait, `CommandFingerprint`, or
  `resolve_program_abs`. If exec-os-sandbox adds a `SandboxPolicy` field to `Command`/`ExecCx`, it is
  disjoint from my helper deletions — merge order flexible; expect a trivial import-block conflict at
  most.
- **`crates/engine/daemon-core/src/exec/local.rs::run` (65–127)** — SHARED. I change **only** the cwd
  resolution (69–80) to `ContainedRoot::child_cwd`. The spawn mechanics (`tokio::process::Command`
  build, `env_clear`, stdio, `child.spawn`, cancel/kill — 82–107) are **theirs** (env policy from
  Phase 1, OS sandbox from Phase 3). Seam: I hand them a contained, verified cwd path (`/proc/self/fd`
  on Linux); they wrap the spawn. **Recommend merging exec-os-sandbox's `run` changes and mine with
  explicit review of this function.**
- **`tools/daemon-tool-execute-code/`** — I do **not** modify it. Its `sandbox.rs`/`lib.rs`
  (`lib.rs:253` `cx.exec.cwd()`, `lib.rs:290` `create_dir_all(&staging)`, `lib.rs:321`
  `write(&script, code)`) are raw `tokio::fs` opens on a path derived from `cx.exec.cwd()` — a
  containment-adjacent surface, but it is **execute_code**, owned by exec-os-sandbox. **Flagged**:
  those staging opens are *not* migrated to `ContainedRoot` by this track; exec-os-sandbox should
  either route them through `ContainedRoot` or contain them under its sandbox. Left to them to avoid a
  cross-track edit.
- **`tools/daemon-tool-shell/src/lib.rs` background/pty cwd (`resolve_cwd` 129–138, `run_cd`
  455–460)** — I migrate the containment (`contain` → `ContainedRoot::verify_dir`/`child_cwd`); the
  actual background/pty **spawn** goes through `ProcessRegistry` (their spawn domain). I produce the
  contained cwd; they spawn it.
- **`crates/node/daemon-node/src/cron/seed.rs::run_script`** — I contain the script path; the
  `Command::new(path).output()` spawn is a process-exec surface (flag for exec-os-sandbox if they want
  it under the sandbox; functionally my change only tightens *which* path is executed).

No other track in Wave 3 touches `exec/*`, `workspace_fs.rs`, tool-fs, tool-shell, job_worker, or
cron (per the plan's wave disjointness notes).

---

## 8. Tests — bug-reproducing FIRST (the case the interim did NOT close)

Workflow per test: add it, run on the current tree to confirm it **fails** (intermediate symlink is
followed = escape), implement `ContainedRoot`, confirm it **passes**. All symlink tests `#[cfg(unix)]`.

### 8.1 NEW — intermediate-component symlink escape (the interim gap)

1. `daemon-core local.rs::read_rejects_intermediate_symlink` — `root/sub -> /outside` (dir symlink);
   `/outside/secret.txt` = "TOP SECRET"; `env.read("sub/secret.txt")`. Interim: final component
   (`secret.txt`) is a normal file ⇒ followed ⇒ escape. `ContainedRoot`: `sub` is a symlink ⇒
   `openat2` fails ⇒ `Err`, secret bytes never returned.
2. `daemon-core local.rs::write_rejects_intermediate_symlink` — `root/sub -> /outside`;
   `env.write("sub/target.txt", ...)` ⇒ `Err`; `/outside/target.txt` unmodified/absent.
3. `daemon-host workspace_fs.rs::read_rejects_intermediate_symlink_escape` — same via `fs.read`.
4. `daemon-host workspace_fs.rs::list_rejects_intermediate_symlink` — `base/sub -> /outside`;
   `fs.list(Workspace, "sub/inner", ...)` (or list through the symlinked parent) ⇒ `Err`.
5. `daemon-host workspace_fs.rs::search_skips_intermediate_symlink` — needle in a file reachable only
   through an intermediate symlinked dir ⇒ no hit (fd-recursive walk never enters the symlink).
6. `daemon-core local.rs::run_cwd_rejects_intermediate_symlink` — `root/sub -> /outside`;
   `run(cmd.cwd("sub/inner"))` ⇒ `Err`.

### 8.2 KEPT — interim cases stay closed (retargeted to `ContainedRoot`)

The existing interim tests must remain green (RESOLVE_NO_SYMLINKS also rejects a symlinked *final*
component): `read_rejects_symlinked_final_component`, `write_rejects_symlinked_final_component`,
`list_rejects_symlinked_dir`, `run_cwd_rejects_symlinked_dir` (local.rs 227–329);
`read_rejects_symlink_escape`, `write_rejects_symlink_escape`, `search_skips_symlinked_target`,
`list_rejects_symlinked_subdir` (workspace_fs.rs 917–1016). Assertions stay ("no escape"), only the
under-the-hood mechanism changes.

### 8.3 KEPT — no-regression / honesty

- `symlinked_bound_root_still_lists` (workspace_fs.rs 1020–1042) — a session **Bound** onto a dir that
  is itself a symlink still lists (the root is opened following symlinks once; only relative lookups are
  hardened). Must pass.
- `local_env_write_read_list_roundtrip`, `contain_accepts_relative_and_rejects_escapes`,
  `local_env_rejects_out_of_workspace_write`, pagination/cursor/watch tests — must pass unchanged.
- NEW `bound_root_canonicalized_at_bind` (resolve.rs tests) — `Bound(symlink_to_real)` records/opens
  the canonical real path (assert `roots.session_root` / exec root resolves to the canonicalized dir).

### 8.4 Migrated-caller coverage

- tool-fs: NEW `atomic_write_rejects_intermediate_symlink` + `op_delete_rejects_symlink`;
  existing op_write/op_edit/op_grep/op_glob tests stay green.
- job_worker: NEW `attachment_read_rejects_symlink_escape` (parent workspace attachment via a symlink
  ⇒ skipped/rejected, never materialized into the child inbox).

### 8.5 Residual coverage (honest statement)

**Closed as a class (Linux openat2; macOS fd-walk):** final- AND intermediate-component symlink
follow, and check-then-open TOCTOU, on `read`/`write`/`list`/`stat`/`revision`/`read_dir`/`create`/
`remove`/`rename` across `LocalEnvironment`, `WorkspaceFs`, tool-fs, lint, job_worker.

**Residuals (documented):**
- **grep/glob** use a path-based `ignore` walker after a `verify_dir` entry check; `follow_links(false)`
  prevents symlink traversal, but the walk re-resolves paths internally (read-only enumeration only).
- **Child/exec cwd on macOS** returns a verified absolute path (no `/proc/self/fd`), leaving a
  dir→symlink swap window between verify and spawn; the exec-os-sandbox track owns spawn hardening.
- **Non-Linux fd-walk** (macOS) has a per-component rename race far weaker than today's full re-walk;
  openat2 (Linux) has none.
- **Windows v1** uses the lexical stub (best-effort; ships no engine fs tools).

---

## 9. Gate (from worktree root, paste tails on implementation)

```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo deny check          # rustix promoted to a direct dep
```

- No wire type changes ⇒ `cargo test -p daemon-api --features arbitrary` **not required**; run it
  anyway if any doubt arises during implementation.
- Machine-load note: if `bins/daemon/tests/host_launch.rs` flakes under the parallel `--workspace`
  run, re-run isolated: `cargo test -p daemon --test host_launch -- --test-threads=2`. Known
  pre-existing flakes (`detached_delegation` ×2, `process_notify` store-seam) are not caused by this
  track — only new/different signatures are real.

Commit on `hardening/contained-root-type`. Do **not** merge; do **not** remove the worktree.

---

## 10. Risks / ambiguities

- **openat2 availability.** Linux 5.6+. Older kernels / restrictive seccomp return ENOSYS/EPERM →
  handled by the `walk_openat` fallback (§2.4). Verified on first implementation build.
- **`rustix` `fs` feature.** `openat2`/`fstatat`/`mkdirat`/`unlinkat`/`renameat` need `features=["fs"]`
  (declared). rustix is already compiled with `fs` transitively, so no resolver churn expected.
- **Blocking on the async reactor.** All fd resolution + byte I/O runs inside `spawn_blocking`
  (mirrors today's `tokio::fs` threadpool behavior and the existing `search` blocking walk).
- **Behavior parity.** `RESOLVE_NO_SYMLINKS` keeps the interim's "reject all symlinks below root"
  posture, so no user-visible regression; relaxing to permit in-root symlinks (drop `NO_SYMLINKS`) is a
  deferred option, not this track.
- **`contain()` retained.** Kept as the non-unix fallback + to avoid a wide public-API break; Phase 4
  `clippy-disallow` fences raw fs around `ContainedRoot`, and a follow-up can privatize `contain` once
  all callers are migrated.
- **Cross-track `run`/execute_code seam** — see §7; recommend explicit review when merging
  exec-os-sandbox.
