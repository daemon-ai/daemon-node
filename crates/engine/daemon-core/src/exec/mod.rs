// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Execution environments (§13) — the seam a tool uses to touch the world.
//!
//! Tools never spawn subprocesses or open files directly; they go through an [`ExecutionEnvironment`]
//! so the engine controls *where* work happens (a per-session workspace, and later a host-owned or
//! remote env) and *that it stays contained*. The only backend required for v1 is the in-core
//! [`LocalEnvironment`] (this module's `local` submodule), rooted at a per-session workspace dir; the
//! trait is deliberately object-safe so a future host-routed env (driving fs/exec over the §17 host
//! port, [`crate::turn::TurnCx::host`]) drops in without touching the tools.
//!
//! Note (lifecycle §16.1): a *long-running watched* OS process is host-owned. [`ExecutionEnvironment`]
//! runs only **transient** commands that complete within a tool call — it does not own live
//! background processes, so it never blocks the engine from dehydrating at a phase boundary.

pub mod local;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};
use tokio_util::sync::CancellationToken;

pub use local::LocalEnvironment;

/// A command to run in an [`ExecutionEnvironment`] (a transient process, run to completion).
#[derive(Clone, Debug, Default)]
pub struct Command {
    /// The program to execute.
    pub program: String,
    /// Its arguments.
    pub args: Vec<String>,
    /// The working directory to run in, resolved against (and contained within) the environment's
    /// root. `None` runs at the root itself (the historical behavior).
    pub cwd: Option<PathBuf>,
}

impl Command {
    /// A command invoking `program` with no arguments.
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            cwd: None,
        }
    }

    /// Append one argument.
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Append several arguments.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Run in `dir` (workspace-relative or absolute-within-root; the environment contains it).
    pub fn cwd(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cwd = Some(dir.into());
        self
    }
}

/// The captured result of running a [`Command`].
#[derive(Clone, Debug, Default)]
pub struct ExecResult {
    /// The process exit code (`-1` if terminated by a signal / cancellation).
    pub exit_code: i32,
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
}

impl ExecResult {
    /// Whether the command exited successfully (`exit_code == 0`).
    pub fn ok(&self) -> bool {
        self.exit_code == 0
    }
}

/// Per-call execution context: the cooperative cancellation token the env honors mid-run.
pub struct ExecCx<'a> {
    /// Cancellation, checked while a command runs (a kill on cancel).
    pub cancel: &'a CancellationToken,
}

/// A §13 execution environment: a contained place tools read/write files and run commands.
///
/// Object-safe and cloneable behind `Arc`. Every path is resolved relative to [`cwd`](Self::cwd) and
/// must stay within the environment's root — implementations reject escapes (`..`/absolute) so a tool
/// can never read or write outside its workspace.
#[async_trait::async_trait]
pub trait ExecutionEnvironment: Send + Sync {
    /// Run a transient command to completion, honoring `cx.cancel`.
    async fn run(&self, cmd: Command, cx: &ExecCx<'_>) -> std::io::Result<ExecResult>;
    /// Read a (workspace-relative) file.
    async fn read(&self, path: &Path) -> std::io::Result<Vec<u8>>;
    /// Write a (workspace-relative) file, creating parent dirs as needed.
    async fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()>;
    /// List the entries of a (workspace-relative) directory by name.
    async fn list(&self, path: &Path) -> std::io::Result<Vec<String>>;
    /// The environment's working directory (its containment root).
    fn cwd(&self) -> &Path;
}

/// Resolve `requested` against `root` and assert workspace containment.
///
/// Lexical (not symlink-resolving) normalization: it collapses `.`/`..` and rejects any path that
/// would climb above `root`, plus absolute paths that point outside it. This is the containment
/// floor every [`ExecutionEnvironment`] enforces (the §12 `path_security` stage); it works for files
/// that do not exist yet (writes) where `canonicalize` cannot.
pub fn contain(root: &Path, requested: &Path) -> std::io::Result<PathBuf> {
    let joined = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        root.join(requested)
    };

    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::Prefix(p) => normalized.push(p.as_os_str()),
            Component::RootDir => normalized.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(escape_error(requested));
                }
            }
            Component::Normal(seg) => normalized.push(seg),
        }
    }

    if normalized.starts_with(root) {
        Ok(normalized)
    } else {
        Err(escape_error(requested))
    }
}

fn escape_error(requested: &Path) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "path escapes the workspace sandbox: {}",
            requested.display()
        ),
    )
}

// --- Interim symlink / TOCTOU guard (Cluster C stopgap) -------------------------------------------
//
// [`contain`] is lexical: it proves a path *string* stays under `root`, but the subsequent open
// re-walks that path and follows symlinks. Between the `contain` check and the open, or via a
// symlink already present at check time, the final component can redirect the open outside `root`.
// These helpers re-verify at the open that FOLLOWS `contain`, so a symlinked final component (or a
// final component swapped for a symlink) is rejected rather than followed.
//
// COVERAGE (be honest — this is an interim guard, superseded by the Phase 3 cap-std/openat2
// `ContainedRoot`):
//   - CLOSED: a symlinked FINAL component on a file open is refused. On unix this is ATOMIC
//     (`O_NOFOLLOW` on the open itself — no check-then-open window on that component).
//   - NOT CLOSED: intermediate-component symlinks (a symlink at a PARENT directory in the path) are
//     still followed; directory / metadata opens and the whole non-unix fallback re-verify by an
//     `lstat` that precedes the use, so a residual check-then-use TOCTOU window remains. Phase 3
//     (`openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` over an open root fd) eliminates the class.

/// The error returned when a contained path's final component is a symlink we refuse to traverse.
fn symlink_escape_error(path: &Path) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "refusing to follow symlink at workspace path: {}",
            path.display()
        ),
    )
}

/// Open a contained file for reading, refusing a symlinked final component.
///
/// On unix the open carries `O_NOFOLLOW`, so a symlinked final component fails atomically (`ELOOP`)
/// with no check-then-open window. On other platforms an `lstat` pre-check rejects a symlink first
/// (leaving a small residual TOCTOU window; see the module note above).
pub async fn open_read_guarded(path: &Path) -> std::io::Result<tokio::fs::File> {
    #[cfg(unix)]
    {
        // `custom_flags` is tokio's inherent unix method on `OpenOptions` (no trait import needed).
        tokio::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .await
    }
    #[cfg(not(unix))]
    {
        reject_symlink_final(path).await?;
        tokio::fs::File::open(path).await
    }
}

/// Open a contained file for writing (create + truncate), refusing a symlinked final component.
///
/// With `O_NOFOLLOW` on unix: an existing symlink final component is refused (`ELOOP`), an existing
/// regular file is truncated, and a missing file is created as a regular file — matching
/// `tokio::fs::write` semantics for a real target while never following a link. The caller creates
/// parent directories first (as before); only the final open is guarded here.
pub async fn open_write_guarded(path: &Path) -> std::io::Result<tokio::fs::File> {
    #[cfg(unix)]
    {
        // `custom_flags` is tokio's inherent unix method on `OpenOptions` (no trait import needed).
        tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .await
    }
    #[cfg(not(unix))]
    {
        reject_symlink_final(path).await?;
        tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .await
    }
}

/// Reject a path whose FINAL component is a symlink, for directory / metadata opens where an
/// `O_NOFOLLOW` file open does not apply (`read_dir`, and the `cwd` a child runs in). Uses `lstat`,
/// so a non-existent path is allowed (the create case) and this re-verify precedes the use — a
/// residual check-then-use window remains (Phase 3 closes it).
pub async fn reject_symlink_final(path: &Path) -> std::io::Result<()> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(meta) if meta.file_type().is_symlink() => Err(symlink_escape_error(path)),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// As [`reject_symlink_final`], but never rejects the trusted `root` itself — an operator may
/// legitimately bind a workspace onto a symlinked directory (and macOS `/tmp` -> `/private/tmp`).
/// Only components strictly below `root` are guarded. Used by directory ops (list / watch / cwd).
pub async fn reject_symlink_final_below(root: &Path, path: &Path) -> std::io::Result<()> {
    if path == root {
        return Ok(());
    }
    reject_symlink_final(path).await
}

// --- Exec-approval fingerprinting (Cluster B) ----------------------------------------------------
//
// A command's approval decision and operator-facing display are bound to a hash of the FULLY-RESOLVED
// command tuple `(exec-surface, absolute-binary, argv, env-delta, cwd)`. The engine refuses to run a
// parked approval whose resolved tuple no longer matches what was approved/displayed, closing the
// approve-then-swap TOCTOU on the durable HITL path.
//
// The ambient `PATH` value is intentionally NOT part of the tuple: the *resolved absolute binary*
// already captures what `PATH` would have selected at approval time, and hashing the raw `PATH` would
// spuriously refuse on benign daemon-env changes with no added security. Only the explicit env-delta
// (vars the command sets, e.g. `PYTHONUNBUFFERED`) is hashed — never the ambient `PATH`.
//
// COVERAGE (honest): this binds the resolved *path* and argv, not the binary's *contents* — a
// same-absolute-path content swap between approval and exec is a file-level TOCTOU left to the Phase 3
// OS exec sandbox / artifact-provenance work.

/// A stable fingerprint of a fully-resolved command — lowercase-hex SHA-256 over the tuple
/// `(exec-surface, absolute-binary, argv, env-delta, cwd)`. Bound to an exec approval so the engine
/// can refuse a command whose resolved form differs from what was approved (Cluster B).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandFingerprint(String);

impl CommandFingerprint {
    /// Compute the fingerprint over the resolved command tuple. `env_delta` is the *explicit*
    /// environment set for the child (never the ambient `PATH`); it is sorted here so entry order
    /// never changes the digest. Fields are length-prefixed and label-tagged so no two distinct
    /// tuples share an encoding.
    pub fn compute(
        surface: &str,
        program_abs: &Path,
        argv: &[String],
        env_delta: &[(String, String)],
        cwd: &Path,
    ) -> Self {
        let mut env: Vec<(&str, &str)> = env_delta
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        env.sort_unstable();

        let mut h = Sha256::new();
        feed(&mut h, b"surface", surface.as_bytes());
        feed(
            &mut h,
            b"program",
            program_abs.as_os_str().as_encoded_bytes(),
        );
        feed(&mut h, b"argc", &(argv.len() as u64).to_le_bytes());
        for a in argv {
            feed(&mut h, b"arg", a.as_bytes());
        }
        feed(&mut h, b"envc", &(env.len() as u64).to_le_bytes());
        for (k, v) in &env {
            feed(&mut h, b"envk", k.as_bytes());
            feed(&mut h, b"envv", v.as_bytes());
        }
        feed(&mut h, b"cwd", cwd.as_os_str().as_encoded_bytes());

        let digest = h.finalize();
        let mut hex = String::with_capacity(digest.len() * 2);
        for b in digest {
            let _ = write!(hex, "{b:02x}");
        }
        CommandFingerprint(hex)
    }

    /// The full lowercase-hex digest.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A short prefix for compact operator display / correlation.
    pub fn short(&self) -> &str {
        &self.0[..self.0.len().min(12)]
    }
}

/// Length-prefixed, domain-separated feed into the hasher: `len(label) || label || len(bytes) || bytes`,
/// so field boundaries are unambiguous.
fn feed(h: &mut Sha256, label: &[u8], bytes: &[u8]) {
    h.update((label.len() as u64).to_le_bytes());
    h.update(label);
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
}

/// Resolve `program` to the absolute executable path the child will exec, using `path_env` (the
/// child's `PATH`) for a bare name and `cwd` for a relative path. Canonicalizes the result so the
/// fingerprint pins the real target.
///
/// SELF-CONTAINED / purely additive: it does not touch [`contain`], [`Command`], or the
/// [`ExecutionEnvironment`] trait, so it merges cleanly alongside other additions to this module.
///
/// - absolute path: used as-is (verified);
/// - relative path with a separator (`./x`, `sub/x`): resolved against `cwd`;
/// - bare name (`printf`): the first entry in `path_env` that names an executable file wins.
///
/// Errors with [`std::io::ErrorKind::NotFound`] if no executable file is found.
pub fn resolve_program_abs(
    program: &str,
    cwd: &Path,
    path_env: &OsStr,
) -> std::io::Result<PathBuf> {
    let p = Path::new(program);
    let candidate = if p.is_absolute() {
        Some(p.to_path_buf())
    } else if p.components().count() > 1 {
        Some(cwd.join(p))
    } else {
        std::env::split_paths(path_env)
            .map(|dir| dir.join(p))
            .find(|c| is_executable_file(c))
    };
    match candidate {
        Some(c) if is_executable_file(&c) => Ok(abs_preserving_name(&c)),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no executable found for program: {program}"),
        )),
    }
}

/// Return an absolute path that pins the resolved *directory* (canonicalized, so symlinked dirs
/// resolve to a stable location) while KEEPING the original final component name.
///
/// We deliberately do NOT canonicalize the final component: multicall binaries (nix/busybox
/// `coreutils`, where `pwd`/`printf`/… are symlinks to one dispatcher) dispatch on the basename of
/// the exec path, so following that final symlink to `…/coreutils` would break the exec. Pinning the
/// directory + name still prevents `PATH`-divergence between approval and exec; a final-symlink
/// retarget is the same-path-content-swap residual left to the Phase 3 exec sandbox.
fn abs_preserving_name(candidate: &Path) -> PathBuf {
    match (candidate.parent(), candidate.file_name()) {
        (Some(parent), Some(name)) => match parent.canonicalize() {
            Ok(dir) => dir.join(name),
            Err(_) => candidate.to_path_buf(),
        },
        _ => candidate.to_path_buf(),
    }
}

/// Whether `path` names a regular file with an executable bit (unix) / any regular file (non-unix).
fn is_executable_file(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(m) if m.is_file() => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                m.permissions().mode() & 0o111 != 0
            }
            #[cfg(not(unix))]
            {
                true
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn command_fingerprint_is_field_sensitive_and_deterministic() {
        let base = || {
            CommandFingerprint::compute(
                "exec.argv",
                Path::new("/bin/x"),
                &["--flag".to_string()],
                &[],
                Path::new("/ws"),
            )
        };
        // Same tuple -> same digest.
        assert_eq!(base(), base());
        // Every field participates in the identity.
        assert_ne!(
            base(),
            CommandFingerprint::compute(
                "exec.shell",
                Path::new("/bin/x"),
                &["--flag".to_string()],
                &[],
                Path::new("/ws")
            )
        );
        assert_ne!(
            base(),
            CommandFingerprint::compute(
                "exec.argv",
                Path::new("/bin/y"),
                &["--flag".to_string()],
                &[],
                Path::new("/ws")
            )
        );
        assert_ne!(
            base(),
            CommandFingerprint::compute(
                "exec.argv",
                Path::new("/bin/x"),
                &["--other".to_string()],
                &[],
                Path::new("/ws")
            )
        );
        assert_ne!(
            base(),
            CommandFingerprint::compute(
                "exec.argv",
                Path::new("/bin/x"),
                &["--flag".to_string()],
                &[("K".to_string(), "V".to_string())],
                Path::new("/ws")
            )
        );
        assert_ne!(
            base(),
            CommandFingerprint::compute(
                "exec.argv",
                Path::new("/bin/x"),
                &["--flag".to_string()],
                &[],
                Path::new("/elsewhere")
            )
        );
        // env-delta is order-insensitive (sorted before hashing).
        let e1 = CommandFingerprint::compute(
            "exec.argv",
            Path::new("/bin/x"),
            &[],
            &[
                ("A".to_string(), "1".to_string()),
                ("B".to_string(), "2".to_string()),
            ],
            Path::new("/ws"),
        );
        let e2 = CommandFingerprint::compute(
            "exec.argv",
            Path::new("/bin/x"),
            &[],
            &[
                ("B".to_string(), "2".to_string()),
                ("A".to_string(), "1".to_string()),
            ],
            Path::new("/ws"),
        );
        assert_eq!(e1, e2);
    }

    #[cfg(unix)]
    #[test]
    fn resolve_program_abs_resolves_path_binary_and_rejects_missing() {
        let path = std::env::var_os("PATH").unwrap_or_default();
        let cwd = std::env::temp_dir();
        // A real PATH binary resolves to an absolute path that PRESERVES its final name (so a
        // multicall dispatcher like coreutils still dispatches correctly on exec).
        let sh = resolve_program_abs("sh", &cwd, &path).expect("sh is on PATH");
        assert!(
            sh.is_absolute(),
            "resolved to an absolute path: {}",
            sh.display()
        );
        assert_eq!(
            sh.file_name().and_then(|n| n.to_str()),
            Some("sh"),
            "final component name preserved: {}",
            sh.display()
        );
        // A bare name not on PATH is a NotFound error.
        let err = resolve_program_abs("definitely-not-a-real-binary-xyz", &cwd, &path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn contain_accepts_relative_and_rejects_escapes() {
        let root = Path::new("/ws/session");
        assert_eq!(
            contain(root, Path::new("a/b.txt")).unwrap(),
            PathBuf::from("/ws/session/a/b.txt")
        );
        // `..` that climbs above the root is rejected.
        assert!(contain(root, Path::new("../secret")).is_err());
        assert!(contain(root, Path::new("a/../../secret")).is_err());
        // An absolute path outside the root is rejected.
        assert!(contain(root, Path::new("/etc/passwd")).is_err());
        // `..` that stays within the root is fine.
        assert_eq!(
            contain(root, Path::new("a/../b.txt")).unwrap(),
            PathBuf::from("/ws/session/b.txt")
        );
    }

    fn temp_root(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("daemon-exec-test-{tag}-{nanos}"))
    }

    #[tokio::test]
    async fn local_env_write_read_list_roundtrip() {
        let root = temp_root("rw");
        let env = LocalEnvironment::new(&root);
        env.write(Path::new("sub/hello.txt"), b"hi there")
            .await
            .unwrap();
        let bytes = env.read(Path::new("sub/hello.txt")).await.unwrap();
        assert_eq!(bytes, b"hi there");
        let entries = env.list(Path::new("sub")).await.unwrap();
        assert_eq!(entries, vec!["hello.txt".to_string()]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn local_env_rejects_out_of_workspace_write() {
        let root = temp_root("escape");
        let env = LocalEnvironment::new(&root);
        let err = env
            .write(Path::new("../escaped.txt"), b"nope")
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn local_env_runs_a_command() {
        let root = temp_root("run");
        let env = LocalEnvironment::new(&root);
        let cancel = CancellationToken::new();
        let result = env
            .run(
                Command::new("printf").arg("out-%s").arg("ok"),
                &ExecCx { cancel: &cancel },
            )
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("out-ok"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
