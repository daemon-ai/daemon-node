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
}

impl Command {
    /// A command invoking `program` with no arguments.
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

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
