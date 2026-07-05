// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`LocalEnvironment`] — the in-core v1 execution backend (§13).
//!
//! Runs commands and file I/O on the local machine, rooted at a per-session workspace directory. All
//! paths are resolved against that root and contained ([`super::contain`]); child commands inherit a
//! scrubbed environment (no inherited secrets) so a tool's exec never leaks the host's credentials.

use super::{Command, ContainedRoot, ExecCx, ExecResult, ExecutionEnvironment};
use std::path::{Path, PathBuf};
use tokio::io::AsyncReadExt;

/// The local execution environment: a contained per-session workspace on the host filesystem.
pub struct LocalEnvironment {
    root: PathBuf,
    /// Whether `root` is a node-managed isolated sandbox (`true`) vs. an operator-bound external
    /// directory whose contents may be attacker-influenced (`false`). Surfaced via
    /// [`ExecutionEnvironment::workspace_trusted`]; consumed by tools that must not auto-trust
    /// workspace-discovered artifacts on an untrusted root (Cluster E — `execute_code` venv trust).
    trusted: bool,
}

impl LocalEnvironment {
    /// A local environment rooted at `root` (the session's workspace), treated as **trusted** (the
    /// node-managed isolated-sandbox case). The directory is created on first use; reads/writes/
    /// commands are confined to it.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            trusted: true,
        }
    }

    /// A local environment rooted at `root` with an explicit trust flag. Pass `trusted = false` for
    /// an operator-bound external directory (`WorkspaceBinding::Bound`), whose contents may be
    /// attacker-influenced, so tools suppress auto-trust of workspace-discovered artifacts.
    pub fn with_trust(root: impl Into<PathBuf>, trusted: bool) -> Self {
        Self {
            root: root.into(),
            trusted,
        }
    }

    /// A local environment under the OS temp dir, keyed by `session` — the default sandbox when the
    /// host has not provisioned a workspace (in-process and tests).
    pub fn sandbox(session: &str) -> Self {
        let safe: String = session
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        Self::new(std::env::temp_dir().join(format!("daemon-ws-{safe}")))
    }

    async fn ensure_root(&self) -> std::io::Result<()> {
        tokio::fs::create_dir_all(&self.root).await
    }
}

#[async_trait::async_trait]
impl ExecutionEnvironment for LocalEnvironment {
    async fn run(&self, cmd: Command, cx: &ExecCx<'_>) -> std::io::Result<ExecResult> {
        self.ensure_root().await?;
        // A per-command working directory resolves against — and must stay within — the root
        // (`shell(workdir=...)`), created on demand. `ContainedRoot::child_cwd` proves containment
        // via openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS) and hands back a spawn-safe cwd — on
        // Linux `/proc/self/fd/N`, so the kernel resolves the *verified* fd at spawn with no path
        // re-resolution (no symlink escape, no verify→spawn TOCTOU). The guard keeps that fd alive
        // until after the child is spawned. (Spawn mechanics below are the exec-os-sandbox seam.)
        let cwd_guard = match &cmd.cwd {
            Some(requested) => Some(
                ContainedRoot::open(&self.root)?
                    .child_cwd(requested)
                    .await?,
            ),
            None => None,
        };
        let dir: &Path = cwd_guard
            .as_ref()
            .map(|g| g.path.as_path())
            .unwrap_or(&self.root);
        // Scrubbed child env: nothing inherited (no host secrets leak into a tool's subprocess).
        let mut command = tokio::process::Command::new(&cmd.program);
        command
            .args(&cmd.args)
            .current_dir(dir)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = command.spawn()?;
        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();

        let status = tokio::select! {
            status = child.wait() => status?,
            _ = cx.cancel.cancelled() => {
                // Cancellation: kill the transient process and report a signal-style exit.
                let _ = child.kill().await;
                return Ok(ExecResult {
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: "cancelled".into(),
                });
            }
        };

        let mut stdout = String::new();
        if let Some(pipe) = stdout_pipe.as_mut() {
            let mut buf = Vec::new();
            pipe.read_to_end(&mut buf).await?;
            stdout = String::from_utf8_lossy(&buf).into_owned();
        }
        let mut stderr = String::new();
        if let Some(pipe) = stderr_pipe.as_mut() {
            let mut buf = Vec::new();
            pipe.read_to_end(&mut buf).await?;
            stderr = String::from_utf8_lossy(&buf).into_owned();
        }

        Ok(ExecResult {
            exit_code: status.code().unwrap_or(-1),
            stdout,
            stderr,
        })
    }

    async fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        // ContainedRoot resolves via openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS): a symlink at ANY
        // component (intermediate or final) is rejected atomically — no follow, no check-then-open.
        ContainedRoot::open(&self.root)?.read(path).await
    }

    async fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        // Parent dirs are created contained (each component O_NOFOLLOW) and the file is opened
        // openat2-relative, so a write can never clobber a file outside the workspace via a symlink.
        ContainedRoot::open(&self.root)?.write(path, bytes).await?;
        Ok(())
    }

    async fn list(&self, path: &Path) -> std::io::Result<Vec<String>> {
        let entries = ContainedRoot::open(&self.root)?.read_dir(path).await?;
        Ok(entries.into_iter().map(|e| e.name).collect())
    }

    fn cwd(&self) -> &Path {
        &self.root
    }

    fn workspace_trusted(&self) -> bool {
        self.trusted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    fn temp_root(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("daemon-exec-local-{tag}-{nanos}"))
    }

    #[tokio::test]
    async fn command_cwd_runs_in_the_contained_subdir() {
        let root = temp_root("cwd");
        let env = LocalEnvironment::new(&root);
        let cancel = CancellationToken::new();
        let result = env
            .run(
                Command::new("pwd").cwd("nested/dir"),
                &ExecCx { cancel: &cancel },
            )
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(
            result.stdout.trim().ends_with("nested/dir"),
            "ran in the requested subdir: {}",
            result.stdout
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn command_cwd_escaping_the_root_is_rejected() {
        let root = temp_root("cwd-escape");
        let env = LocalEnvironment::new(&root);
        let cancel = CancellationToken::new();
        let err = env
            .run(
                Command::new("pwd").cwd("../outside"),
                &ExecCx { cancel: &cancel },
            )
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
        let _ = std::fs::remove_dir_all(&root);
    }

    // Cluster C interim guard: a symlink whose final component points OUTSIDE the workspace is
    // lexically contained ("link.txt" starts_with root) but must not be followed on the open.
    // Before the guard, `read` followed the link and returned the outside secret; after, it is
    // rejected. (Unix: O_NOFOLLOW -> ELOOP; the invariant asserted is "no escape", not the kind.)
    #[cfg(unix)]
    #[tokio::test]
    async fn read_rejects_symlinked_final_component() {
        use std::os::unix::fs::symlink;
        let root = temp_root("symlink-read");
        std::fs::create_dir_all(&root).unwrap();
        let outside = temp_root("symlink-read-secret");
        std::fs::create_dir_all(&outside).unwrap();
        let secret = outside.join("secret.txt");
        std::fs::write(&secret, b"TOP SECRET").unwrap();
        symlink(&secret, root.join("link.txt")).unwrap();

        let env = LocalEnvironment::new(&root);
        let result = env.read(Path::new("link.txt")).await;
        assert!(
            result.is_err(),
            "reading through an escaping symlink must be rejected, got {result:?}"
        );
        // A real in-root file still reads fine (the guard rejects only symlinks, not ordinary files).
        env.write(Path::new("real.txt"), b"ok").await.unwrap();
        assert_eq!(env.read(Path::new("real.txt")).await.unwrap(), b"ok");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // Before the guard, `write` followed a symlinked final component and clobbered the outside
    // target; after, it is rejected and the outside target is untouched.
    #[cfg(unix)]
    #[tokio::test]
    async fn write_rejects_symlinked_final_component() {
        use std::os::unix::fs::symlink;
        let root = temp_root("symlink-write");
        std::fs::create_dir_all(&root).unwrap();
        let outside = temp_root("symlink-write-target");
        std::fs::create_dir_all(&outside).unwrap();
        let target = outside.join("target.txt");
        std::fs::write(&target, b"ORIGINAL").unwrap();
        symlink(&target, root.join("link.txt")).unwrap();

        let env = LocalEnvironment::new(&root);
        let result = env.write(Path::new("link.txt"), b"OVERWRITTEN").await;
        assert!(
            result.is_err(),
            "writing through an escaping symlink must be rejected, got {result:?}"
        );
        assert_eq!(
            std::fs::read(&target).unwrap(),
            b"ORIGINAL",
            "the outside target must not be written through the symlink"
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // Dir-op case: a symlinked directory final component is rejected by the lstat-based guard
    // (PermissionDenied), so a `list` cannot enumerate an outside directory through a link.
    #[cfg(unix)]
    #[tokio::test]
    async fn list_rejects_symlinked_dir() {
        use std::os::unix::fs::symlink;
        let root = temp_root("symlink-list");
        std::fs::create_dir_all(&root).unwrap();
        let outside = temp_root("symlink-list-target");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("hidden.txt"), b"x").unwrap();
        symlink(&outside, root.join("sub")).unwrap();

        let env = LocalEnvironment::new(&root);
        let err = env.list(Path::new("sub")).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // A command whose contained cwd resolves to a symlinked directory is rejected before the child
    // spawns, so a tool cannot run outside the workspace via a cwd symlink.
    #[cfg(unix)]
    #[tokio::test]
    async fn run_cwd_rejects_symlinked_dir() {
        use std::os::unix::fs::symlink;
        let root = temp_root("symlink-cwd");
        std::fs::create_dir_all(&root).unwrap();
        let outside = temp_root("symlink-cwd-target");
        std::fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("cwdlink")).unwrap();

        let env = LocalEnvironment::new(&root);
        let cancel = CancellationToken::new();
        let err = env
            .run(
                Command::new("pwd").cwd("cwdlink"),
                &ExecCx { cancel: &cancel },
            )
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // Phase 3 (ContainedRoot): the case the interim guard did NOT close — an INTERMEDIATE path
    // component is a symlink out of the workspace. The interim only guarded the FINAL component, so
    // `sub/secret.txt` (final = an ordinary file) was followed through the symlinked parent `sub`.
    // `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` rejects a symlink at ANY component. These
    // tests FAIL on the interim tree (the read/write/run succeeds = escape) and pass after the fix.
    #[cfg(unix)]
    #[tokio::test]
    async fn read_rejects_intermediate_symlink() {
        use std::os::unix::fs::symlink;
        let root = temp_root("isym-read");
        std::fs::create_dir_all(&root).unwrap();
        let outside = temp_root("isym-read-secret");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), b"TOP SECRET").unwrap();
        // `sub` is an intermediate component that is a symlink to the outside dir.
        symlink(&outside, root.join("sub")).unwrap();

        let env = LocalEnvironment::new(&root);
        let result = env.read(Path::new("sub/secret.txt")).await;
        assert!(
            result.is_err(),
            "reading through an intermediate symlinked dir must be rejected, got {result:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_rejects_intermediate_symlink() {
        use std::os::unix::fs::symlink;
        let root = temp_root("isym-write");
        std::fs::create_dir_all(&root).unwrap();
        let outside = temp_root("isym-write-target");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("target.txt"), b"ORIGINAL").unwrap();
        symlink(&outside, root.join("sub")).unwrap();

        let env = LocalEnvironment::new(&root);
        let result = env.write(Path::new("sub/target.txt"), b"OVERWRITTEN").await;
        assert!(
            result.is_err(),
            "writing through an intermediate symlinked dir must be rejected, got {result:?}"
        );
        assert_eq!(
            std::fs::read(outside.join("target.txt")).unwrap(),
            b"ORIGINAL",
            "the outside target must not be written through the intermediate symlink"
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_cwd_rejects_intermediate_symlink() {
        use std::os::unix::fs::symlink;
        let root = temp_root("isym-cwd");
        std::fs::create_dir_all(&root).unwrap();
        let outside = temp_root("isym-cwd-target");
        std::fs::create_dir_all(outside.join("inner")).unwrap();
        symlink(&outside, root.join("sub")).unwrap();

        let env = LocalEnvironment::new(&root);
        let cancel = CancellationToken::new();
        let err = env
            .run(
                Command::new("pwd").cwd("sub/inner"),
                &ExecCx { cancel: &cancel },
            )
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }
}
