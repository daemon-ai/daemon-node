// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`LocalEnvironment`] — the in-core v1 execution backend (§13).
//!
//! Runs commands and file I/O on the local machine, rooted at a per-session workspace directory. All
//! paths are resolved against that root and contained ([`super::contain`]); child commands inherit a
//! scrubbed environment (no inherited secrets) so a tool's exec never leaks the host's credentials.

use super::{contain, Command, ExecCx, ExecResult, ExecutionEnvironment};
use std::path::{Path, PathBuf};
use tokio::io::AsyncReadExt;

/// The local execution environment: a contained per-session workspace on the host filesystem.
pub struct LocalEnvironment {
    root: PathBuf,
}

impl LocalEnvironment {
    /// A local environment rooted at `root` (the session's workspace). The directory is created on
    /// first use; reads/writes/commands are confined to it.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
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
        // Scrubbed child env: nothing inherited (no host secrets leak into a tool's subprocess).
        let mut command = tokio::process::Command::new(&cmd.program);
        command
            .args(&cmd.args)
            .current_dir(&self.root)
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
        let resolved = contain(&self.root, path)?;
        tokio::fs::read(resolved).await
    }

    async fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        let resolved = contain(&self.root, path)?;
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(resolved, bytes).await
    }

    async fn list(&self, path: &Path) -> std::io::Result<Vec<String>> {
        let resolved = contain(&self.root, path)?;
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(resolved).await?;
        while let Some(entry) = dir.next_entry().await? {
            entries.push(entry.file_name().to_string_lossy().into_owned());
        }
        entries.sort();
        Ok(entries)
    }

    fn cwd(&self) -> &Path {
        &self.root
    }
}
