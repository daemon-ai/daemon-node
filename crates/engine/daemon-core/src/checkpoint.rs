//! Tool checkpoints + rewind (§12 safety) — the reserved checkpoint stage of the tool pipeline.
//!
//! Before a **mutating** tool runs (an fs write/edit or a shell command — see [`Tool::mutates`]),
//! the engine records a [`CheckpointRecord`] of the workspace so an operator/GUI can later rewind to
//! it. Capture is **best-effort and non-blocking**: a checkpoint failure logs and continues — it
//! never fails the turn (a safety net must not become a new failure mode).
//!
//! Two capture strategies, tried in order over the [`ExecutionEnvironment`] root:
//! - **git-first**: when the workspace is a git repo, a non-destructive `git stash create` records
//!   the full working tree as a dangling commit (the working tree and index are left untouched), and
//!   rewind restores tracked files from it.
//! - **snapshot fallback**: otherwise a recursive copy of the workspace (minus `.git`) under the
//!   store's data-root; rewind clears the workspace and copies the snapshot back.
//!
//! The ledger is a JSON-lines file under the data-root, so it survives a node restart and backs the
//! `Checkpoint{List,Rewind}` control surface ([`crate::control`] / `daemon-api`).
//!
//! [`Tool::mutates`]: crate::tools::Tool::mutates

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tracing::Instrument;

use crate::exec::ExecutionEnvironment;

/// How a checkpoint captured the workspace (determines how [`CheckpointStore::restore`] rewinds).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointKind {
    /// A non-destructive git stash/commit sha capturing the tracked working tree.
    Git {
        /// The dangling commit sha (`git stash create` output, or `HEAD` when the tree was clean).
        reference: String,
    },
    /// A recursive copy of the workspace under the store's data-root.
    Snapshot {
        /// The snapshot directory (absolute).
        dir: String,
    },
}

/// One recorded checkpoint: enough to list it (`session`/`tool`/`created_unix`) and to rewind to it
/// (`workspace` + `kind`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointRecord {
    /// The checkpoint's stable id (the rewind key).
    pub id: String,
    /// The session whose turn produced it.
    pub session: String,
    /// The tool call that triggered it.
    pub call_id: String,
    /// The mutating tool's name.
    pub tool: String,
    /// Unix seconds at capture.
    pub created_unix: u64,
    /// The captured workspace root (absolute) — where a rewind restores to.
    pub workspace: String,
    /// The capture strategy + its locator.
    pub kind: CheckpointKind,
}

/// The checkpoint ledger + capture/restore engine (§12). Object-safe so the host can hold it behind
/// `Arc<dyn CheckpointStore>` and share one instance across engines and the control surface.
#[async_trait::async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Record a checkpoint of `env`'s workspace before a mutating tool runs. Returns the record on
    /// success, `None` on a best-effort failure (logged by the impl; never propagated as a turn error).
    async fn capture(
        &self,
        session: &str,
        call_id: &str,
        tool: &str,
        env: &dyn ExecutionEnvironment,
    ) -> Option<CheckpointRecord>;

    /// Rewind the workspace to a recorded checkpoint.
    async fn restore(&self, record: &CheckpointRecord) -> std::io::Result<()>;

    /// List recorded checkpoints, newest first — all sessions, or one when `session` is given.
    async fn list(&self, session: Option<&str>) -> Vec<CheckpointRecord>;

    /// Fetch one record by id.
    async fn get(&self, id: &str) -> Option<CheckpointRecord>;
}

/// The default [`CheckpointStore`]: a git-first / snapshot-fallback store with a JSON-lines ledger
/// under `root` (typically `<data_dir>/checkpoints`).
pub struct LocalCheckpointStore {
    root: PathBuf,
    /// Serializes ledger appends (the file is the source of truth; the lock just orders writers).
    ledger_lock: Mutex<()>,
}

impl LocalCheckpointStore {
    /// A store rooted at `root` (created on first capture).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            ledger_lock: Mutex::new(()),
        }
    }

    fn ledger_path(&self) -> PathBuf {
        self.root.join("ledger.jsonl")
    }

    fn snapshots_root(&self) -> PathBuf {
        self.root.join("snapshots")
    }

    fn append_ledger(&self, record: &CheckpointRecord) -> std::io::Result<()> {
        use std::io::Write;
        let _guard = self.ledger_lock.lock().expect("checkpoint ledger poisoned");
        std::fs::create_dir_all(&self.root)?;
        let line = serde_json::to_string(record).map_err(std::io::Error::other)?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.ledger_path())?;
        writeln!(f, "{line}")
    }

    fn read_ledger(&self) -> Vec<CheckpointRecord> {
        let Ok(text) = std::fs::read_to_string(self.ledger_path()) else {
            return Vec::new();
        };
        text.lines()
            .filter_map(|l| serde_json::from_str::<CheckpointRecord>(l).ok())
            .collect()
    }
}

#[async_trait::async_trait]
impl CheckpointStore for LocalCheckpointStore {
    async fn capture(
        &self,
        session: &str,
        call_id: &str,
        tool: &str,
        env: &dyn ExecutionEnvironment,
    ) -> Option<CheckpointRecord> {
        let workspace = env.cwd().to_path_buf();
        let id = checkpoint_id(session, call_id);
        let snapshots_root = self.snapshots_root();
        let id_for_blocking = id.clone();
        let ws_for_blocking = workspace.clone();

        // All fs/git work is synchronous; keep it off the async worker.
        let span = tracing::debug_span!(
            "checkpoint.capture",
            session,
            call_id,
            tool,
            checkpoint_id = %id
        );
        let kind = tokio::task::spawn_blocking(move || {
            capture_blocking(&ws_for_blocking, &snapshots_root, &id_for_blocking)
        })
        .instrument(span)
        .await
        .ok()
        .flatten()?;

        let record = CheckpointRecord {
            id,
            session: session.to_string(),
            call_id: call_id.to_string(),
            tool: tool.to_string(),
            created_unix: now_unix(),
            workspace: workspace.to_string_lossy().into_owned(),
            kind,
        };
        if let Err(e) = self.append_ledger(&record) {
            tracing::warn!(
                error = %e,
                session,
                call_id,
                tool,
                checkpoint_id = %record.id,
                "checkpoint.capture.failed"
            );
            return None;
        }
        tracing::debug!(
            session,
            call_id,
            tool,
            checkpoint_id = %record.id,
            kind = ?record.kind,
            "checkpoint.capture"
        );
        Some(record)
    }

    async fn restore(&self, record: &CheckpointRecord) -> std::io::Result<()> {
        let span = tracing::debug_span!(
            "checkpoint.restore",
            session = %record.session,
            call_id = %record.call_id,
            tool = %record.tool,
            checkpoint_id = %record.id,
            kind = ?record.kind
        );
        let record = record.clone();
        tokio::task::spawn_blocking(move || restore_blocking(&record))
            .instrument(span)
            .await
            .map_err(std::io::Error::other)?
    }

    async fn list(&self, session: Option<&str>) -> Vec<CheckpointRecord> {
        let mut records = self.read_ledger();
        if let Some(s) = session {
            records.retain(|r| r.session == s);
        }
        records.reverse(); // newest first
        records
    }

    async fn get(&self, id: &str) -> Option<CheckpointRecord> {
        self.read_ledger().into_iter().find(|r| r.id == id)
    }
}

/// Capture the workspace, git-first then snapshot-fallback. Returns `None` only on a hard failure.
fn capture_blocking(workspace: &Path, snapshots_root: &Path, id: &str) -> Option<CheckpointKind> {
    if workspace.join(".git").exists() {
        if let Some(reference) = git_stash_create(workspace) {
            return Some(CheckpointKind::Git { reference });
        }
    }
    let dir = snapshots_root.join(id);
    match copy_tree(workspace, &dir) {
        Ok(()) => Some(CheckpointKind::Snapshot {
            dir: dir.to_string_lossy().into_owned(),
        }),
        Err(e) => {
            tracing::warn!(error = %e, "checkpoint snapshot copy failed");
            None
        }
    }
}

/// Rewind to a recorded checkpoint.
fn restore_blocking(record: &CheckpointRecord) -> std::io::Result<()> {
    let workspace = PathBuf::from(&record.workspace);
    match &record.kind {
        CheckpointKind::Git { reference } => {
            // Restore tracked files to the checkpoint state (non-destructive to history).
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(&workspace)
                .arg("checkout")
                .arg(reference)
                .arg("--")
                .arg(".")
                .status()?;
            if status.success() {
                Ok(())
            } else {
                Err(std::io::Error::other(format!(
                    "git checkout {reference} failed"
                )))
            }
        }
        CheckpointKind::Snapshot { dir } => {
            clear_dir_except_git(&workspace)?;
            copy_tree(Path::new(dir), &workspace)
        }
    }
}

/// `git stash create`: records the working tree as a dangling commit without touching the tree or
/// index. Returns the sha, or `HEAD`'s sha when the tree is clean (stash-create prints nothing).
fn git_stash_create(workspace: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(workspace)
        .arg("stash")
        .arg("create")
        .arg("daemon-checkpoint")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if !sha.is_empty() {
        return Some(sha);
    }
    // Clean tree: fall back to HEAD so a rewind still restores tracked files to a known commit.
    let head = std::process::Command::new("git")
        .arg("-C")
        .arg(workspace)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .ok()?;
    if !head.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&head.stdout).trim().to_string();
    (!sha.is_empty()).then_some(sha)
}

/// Recursively copy `src` into `dst` (creating `dst`), skipping any `.git` directory.
fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == ".git" {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        let ft = entry.file_type()?;
        if ft.is_dir() {
            copy_tree(&from, &to)?;
        } else if ft.is_file() {
            std::fs::copy(&from, &to)?;
        }
        // Symlinks and special files are skipped (a workspace snapshot is best-effort).
    }
    Ok(())
}

/// Remove every entry of `dir` except `.git` (the pre-restore clear of a snapshot rewind).
fn clear_dir_except_git(dir: &Path) -> std::io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_name() == ".git" {
            continue;
        }
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(&path)?;
        } else {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}

/// A filesystem-safe, collision-resistant checkpoint id from `(session, call_id)` + a nanosecond tag.
fn checkpoint_id(session: &str, call_id: &str) -> String {
    let sanitize = |s: &str| -> String {
        s.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    };
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}-{nanos}", sanitize(session), sanitize(call_id))
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::LocalEnvironment;

    fn temp_root(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("daemon-ckpt-{tag}-{nanos}"))
    }

    #[tokio::test]
    async fn snapshot_capture_and_rewind_round_trip() {
        let ws = temp_root("ws");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("file.txt"), b"v1").unwrap();
        std::fs::create_dir_all(ws.join("sub")).unwrap();
        std::fs::write(ws.join("sub/nested.txt"), b"nested-v1").unwrap();

        let store_root = temp_root("store");
        let store = LocalCheckpointStore::new(&store_root);
        let env = LocalEnvironment::new(&ws);

        let record = store
            .capture("sess", "call-1", "fs", &env)
            .await
            .expect("capture");
        assert!(matches!(record.kind, CheckpointKind::Snapshot { .. }));

        // Mutate after the checkpoint, then rewind.
        std::fs::write(ws.join("file.txt"), b"v2-edited").unwrap();
        std::fs::write(ws.join("brand-new.txt"), b"added").unwrap();
        std::fs::remove_file(ws.join("sub/nested.txt")).unwrap();

        store.restore(&record).await.expect("restore");

        assert_eq!(std::fs::read(ws.join("file.txt")).unwrap(), b"v1");
        assert_eq!(
            std::fs::read(ws.join("sub/nested.txt")).unwrap(),
            b"nested-v1"
        );
        // A file created after the checkpoint is removed by the rewind.
        assert!(!ws.join("brand-new.txt").exists());

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&store_root);
    }

    #[tokio::test]
    async fn ledger_lists_newest_first_and_survives_reopen() {
        let ws = temp_root("ws2");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("a.txt"), b"a").unwrap();
        let store_root = temp_root("store2");
        let env = LocalEnvironment::new(&ws);

        {
            let store = LocalCheckpointStore::new(&store_root);
            store.capture("s1", "c1", "fs", &env).await.unwrap();
            store.capture("s2", "c2", "shell", &env).await.unwrap();
        }
        // A fresh store reads the persisted ledger (survives "restart").
        let store = LocalCheckpointStore::new(&store_root);
        let all = store.list(None).await;
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].session, "s2"); // newest first
        let only_s1 = store.list(Some("s1")).await;
        assert_eq!(only_s1.len(), 1);
        assert_eq!(only_s1[0].tool, "fs");

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&store_root);
    }
}
