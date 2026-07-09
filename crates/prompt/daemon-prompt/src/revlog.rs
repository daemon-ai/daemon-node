// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: fs here is the daemon-internal persona revision log under the caller-supplied profile
// data dir (node-managed, not attacker-influenced). Raw fs allowed file-wide; no process spawns.
#![allow(clippy::disallowed_methods)]

//! A minimal append-only JSONL revision log for persona (`SOUL.md`) changes.
//!
//! Deliberately local to this crate: reusing `daemon-host`'s `FileRevisionLog` would drag the
//! whole host crate into this content-only crate (and `daemon-common`'s `RevisionKind` has no
//! persona variant). The line shape mirrors `daemon-host/src/revision.rs`'s `index.jsonl`
//! convention (`seq`, `parent`, `hash`, `author`, `reason`, `ts_ms`; `author` serialized
//! snake_case exactly like `daemon_common::Author`), so the integration lane can later swap in
//! the host log without a data migration if it wants one mechanism.
//!
//! One log = one JSONL file, kept next to the artifact it versions. Unlike the host log there is
//! no blob store: persona texts are small and each line's `hash` (SHA-256 of the content) plus
//! the live `SOUL.md` are enough for audit/dedupe purposes at this layer.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::PromptError;

/// Who authored a revision. Mirrors `daemon_common::Author` (same variants, same snake_case
/// serialization) without depending on that crate — this crate's only daemon dependency is
/// daemon-core's `ExecutionEnvironment`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Author {
    /// A human operator acting over the node's control surface.
    Operator,
    /// The agent itself, labeled by the write source (e.g. `profile_manage`).
    Agent(String),
}

/// One recorded revision: a JSONL line in the log file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionEntry {
    /// The 1-based monotonic sequence within this log.
    pub seq: u64,
    /// The previous head's `seq` (`None` for the first revision).
    pub parent: Option<u64>,
    /// Lowercase-hex SHA-256 of the revision's content.
    pub hash: String,
    /// Who wrote it.
    pub author: Author,
    /// Why (a short free-form reason, e.g. `seed default persona`).
    pub reason: String,
    /// Unix milliseconds when it was recorded.
    pub ts_ms: u64,
}

/// An append-only JSONL revision log at a fixed path (created on first append).
pub(crate) struct RevisionLog {
    path: PathBuf,
}

impl RevisionLog {
    /// A log backed by the JSONL file at `path`.
    pub(crate) fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Append a revision of `blob`, returning the recorded entry. The new entry becomes the head.
    pub(crate) fn append(
        &self,
        blob: &[u8],
        author: Author,
        reason: &str,
    ) -> Result<RevisionEntry, PromptError> {
        let existing = self.entries()?;
        let parent = existing.last().map(|e| e.seq);
        let entry = RevisionEntry {
            seq: parent.unwrap_or(0) + 1,
            parent,
            hash: sha256_hex(blob),
            author,
            reason: reason.to_string(),
            ts_ms: now_ms(),
        };
        let encoded =
            serde_json::to_string(&entry).map_err(|e| PromptError::Codec(e.to_string()))?;
        if let Some(parent_dir) = self.path.parent() {
            std::fs::create_dir_all(parent_dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(f, "{encoded}")?;
        Ok(entry)
    }

    /// The full history, oldest first. Empty (not an error) when the log file doesn't exist.
    pub(crate) fn entries(&self) -> Result<Vec<RevisionEntry>, PromptError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let body = std::fs::read_to_string(&self.path)?;
        let mut out = Vec::new();
        for line in body.lines() {
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(line).map_err(|e| PromptError::Codec(e.to_string()))?);
        }
        Ok(out)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Atomically write `content` to `path`: temp file in the same directory (same filesystem, so
/// the rename is atomic) + `rename`. Readers always see either the old complete file or the new
/// one — never a truncated intermediate.
pub(crate) fn atomic_write(path: &Path, content: &str) -> Result<(), PromptError> {
    let parent = path
        .parent()
        .ok_or_else(|| PromptError::Io(format!("no parent dir for {}", path.display())))?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.tmp.{}.{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("write"),
        std::process::id(),
        now_ms(),
    ));
    let write_and_rename = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(content.as_bytes())?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if write_and_rename.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    write_and_rename.map_err(Into::into)
}

/// Restrict a profile id to a filename-safe slug (mirrors the daemon-host revision-store
/// sanitizer: `[A-Za-z0-9._-]`, everything else becomes `_`). Additionally, an id that would
/// sanitize to a path-traversal component (`.`, `..`) or nothing at all becomes `_` — a store
/// keyed by attacker-chosen ids must never join `..` onto its data dir.
pub(crate) fn sanitize_id(id: &str) -> String {
    let slug: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if slug.is_empty() || slug.chars().all(|c| c == '.') {
        return "_".to_string();
    }
    slug
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_is_monotonic_and_durable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("SOUL.revisions.jsonl");
        let log = RevisionLog::at(&path);

        let r1 = log.append(b"v1", Author::Operator, "seed").unwrap();
        assert_eq!(r1.seq, 1);
        assert_eq!(r1.parent, None);
        let r2 = log
            .append(b"v2", Author::Agent("profile_manage".into()), "update")
            .unwrap();
        assert_eq!(r2.seq, 2);
        assert_eq!(r2.parent, Some(1));
        assert_ne!(r1.hash, r2.hash);

        // Survives reopen (durable) and preserves order + provenance.
        let reopened = RevisionLog::at(&path);
        let entries = reopened.entries().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].author, Author::Operator);
        assert_eq!(entries[1].author, Author::Agent("profile_manage".into()));
        assert_eq!(entries[1].reason, "update");
    }

    #[test]
    fn identical_content_hashes_identically() {
        let dir = tempfile::tempdir().unwrap();
        let log = RevisionLog::at(dir.path().join("log.jsonl"));
        let a = log.append(b"same", Author::Operator, "one").unwrap();
        let b = log.append(b"same", Author::Operator, "two").unwrap();
        assert_eq!(a.hash, b.hash);
        assert_ne!(a.seq, b.seq);
    }

    #[test]
    fn author_serializes_snake_case_like_daemon_common() {
        // Pin the wire shape so a later swap to daemon-host's FileRevisionLog reads our lines.
        assert_eq!(
            serde_json::to_string(&Author::Operator).unwrap(),
            r#""operator""#
        );
        assert_eq!(
            serde_json::to_string(&Author::Agent("soul_set".into())).unwrap(),
            r#"{"agent":"soul_set"}"#
        );
    }

    #[test]
    fn atomic_write_replaces_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("file.md");
        atomic_write(&path, "first").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first");
        atomic_write(&path, "second").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
        // No temp litter left behind.
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn sanitize_id_keeps_safe_chars_only() {
        assert_eq!(sanitize_id("opus-4.6_x"), "opus-4.6_x");
        assert_eq!(sanitize_id("../escape me"), ".._escape_me");
        assert_eq!(sanitize_id("a/b\\c"), "a_b_c");
    }

    #[test]
    fn sanitize_id_never_yields_traversal_components() {
        assert_eq!(sanitize_id(".."), "_");
        assert_eq!(sanitize_id("."), "_");
        assert_eq!(sanitize_id(""), "_");
        assert_eq!(sanitize_id("//"), "__");
    }
}
