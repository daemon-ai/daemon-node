// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! A file-backed [`RevisionLog`]: the append-only, content-addressed version history backing the
//! node's profile + skill versioning surface.
//!
//! Layout under `<root>/<kind>/<id>/`:
//! - `index.jsonl` — one JSON line per revision (`seq`, `parent`, `hash`, `author`, `reason`,
//!   `ts_ms`), appended on every mutation; the head is the last line.
//! - `blobs/<hex>.bin` — the content-addressed snapshot blob (SHA-256 of the bytes), written
//!   write-if-absent so identical content across revisions dedupes.
//!
//! Reverting is non-destructive: the caller re-appends an older revision's blob as a new head
//! (`daemon-host` `node_api`), so `index.jsonl` only ever grows and roll-forward is reverting to a
//! later `seq`. Mirrors the durable journal's append-only + content-hash idiom without pulling the
//! crypto stack into this path (a plain SHA-256 digest is enough for dedupe/integrity here).

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_common::{Author, ContentHash, Revision, RevisionError, RevisionKind, RevisionLog};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A file-backed revision log rooted at `dir` (created on demand). One subtree per `(kind, id)`.
pub struct FileRevisionLog {
    dir: PathBuf,
    /// Serializes appends so `seq`/`parent` stay consistent across threads.
    lock: RwLock<()>,
}

/// The on-disk JSON shape of one `index.jsonl` line (hash stored as lowercase hex).
#[derive(Serialize, Deserialize)]
struct IndexLine {
    seq: u64,
    parent: Option<u64>,
    hash: String,
    author: Author,
    reason: String,
    ts_ms: u64,
}

impl FileRevisionLog {
    /// Open (creating the directory) a file-backed revision log rooted at `dir`.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, RevisionError> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(io)?;
        Ok(Self {
            dir,
            lock: RwLock::new(()),
        })
    }

    fn artifact_dir(&self, kind: RevisionKind, id: &str) -> PathBuf {
        self.dir.join(kind.as_str()).join(sanitize(id))
    }

    fn index_path(&self, kind: RevisionKind, id: &str) -> PathBuf {
        self.artifact_dir(kind, id).join("index.jsonl")
    }

    fn blob_path(&self, kind: RevisionKind, id: &str, hex: &str) -> PathBuf {
        self.artifact_dir(kind, id)
            .join("blobs")
            .join(format!("{hex}.bin"))
    }

    fn read_index(&self, kind: RevisionKind, id: &str) -> Result<Vec<IndexLine>, RevisionError> {
        let path = self.index_path(kind, id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let body = fs::read_to_string(&path).map_err(io)?;
        let mut out = Vec::new();
        for line in body.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let entry: IndexLine =
                serde_json::from_str(line).map_err(|e| RevisionError::Codec(e.to_string()))?;
            out.push(entry);
        }
        Ok(out)
    }
}

impl RevisionLog for FileRevisionLog {
    fn append(
        &self,
        kind: RevisionKind,
        id: &str,
        blob: &[u8],
        author: Author,
        reason: &str,
    ) -> Result<Revision, RevisionError> {
        let _g = self.lock.write().unwrap();
        let existing = self.read_index(kind, id)?;
        let parent = existing.last().map(|l| l.seq);
        let seq = parent.unwrap_or(0) + 1;
        let ts_ms = now_ms();

        let hash_bytes = sha256(blob);
        let hex = to_hex(&hash_bytes);

        let adir = self.artifact_dir(kind, id);
        fs::create_dir_all(adir.join("blobs")).map_err(io)?;
        let blob_path = self.blob_path(kind, id, &hex);
        if !blob_path.exists() {
            fs::write(&blob_path, blob).map_err(io)?;
        }

        let line = IndexLine {
            seq,
            parent,
            hash: hex,
            author: author.clone(),
            reason: reason.to_string(),
            ts_ms,
        };
        let encoded =
            serde_json::to_string(&line).map_err(|e| RevisionError::Codec(e.to_string()))?;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.index_path(kind, id))
            .map_err(io)?;
        writeln!(f, "{encoded}").map_err(io)?;

        Ok(Revision {
            seq,
            parent,
            content_hash: ContentHash::new(hash_bytes),
            author,
            reason: reason.to_string(),
            ts_ms,
        })
    }

    fn history(&self, kind: RevisionKind, id: &str) -> Result<Vec<Revision>, RevisionError> {
        let _g = self.lock.read().unwrap();
        Ok(self
            .read_index(kind, id)?
            .into_iter()
            .map(|l| revision_of(&l))
            .collect())
    }

    fn get_at(&self, kind: RevisionKind, id: &str, seq: u64) -> Result<Vec<u8>, RevisionError> {
        let _g = self.lock.read().unwrap();
        let index = self.read_index(kind, id)?;
        let line = index
            .iter()
            .find(|l| l.seq == seq)
            .ok_or_else(|| RevisionError::NotFound {
                kind: kind.as_str().to_string(),
                id: id.to_string(),
                seq,
            })?;
        fs::read(self.blob_path(kind, id, &line.hash)).map_err(io)
    }

    fn head(&self, kind: RevisionKind, id: &str) -> Result<Option<Revision>, RevisionError> {
        let _g = self.lock.read().unwrap();
        Ok(self.read_index(kind, id)?.last().map(revision_of))
    }
}

fn revision_of(l: &IndexLine) -> Revision {
    Revision {
        seq: l.seq,
        parent: l.parent,
        content_hash: ContentHash::new(from_hex(&l.hash)),
        author: l.author.clone(),
        reason: l.reason.clone(),
        ts_ms: l.ts_ms,
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

fn to_hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn from_hex(hex: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let j = i * 2;
        if let Some(byte) = hex
            .get(j..j + 2)
            .and_then(|p| u8::from_str_radix(p, 16).ok())
        {
            *slot = byte;
        }
    }
    out
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn io(e: std::io::Error) -> RevisionError {
    RevisionError::Io(e.to_string())
}

/// Restrict an artifact id to a filename-safe slug (mirrors the profile-store sanitizer).
fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_history_get_head_roundtrip() {
        let dir = std::env::temp_dir().join(format!("daemon-revlog-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let log = FileRevisionLog::open(&dir).unwrap();

        let r1 = log
            .append(
                RevisionKind::Profile,
                "opus",
                b"v1",
                Author::Operator,
                "create",
            )
            .unwrap();
        assert_eq!(r1.seq, 1);
        assert_eq!(r1.parent, None);
        let r2 = log
            .append(
                RevisionKind::Profile,
                "opus",
                b"v2",
                Author::Agent("skill_manage".into()),
                "update",
            )
            .unwrap();
        assert_eq!(r2.seq, 2);
        assert_eq!(r2.parent, Some(1));

        let hist = log.history(RevisionKind::Profile, "opus").unwrap();
        assert_eq!(hist.len(), 2);
        assert_eq!(log.get_at(RevisionKind::Profile, "opus", 1).unwrap(), b"v1");
        assert_eq!(
            log.head(RevisionKind::Profile, "opus")
                .unwrap()
                .unwrap()
                .seq,
            2
        );

        // Non-destructive revert: re-append v1's content as a new head; history still grows.
        let blob = log.get_at(RevisionKind::Profile, "opus", 1).unwrap();
        let r3 = log
            .append(
                RevisionKind::Profile,
                "opus",
                &blob,
                Author::Operator,
                "revert to 1",
            )
            .unwrap();
        assert_eq!(r3.seq, 3);
        assert_eq!(log.get_at(RevisionKind::Profile, "opus", 3).unwrap(), b"v1");
        // Roll-forward: revert to the later seq again.
        let blob2 = log.get_at(RevisionKind::Profile, "opus", 2).unwrap();
        assert_eq!(blob2, b"v2");

        // Survives reopen (durable).
        let reopened = FileRevisionLog::open(&dir).unwrap();
        assert_eq!(
            reopened
                .history(RevisionKind::Profile, "opus")
                .unwrap()
                .len(),
            3
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn distinct_kinds_and_ids_are_isolated() {
        let dir = std::env::temp_dir().join(format!("daemon-revlog-iso-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let log = FileRevisionLog::open(&dir).unwrap();
        log.append(RevisionKind::Profile, "a", b"pa", Author::Operator, "c")
            .unwrap();
        log.append(RevisionKind::Skill, "a", b"sa", Author::Operator, "c")
            .unwrap();
        assert_eq!(log.history(RevisionKind::Profile, "a").unwrap().len(), 1);
        assert_eq!(log.history(RevisionKind::Skill, "a").unwrap().len(), 1);
        assert!(log.head(RevisionKind::Profile, "b").unwrap().is_none());
        let _ = fs::remove_dir_all(&dir);
    }
}
