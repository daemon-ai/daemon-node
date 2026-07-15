// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`FsPayloadStore`] — a filesystem [`PayloadStore`] rooted at a directory.
//!
//! The local-mode stand-in for the `r2` / `iroh-blobs` payload planes (§7.1): objects live under a
//! `<run>/<round>/<peer>.bin` layout beneath a [`ContainedRoot`], so peer-supplied key components
//! (run id, peer pubkey) can never escape the store root (openat2 RESOLVE_BENEATH|NO_SYMLINKS —
//! the workspace-wide fs ban points here).
//!
//! Retention (§7.4, NET-8): [`FsPayloadStore::prune`] deletes objects older than
//! `payload_retention_rounds`; a subsequent `get`/`head` of a pruned object is a typed
//! [`SwarmNetError::PayloadMiss`], which the §6.4 stall ladder consumes. The `head` (`stat`) result
//! is what the [`ReceiptProducer`](crate::receipt::ReceiptProducer) turns into a signed
//! `StorageReceipt` (§6.4 I6).

use std::io;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use daemon_core::ContainedRoot;
use daemon_swarm_proto::blake3_hash;

use crate::seam::{ContentHash, PayloadKey, RoundId, RunId};
use crate::transport::{PayloadStat, PayloadStore};
use crate::SwarmNetError;

/// A filesystem payload plane with a bounded retention window.
#[derive(Clone)]
pub struct FsPayloadStore {
    root: ContainedRoot,
    retention_rounds: u64,
}

impl FsPayloadStore {
    /// Open a store rooted at `root`, keeping objects for `retention_rounds` rounds. The root is
    /// created if missing.
    pub fn open(root: &Path, retention_rounds: u64) -> Result<Self, SwarmNetError> {
        let root = ContainedRoot::open(root)
            .map_err(|e| SwarmNetError::Transport(format!("open payload store root: {e}")))?;
        Ok(Self {
            root,
            retention_rounds,
        })
    }

    /// The configured retention window (rounds).
    #[must_use]
    pub fn retention_rounds(&self) -> u64 {
        self.retention_rounds
    }

    /// Relative path of one object: `<run>/<round>/<peer>.bin`.
    fn object_rel(key: &PayloadKey) -> PathBuf {
        Self::round_rel(&key.run, key.round).join(format!("{}.bin", key.peer.to_hex()))
    }

    /// Relative path of a run's round directory.
    fn round_rel(run: &RunId, round: RoundId) -> PathBuf {
        PathBuf::from(hex_segment(run.as_str().as_bytes())).join(format!("{round:020}"))
    }

    /// Relative path of a run's directory.
    fn run_rel(run: &RunId) -> PathBuf {
        PathBuf::from(hex_segment(run.as_str().as_bytes()))
    }

    /// Delete every object of `run` whose round has fallen outside the retention window relative to
    /// `current_round` (i.e. `round + retention_rounds < current_round`). Returns the number of
    /// round directories removed. A subsequent fetch of a pruned object is a typed
    /// [`SwarmNetError::PayloadMiss`] (feeds the stall ladder).
    pub async fn prune(&self, run: &RunId, current_round: RoundId) -> Result<u64, SwarmNetError> {
        let run_rel = Self::run_rel(run);
        let entries = match self.root.read_dir(&run_rel).await {
            Ok(entries) => entries,
            // Nothing stored for this run yet — nothing to prune.
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(transport_err("read run dir", &e)),
        };
        let mut removed = 0u64;
        for entry in entries {
            if !entry.meta.is_dir {
                continue;
            }
            let Some(round) = parse_round(&entry.name) else {
                continue;
            };
            if round + self.retention_rounds < current_round {
                self.remove_round_dir(&run_rel.join(&entry.name)).await?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Remove a round directory and all objects beneath it (contained; no symlink traversal).
    async fn remove_round_dir(&self, round_rel: &Path) -> Result<(), SwarmNetError> {
        let children = self
            .root
            .read_dir(round_rel)
            .await
            .map_err(|e| transport_err("read round dir", &e))?;
        for child in children {
            self.root
                .remove_file(&round_rel.join(&child.name))
                .await
                .map_err(|e| transport_err("remove object", &e))?;
        }
        self.root
            .remove_dir(round_rel)
            .await
            .map_err(|e| transport_err("remove round dir", &e))
    }
}

#[async_trait]
impl PayloadStore for FsPayloadStore {
    async fn put(&self, key: &PayloadKey, bytes: &[u8]) -> Result<ContentHash, SwarmNetError> {
        let round_rel = Self::round_rel(&key.run, key.round);
        self.root
            .create_dir_all(&round_rel)
            .await
            .map_err(|e| transport_err("create round dir", &e))?;
        self.root
            .write(&Self::object_rel(key), bytes)
            .await
            .map_err(|e| transport_err("write object", &e))?;
        Ok(blake3_hash(bytes))
    }

    async fn get(
        &self,
        key: &PayloadKey,
        expected: &ContentHash,
    ) -> Result<Vec<u8>, SwarmNetError> {
        let bytes = self
            .root
            .read(&Self::object_rel(key))
            .await
            .map_err(|e| miss_or_err("read object", &e, key))?;
        let actual = blake3_hash(&bytes);
        if &actual != expected {
            return Err(SwarmNetError::HashMismatch {
                expected: expected.to_hex(),
                actual: actual.to_hex(),
            });
        }
        Ok(bytes)
    }

    async fn head(&self, key: &PayloadKey) -> Result<PayloadStat, SwarmNetError> {
        // A local fs store re-reads to attest the content hash; a network HEAD would carry the size
        // from object metadata and the hash from the commitment (the trait allows either).
        let bytes = self
            .root
            .read(&Self::object_rel(key))
            .await
            .map_err(|e| miss_or_err("stat object", &e, key))?;
        Ok(PayloadStat {
            hash: blake3_hash(&bytes),
            size: bytes.len() as u64,
        })
    }
}

/// Hex-encode arbitrary bytes into a single filesystem-safe path segment (so a run id containing
/// `/` or `..` cannot become a nested path or an escape — belt-and-suspenders over `ContainedRoot`).
fn hex_segment(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).expect("nibble"));
        s.push(char::from_digit((b & 0xf) as u32, 16).expect("nibble"));
    }
    s
}

/// Parse a zero-padded round directory name back to a [`RoundId`].
fn parse_round(name: &str) -> Option<RoundId> {
    name.parse().ok()
}

/// Map a `ContainedRoot` io error onto a transport error.
fn transport_err(op: &str, e: &io::Error) -> SwarmNetError {
    SwarmNetError::Transport(format!("{op}: {e}"))
}

/// Map a `ContainedRoot` io error onto a typed miss (NotFound) or a transport error (everything
/// else). A pruned/never-present object surfaces as [`SwarmNetError::PayloadMiss`] for the stall
/// ladder.
fn miss_or_err(op: &str, e: &io::Error, key: &PayloadKey) -> SwarmNetError {
    if e.kind() == io::ErrorKind::NotFound {
        SwarmNetError::PayloadMiss(format!(
            "{}@r{}/{}",
            key.run.as_str(),
            key.round,
            key.peer.to_hex()
        ))
    } else {
        transport_err(op, e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::PeerId;
    use crate::test_support::temp_root;

    fn key(run: &str, round: RoundId, peer: u8) -> PayloadKey {
        PayloadKey::new(RunId::new(run), round, PeerId([peer; 32]))
    }

    #[tokio::test]
    async fn put_get_round_trips() {
        let dir = temp_root("fsstore-roundtrip");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let k = key("run-a", 3, 0x11);

        let hash = store.put(&k, b"update-bytes").await.unwrap();
        assert_eq!(hash, blake3_hash(b"update-bytes"));

        let got = store.get(&k, &hash).await.unwrap();
        assert_eq!(got, b"update-bytes");

        let stat = store.head(&k).await.unwrap();
        assert_eq!(stat.hash, hash);
        assert_eq!(stat.size, b"update-bytes".len() as u64);
    }

    #[tokio::test]
    async fn get_rejects_tampered_hash() {
        let dir = temp_root("fsstore-tamper");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let k = key("run-a", 0, 0x22);
        store.put(&k, b"honest").await.unwrap();

        let wrong = blake3_hash(b"different");
        let err = store.get(&k, &wrong).await.unwrap_err();
        assert!(
            matches!(err, SwarmNetError::HashMismatch { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn missing_object_is_typed_miss() {
        let dir = temp_root("fsstore-missing");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let k = key("run-a", 7, 0x33);
        let err = store.get(&k, &blake3_hash(b"x")).await.unwrap_err();
        assert!(matches!(err, SwarmNetError::PayloadMiss(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn retained_object_is_fetchable() {
        let dir = temp_root("fsstore-retain");
        let store = FsPayloadStore::open(dir.path(), 2).unwrap();
        let k = key("run-a", 4, 0x44);
        let hash = store.put(&k, b"recent").await.unwrap();

        // current=5, retention=2: 4 + 2 = 6 >= 5 -> within window, kept.
        let removed = store.prune(&k.run, 5).await.unwrap();
        assert_eq!(removed, 0);
        assert_eq!(store.get(&k, &hash).await.unwrap(), b"recent");
    }

    #[tokio::test]
    async fn expired_object_is_typed_miss() {
        let dir = temp_root("fsstore-expire");
        let store = FsPayloadStore::open(dir.path(), 2).unwrap();
        let k = key("run-a", 0, 0x55);
        let hash = store.put(&k, b"stale").await.unwrap();

        // current=5, retention=2: 0 + 2 = 2 < 5 -> expired, pruned.
        let removed = store.prune(&k.run, 5).await.unwrap();
        assert_eq!(removed, 1);
        let err = store.get(&k, &hash).await.unwrap_err();
        assert!(matches!(err, SwarmNetError::PayloadMiss(_)), "got {err:?}");
    }
}
