// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Production [`ContentStore`] implementations — the **content-addressed** payload plane the role
//! session services module `payload_put`/`payload_get` against (ABI §12.6 [RS-4]).
//!
//! The module names CONTENT (`payload/<blake3>`), the store moves bytes, and the run pump
//! re-verifies every fetched object against the requested hash before delivery — the store is
//! untrusted by construction. No run/round/peer coordinate ever appears in the key the module
//! sees (that vocabulary is round-coupled and survives only in the harness-era
//! [`PayloadStore`](crate::transport::PayloadStore) consumers).
//!
//! Two production planes:
//! * [`FsContentStore`] — the filesystem plane, rooted in the run's state directory
//!   (`<run state dir>/payload/`); the local / single-host / acceptance-baseline store.
//! * [`R2ContentStore`](crate::r2_store::R2ContentStore) — the presigned R2/S3 plane (lives with
//!   the other presign machinery in [`crate::r2_store`]).

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_core::ContainedRoot;
use daemon_vhc_proto::blake3_hash;

use crate::seam::ContentHash;
use crate::transport::ContentStore;
use crate::VhcNetError;

/// A filesystem [`ContentStore`]: objects live flat under a [`ContainedRoot`] named by their
/// blake3 hex (`<root>/<hex>`), so nothing attacker-influenced ever shapes a path (the hex is
/// derived from the bytes, never taken from a peer).
///
/// Writes are idempotent by construction (same bytes ⇒ same address ⇒ same file content); an
/// in-process RwLock linearizes same-address concurrent writers against readers exactly like
/// [`crate::store::FsPayloadStore`] (ContainedRoot exposes no rename, so a plain truncate+write
/// could otherwise expose a torn object to a concurrent reader).
#[derive(Clone)]
pub struct FsContentStore {
    root: ContainedRoot,
    lock: Arc<tokio::sync::RwLock<()>>,
}

impl FsContentStore {
    /// Open a store rooted at `root` (created if missing).
    ///
    /// # Errors
    /// [`VhcNetError::Transport`] if the root cannot be opened/created.
    pub fn open(root: &Path) -> Result<Self, VhcNetError> {
        let root = ContainedRoot::open(root)
            .map_err(|e| VhcNetError::Transport(format!("open content store root: {e}")))?;
        Ok(Self {
            root,
            lock: Arc::new(tokio::sync::RwLock::new(())),
        })
    }

    /// The relative object path of one content hash.
    fn object_rel(hash: &ContentHash) -> String {
        hash.to_hex()
    }
}

#[async_trait]
impl ContentStore for FsContentStore {
    async fn put_content(&self, bytes: &[u8]) -> Result<ContentHash, VhcNetError> {
        let hash = blake3_hash(bytes);
        let _w = self.lock.write().await;
        self.root
            .write(Path::new(&Self::object_rel(&hash)), bytes)
            .await
            .map_err(|e| VhcNetError::Transport(format!("write content object: {e}")))?;
        Ok(hash)
    }

    async fn get_content(&self, hash: &ContentHash) -> Result<Vec<u8>, VhcNetError> {
        let _r = self.lock.read().await;
        let bytes = match self.root.read(Path::new(&Self::object_rel(hash))).await {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(VhcNetError::PayloadMiss(hash.to_hex()))
            }
            Err(e) => return Err(VhcNetError::Transport(format!("read content object: {e}"))),
        };
        // Defense in depth: the caller's pump re-verifies regardless, but a store that returns
        // bytes not matching their own address is broken and says so typed.
        let actual = blake3_hash(&bytes);
        if &actual != hash {
            return Err(VhcNetError::HashMismatch {
                expected: hash.to_hex(),
                actual: actual.to_hex(),
            });
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_root;

    #[tokio::test]
    async fn put_get_round_trips_by_content_address() {
        let dir = temp_root("content-fs-roundtrip");
        let store = FsContentStore::open(dir.path()).unwrap();

        let hash = store.put_content(b"sealed-object").await.unwrap();
        assert_eq!(hash, blake3_hash(b"sealed-object"));
        assert_eq!(store.get_content(&hash).await.unwrap(), b"sealed-object");

        // Idempotent re-put: same bytes, same address, no error.
        assert_eq!(store.put_content(b"sealed-object").await.unwrap(), hash);
    }

    #[tokio::test]
    async fn missing_object_is_a_typed_miss() {
        let dir = temp_root("content-fs-miss");
        let store = FsContentStore::open(dir.path()).unwrap();
        let absent = blake3_hash(b"never-stored");
        let err = store.get_content(&absent).await.unwrap_err();
        assert!(matches!(err, VhcNetError::PayloadMiss(_)), "got {err:?}");
    }

    #[tokio::test]
    // Test-only in-place corruption of a fixture file inside the test's own temp root — not a
    // production fs path.
    #[allow(clippy::disallowed_methods)]
    async fn tampered_object_is_a_typed_hash_mismatch() {
        let dir = temp_root("content-fs-tamper");
        let store = FsContentStore::open(dir.path()).unwrap();
        let hash = store.put_content(b"honest").await.unwrap();

        // Corrupt the object in place (the store root is plain files).
        let path = dir.path().join(hash.to_hex());
        std::fs::write(&path, b"tampered").unwrap();

        let err = store.get_content(&hash).await.unwrap_err();
        assert!(
            matches!(err, VhcNetError::HashMismatch { .. }),
            "got {err:?}"
        );
    }
}
