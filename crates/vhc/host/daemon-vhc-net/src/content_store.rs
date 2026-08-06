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
    /// The ambient disk custodian + charge scope (Phase 6): a first-time object reserves its
    /// bytes before it lands; an idempotent re-put of a present object charges nothing.
    custody: Option<(Arc<daemon_vhc_custody::DiskCustodian>, String)>,
}

impl FsContentStore {
    /// Open a store rooted at `root` (created if missing).
    ///
    /// # Errors
    /// [`VhcNetError::Transport`] if the root cannot be opened/created.
    pub fn open(root: &Path) -> Result<Self, VhcNetError> {
        let custody = daemon_vhc_custody::ambient_for(root);
        let root = ContainedRoot::open(root)
            .map_err(|e| VhcNetError::Transport(format!("open content store root: {e}")))?;
        Ok(Self {
            root,
            lock: Arc::new(tokio::sync::RwLock::new(())),
            custody,
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
        let rel = Self::object_rel(&hash);
        let _w = self.lock.write().await;
        // An object already at its address is the idempotent re-put: same bytes by
        // construction, nothing to write, nothing to charge.
        if self
            .root
            .symlink_metadata(Path::new(&rel))
            .await
            .is_ok_and(|m| !m.is_dir)
        {
            return Ok(hash);
        }
        // Reserve before the bytes land (Phase 6): a payload that does not fit refuses typed,
        // never a raw mid-write ENOSPC.
        let reservation = match &self.custody {
            None => None,
            Some((custodian, scope)) => Some(
                custodian
                    .reserve(
                        scope,
                        bytes.len() as u64,
                        daemon_vhc_custody::WriteClass::Normal,
                    )
                    .map_err(|refusal| {
                        VhcNetError::Transport(format!(
                            "content store custody: {}",
                            refusal.to_io()
                        ))
                    })?,
            ),
        };
        self.root
            .write(Path::new(&rel), bytes)
            .await
            .map_err(|e| VhcNetError::Transport(format!("write content object: {e}")))?;
        if let Some(r) = reservation {
            r.commit();
        }
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
        // The store is UNTRUSTED and the requested address is not always the object's plain
        // blake3: a chunk-addressed corpus shard is keyed by its domain-separated CHUNK FOLD
        // (`daemon_vhc_proto::shard_fold`), which never equals `blake3(bytes)`. Verification is
        // the PUMP's job — it re-checks every fetched object against the requested plain hash
        // (payloads/checkpoints/whole artifacts) or the registered covering-chunk hashes
        // (chunk-addressed shards). So this seat serves the keyed bytes verbatim, exactly like
        // the in-process [`MemoryContentStore`]; a plain-hash gate here would wrongly reject every
        // chunk-addressed range fetch.
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
    // Test-only in-place write of a fixture file inside the test's own temp root — not a
    // production fs path.
    #[allow(clippy::disallowed_methods)]
    async fn get_content_serves_keyed_bytes_verbatim() {
        // The content store is UNTRUSTED and its key is not always the object's plain blake3: a
        // chunk-addressed corpus shard is keyed by its domain-separated CHUNK FOLD, which never
        // equals `blake3(bytes)`. So `get_content` serves the keyed bytes verbatim — verification
        // is the PUMP's job (plain hash for payloads/whole artifacts, covering-chunk hashes for
        // chunk-addressed shards). A plain-hash gate here would reject every chunk-addressed range
        // fetch; the pump is the sole arbiter (this is the same behavior as `MemoryContentStore`).
        let dir = temp_root("content-fs-verbatim");
        let store = FsContentStore::open(dir.path()).unwrap();
        let hash = store.put_content(b"honest").await.unwrap();

        // Write different bytes under the SAME key (a chunk-addressed shard is exactly this shape:
        // the key is the fold, not blake3(bytes)). The store returns them verbatim; a downstream
        // pump verifies against the requested hash / registered chunk map.
        let path = dir.path().join(hash.to_hex());
        std::fs::write(&path, b"fold-keyed-bytes").unwrap();
        assert_eq!(
            store.get_content(&hash).await.unwrap(),
            b"fold-keyed-bytes",
            "the untrusted store serves the keyed bytes; the pump verifies"
        );
    }
}
