// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`ContentCache`] — a blake3-keyed, on-disk, size-bounded content cache (P3 lane S, spec §8/§10.6).
//!
//! The P2 gate **pre-staged** the experiment `.wasm` (and used a synthetic corpus) onto every fleet
//! box. Lane S removes that: the worker fetches the module + its assigned corpus shards **by content
//! hash** from the payload store and caches them here, so a box warmed once never re-downloads (a GB
//! artifact is fetched exactly once per content hash, across runs and process restarts).
//!
//! Unlike the in-memory [`ArtifactCache`](crate::ArtifactCache) (RUN-4, bounds resolved bytes within
//! one process), this cache **persists to disk** — the fleet-staging property. It is the on-disk half
//! of the same §10.6 `[swarm].data_cache_gb` budget.
//!
//! ## Layout
//!
//! One file per content hash: `<root>/objects/<hex-blake3>`. The hash IS the file name, so the cache
//! is content-addressed end to end — a module built once is stored once and shared by every run that
//! pins it. Writes are atomic (`.tmp-<hex>` → rename) so a crashed fetch never leaves a truncated
//! object that a later reader would mistake for the real one.
//!
//! ## Verification (never a silent bad artifact)
//!
//! Every read blake3-verifies the file against its key: a corrupt/tampered cache file is treated as a
//! miss **and evicted**, so it cannot smuggle a bad module past the §6.5 assess / §12 tamper reject.
//! [`ContentCache::insert`] likewise refuses bytes whose blake3 does not equal the declared key.
//!
//! ## Eviction policy (documented)
//!
//! **Size-bounded LRU by write time.** The cache holds at most `max_bytes`. On [`insert`], if the new
//! object would exceed the budget, the oldest-`mtime` objects are removed until it fits (content is
//! immutable, so write time == first-fetch time — evicting oldest-written evicts least-recently-first-
//! -fetched). An object larger than the entire budget is returned to the caller but **not** cached
//! (never evict everything for one giant object). `mtime` is the recency signal because the contained
//! fs surface exposes it uniformly and content-addressed files are written once and never rewritten
//! (a `get` does not refresh recency — there is no `utimes` on the contained root, and refetching a
//! wrongly-evicted object is cheap + still content-verified).
//!
//! [`insert`]: ContentCache::insert

use std::io;
use std::path::{Path, PathBuf};

use daemon_core::ContainedRoot;
use daemon_vhc_proto::{blake3_hash, Hash};

use crate::SwarmNetError;

/// The subdirectory under the cache root that holds the content-addressed object files.
const OBJECTS_DIR: &str = "objects";

/// A blake3-keyed, size-bounded, on-disk content cache (spec §8/§10.6).
#[derive(Clone)]
pub struct ContentCache {
    root: ContainedRoot,
    max_bytes: u64,
}

impl std::fmt::Debug for ContentCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentCache")
            .field("root", &self.root.root())
            .field("max_bytes", &self.max_bytes)
            .finish()
    }
}

impl ContentCache {
    /// Open (creating if missing) a cache rooted at `dir`, bounded to `max_bytes` total object bytes.
    pub fn open(dir: &Path, max_bytes: u64) -> Result<Self, SwarmNetError> {
        let root = ContainedRoot::open(dir)
            .map_err(|e| SwarmNetError::Transport(format!("open content cache root: {e}")))?;
        // The objects subdir is created lazily on the first insert; `open` only fixes the boundary.
        Ok(Self { root, max_bytes })
    }

    /// A cache bounded by `data_cache_gb` gibibytes (the §10.6 `[swarm].data_cache_gb` knob).
    pub fn open_gb(dir: &Path, data_cache_gb: u32) -> Result<Self, SwarmNetError> {
        Self::open(dir, u64::from(data_cache_gb) * (1 << 30))
    }

    /// The byte budget.
    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// The relative path of the object file for `hash`.
    fn object_rel(hash: &Hash) -> PathBuf {
        Path::new(OBJECTS_DIR).join(hash.to_hex())
    }

    /// Whether `hash` is present on disk (does not verify the bytes; use [`ContentCache::get`] to
    /// fetch verified bytes).
    pub async fn contains(&self, hash: &Hash) -> bool {
        matches!(
            self.root.symlink_metadata(&Self::object_rel(hash)).await,
            Ok(meta) if meta.is_file
        )
    }

    /// Read `hash`'s bytes from the cache, blake3-verifying them against the key. Returns `Ok(None)`
    /// on a miss; a **corrupt** cache file (bytes do not hash to `hash`) is evicted and reported as a
    /// miss (so a poisoned entry never satisfies a fetch).
    pub async fn get(&self, hash: &Hash) -> Result<Option<Vec<u8>>, SwarmNetError> {
        let rel = Self::object_rel(hash);
        let bytes = match self.root.read(&rel).await {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(SwarmNetError::Transport(format!("read cache object: {e}"))),
        };
        if &blake3_hash(&bytes) != hash {
            // Poisoned entry: drop it and report a miss so the caller re-fetches from the store.
            let _ = self.root.remove_file(&rel).await;
            return Ok(None);
        }
        Ok(Some(bytes))
    }

    /// Insert `bytes` under `hash`, evicting least-recently-written objects until it fits.
    ///
    /// Refuses bytes whose blake3 is not `hash` ([`SwarmNetError::HashMismatch`] — a programming/
    /// integrity error, never stored). An object larger than the whole budget is a no-op (not cached).
    /// The write is atomic (tmp + rename), so a concurrent reader never sees a partial object.
    pub async fn insert(&self, hash: &Hash, bytes: &[u8]) -> Result<(), SwarmNetError> {
        let actual = blake3_hash(bytes);
        if &actual != hash {
            return Err(SwarmNetError::HashMismatch {
                expected: hash.to_hex(),
                actual: actual.to_hex(),
            });
        }
        let len = bytes.len() as u64;
        if len > self.max_bytes {
            return Ok(()); // too big to ever cache — leave uncached rather than evict everything.
        }
        self.root
            .create_dir_all(Path::new(OBJECTS_DIR))
            .await
            .map_err(|e| SwarmNetError::Transport(format!("create cache objects dir: {e}")))?;
        self.evict_to_fit(len).await?;

        // Atomic publish: write to a per-hash temp name, then rename over the final name.
        let hex = hash.to_hex();
        let tmp = Path::new(OBJECTS_DIR).join(format!(".tmp-{hex}"));
        let final_rel = Path::new(OBJECTS_DIR).join(&hex);
        self.root
            .write(&tmp, bytes)
            .await
            .map_err(|e| SwarmNetError::Transport(format!("write cache tmp: {e}")))?;
        self.root
            .rename(&tmp, &final_rel)
            .await
            .map_err(|e| SwarmNetError::Transport(format!("publish cache object: {e}")))?;
        Ok(())
    }

    /// The (size, mtime_ms) of every cached object file, plus the running total.
    async fn scan(&self) -> Result<(u64, Vec<(String, u64, u64)>), SwarmNetError> {
        let entries = match self.root.read_dir(Path::new(OBJECTS_DIR)).await {
            Ok(e) => e,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((0, Vec::new())),
            Err(e) => return Err(SwarmNetError::Transport(format!("scan cache: {e}"))),
        };
        let mut used = 0u64;
        let mut objects = Vec::new();
        for entry in entries {
            // Ignore in-flight temp files (`.tmp-…`) and any stray directory.
            if !entry.meta.is_file || entry.name.starts_with(".tmp-") {
                continue;
            }
            used += entry.meta.size;
            objects.push((entry.name, entry.meta.size, entry.meta.mtime_ms));
        }
        Ok((used, objects))
    }

    /// Evict oldest-`mtime` objects until `incoming` more bytes fit under the budget.
    async fn evict_to_fit(&self, incoming: u64) -> Result<(), SwarmNetError> {
        let (mut used, mut objects) = self.scan().await?;
        if used + incoming <= self.max_bytes {
            return Ok(());
        }
        // Oldest first (smallest mtime). Ties broken by name for determinism.
        objects.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
        for (name, size, _mtime) in objects {
            if used + incoming <= self.max_bytes {
                break;
            }
            match self
                .root
                .remove_file(&Path::new(OBJECTS_DIR).join(&name))
                .await
            {
                Ok(()) => used = used.saturating_sub(size),
                // A racing reader/evictor already removed it — treat as freed.
                Err(e) if e.kind() == io::ErrorKind::NotFound => used = used.saturating_sub(size),
                Err(e) => return Err(SwarmNetError::Transport(format!("evict cache object: {e}"))),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_root;

    #[tokio::test]
    async fn insert_get_round_trips_and_verifies() {
        let dir = temp_root("content-cache-roundtrip");
        let cache = ContentCache::open(dir.path(), 1 << 20).unwrap();
        let bytes = b"experiment-wasm-bytes".to_vec();
        let hash = blake3_hash(&bytes);

        assert!(!cache.contains(&hash).await);
        assert_eq!(cache.get(&hash).await.unwrap(), None);

        cache.insert(&hash, &bytes).await.unwrap();
        assert!(cache.contains(&hash).await);
        assert_eq!(cache.get(&hash).await.unwrap(), Some(bytes));
    }

    #[tokio::test]
    async fn insert_rejects_wrong_key() {
        let dir = temp_root("content-cache-wrongkey");
        let cache = ContentCache::open(dir.path(), 1 << 20).unwrap();
        // Claim a key the bytes do not hash to.
        let err = cache
            .insert(&blake3_hash(b"the-key"), b"different-bytes")
            .await
            .unwrap_err();
        assert!(
            matches!(err, SwarmNetError::HashMismatch { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn corrupt_entry_is_a_miss_and_evicted() {
        let dir = temp_root("content-cache-corrupt");
        let cache = ContentCache::open(dir.path(), 1 << 20).unwrap();
        let bytes = b"honest".to_vec();
        let hash = blake3_hash(&bytes);
        cache.insert(&hash, &bytes).await.unwrap();

        // Corrupt the on-disk file behind the cache's back (a bit-rot / tamper scenario).
        cache
            .root
            .write(&ContentCache::object_rel(&hash), b"tampered")
            .await
            .unwrap();
        // The verifying read reports a miss and drops the poisoned entry.
        assert_eq!(cache.get(&hash).await.unwrap(), None);
        assert!(!cache.contains(&hash).await, "poisoned entry evicted");
    }

    #[tokio::test]
    async fn lru_evicts_oldest_to_fit_budget() {
        // Budget = 3 objects of 100 bytes.
        let dir = temp_root("content-cache-lru");
        let cache = ContentCache::open(dir.path(), 300).unwrap();
        let mk = |seed: u8| {
            let bytes = vec![seed; 100];
            let hash = blake3_hash(&bytes);
            (hash, bytes)
        };
        let (ha, ba) = mk(0xA);
        let (hb, bb) = mk(0xB);
        let (hc, bc) = mk(0xC);
        let (hd, bd) = mk(0xD);

        // Insert a, b, c with strictly increasing mtimes (sleep past the ms resolution).
        cache.insert(&ha, &ba).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        cache.insert(&hb, &bb).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        cache.insert(&hc, &bc).await.unwrap();
        assert!(
            cache.contains(&ha).await && cache.contains(&hb).await && cache.contains(&hc).await
        );

        // Inserting d must evict the oldest (a), keeping b and c.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        cache.insert(&hd, &bd).await.unwrap();
        assert!(!cache.contains(&ha).await, "oldest evicted");
        assert!(cache.contains(&hb).await);
        assert!(cache.contains(&hc).await);
        assert!(cache.contains(&hd).await);
    }

    #[tokio::test]
    async fn oversize_object_is_not_cached() {
        let dir = temp_root("content-cache-oversize");
        let cache = ContentCache::open(dir.path(), 100).unwrap();
        let bytes = vec![0u8; 200];
        let hash = blake3_hash(&bytes);
        cache.insert(&hash, &bytes).await.unwrap();
        assert!(
            !cache.contains(&hash).await,
            "object larger than budget is not cached"
        );
    }

    /// The end-to-end fetch-by-hash distribution path (P3 lane S): a module published at the
    /// content-addressed key `modules/<blake3>.wasm` is fetched via a presigned GET + blake3-verified
    /// by [`ArtifactResolver`], cached here, and served from cache on the second fetch — even after
    /// the object is evicted server-side (the fleet-warm property).
    #[tokio::test]
    async fn fetch_by_hash_then_serve_from_cache() {
        use crate::mock_r2::MockR2;
        use crate::{ArtifactRef, ArtifactResolver, RunId};
        use std::sync::Arc;

        let mock = MockR2::start().await;
        let module = b"experiment-160m-wasm-bytes".to_vec();
        let hash = blake3_hash(&module);
        let key_path = format!("modules/{}.wasm", hash.to_hex());
        // The artifact key layout is `runs/<run>/<path>` (spec §11.3 / cloud keys.ts).
        mock.seed(&format!("runs/run-x/{key_path}"), module.clone());

        let resolver = ArtifactResolver::with_egress(mock.egress())
            .with_presign(Arc::new(mock.presign_client()), RunId::new("run-x"));
        let art = ArtifactRef::new(format!("r2://{key_path}"), hash);

        let dir = temp_root("content-cache-fetch");
        let cache = ContentCache::open(dir.path(), 1 << 30).unwrap();

        // Miss → fetch from the store (presigned GET, blake3-verified) → cache.
        assert_eq!(cache.get(&hash).await.unwrap(), None);
        let fetched = resolver.fetch(&art).await.unwrap();
        assert_eq!(fetched, module);
        cache.insert(&hash, &fetched).await.unwrap();

        // Evict the object server-side; the warmed cache still serves it (no re-download).
        mock.evict(&format!("runs/run-x/{key_path}"));
        assert!(
            resolver.fetch(&art).await.is_err(),
            "object is gone server-side"
        );
        assert_eq!(cache.get(&hash).await.unwrap(), Some(module));
    }

    #[tokio::test]
    async fn persists_across_reopen() {
        let dir = temp_root("content-cache-persist");
        let bytes = b"survives-restart".to_vec();
        let hash = blake3_hash(&bytes);
        {
            let cache = ContentCache::open(dir.path(), 1 << 20).unwrap();
            cache.insert(&hash, &bytes).await.unwrap();
        }
        // A fresh handle (a new process on a warmed fleet box) still serves the object.
        let reopened = ContentCache::open(dir.path(), 1 << 20).unwrap();
        assert_eq!(reopened.get(&hash).await.unwrap(), Some(bytes));
    }
}
