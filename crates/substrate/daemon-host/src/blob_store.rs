//! A file-backed, content-addressed **blob store** - the node content store of
//! `daemon-content-transfer-spec.md` (Phase 1). Immutable bytes are stored by their SHA-256
//! `ContentHash` as `<root>/<sha256-hex>.bin`, written **write-if-absent** so identical content
//! dedupes. It is the anonymous, content-keyed sibling of [`FileRevisionLog`](crate::revision)'s
//! per-artifact blob pool (same digest/dedup idiom, no crypto stack pulled in here).
//!
//! Lifecycle (refcount/pin/sweep/quota) and fetch scoping are deferred to later phases; this layer
//! is a put/get/has/stat store with a coarse per-blob size cap and full-read integrity verification.

use std::path::PathBuf;

use daemon_common::{BlobRef, ByteRange, ContentHash};
use sha2::{Digest, Sha256};

/// Coarse per-blob size cap (a safety bound; a per-node store quota is deferred to a later phase).
pub const MAX_BLOB_SIZE: u64 = 256 * 1024 * 1024;

/// Why a blob operation failed.
#[derive(Debug, thiserror::Error)]
pub enum BlobError {
    /// The blob exceeds [`MAX_BLOB_SIZE`].
    #[error("blob too large: {0} bytes exceeds the {1}-byte limit")]
    TooLarge(u64, u64),
    /// No blob with that hash is stored.
    #[error("blob not found: {0}")]
    NotFound(String),
    /// A full read did not hash to the requested `ContentHash` (corruption / tamper).
    #[error("integrity check failed for blob {0}")]
    Integrity(String),
    /// A ranged read fell outside the blob's bounds.
    #[error("range [{0}, {1}) out of bounds (blob len {2})")]
    Range(u64, u64, u64),
    /// An underlying filesystem error.
    #[error("io: {0}")]
    Io(String),
}

/// The content store: put/get/has/stat immutable blobs by content hash.
#[async_trait::async_trait]
pub trait BlobStore: Send + Sync {
    /// Store `bytes`, returning a [`BlobRef`] (hash + size). Write-if-absent: identical content is
    /// stored once. Rejects blobs over [`MAX_BLOB_SIZE`].
    async fn put(&self, bytes: &[u8]) -> Result<BlobRef, BlobError>;

    /// Read a blob by hash. A full read (`range == None`) is verified against `hash`; a ranged read
    /// returns the slice **unverified** (it cannot be checked against the whole-content hash).
    async fn get(&self, hash: &ContentHash, range: Option<ByteRange>)
        -> Result<Vec<u8>, BlobError>;

    /// Whether a blob with `hash` is present.
    async fn has(&self, hash: &ContentHash) -> bool;

    /// The size of a stored blob, or `None` if absent.
    async fn stat(&self, hash: &ContentHash) -> Option<u64>;
}

/// A blob store rooted at `<root>/<sha256-hex>.bin` (created on demand).
pub struct FileBlobStore {
    root: PathBuf,
}

impl FileBlobStore {
    /// Open (creating the directory) a file-backed blob store rooted at `root`.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, BlobError> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(io)?;
        Ok(Self { root })
    }

    fn blob_path(&self, hash: &ContentHash) -> PathBuf {
        self.root.join(format!("{}.bin", hash.to_hex()))
    }
}

#[async_trait::async_trait]
impl BlobStore for FileBlobStore {
    async fn put(&self, bytes: &[u8]) -> Result<BlobRef, BlobError> {
        let len = bytes.len() as u64;
        if len > MAX_BLOB_SIZE {
            return Err(BlobError::TooLarge(len, MAX_BLOB_SIZE));
        }
        let hash = ContentHash::new(sha256(bytes));
        let path = self.blob_path(&hash);
        // Write-if-absent: identical content dedupes (mirrors FileRevisionLog).
        if !path.exists() {
            tokio::fs::write(&path, bytes).await.map_err(io)?;
        }
        Ok(BlobRef::new(hash, len))
    }

    async fn get(
        &self,
        hash: &ContentHash,
        range: Option<ByteRange>,
    ) -> Result<Vec<u8>, BlobError> {
        let path = self.blob_path(hash);
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(BlobError::NotFound(hash.to_hex()));
            }
            Err(e) => return Err(io(e)),
        };
        match range {
            None => {
                // Full read: verify the bytes still hash to the requested id.
                if ContentHash::new(sha256(&bytes)) != *hash {
                    return Err(BlobError::Integrity(hash.to_hex()));
                }
                Ok(bytes)
            }
            Some(r) => {
                let start = r.offset as usize;
                let end = (r.offset.saturating_add(r.len)) as usize;
                if start > bytes.len() || end > bytes.len() {
                    return Err(BlobError::Range(
                        r.offset,
                        r.offset + r.len,
                        bytes.len() as u64,
                    ));
                }
                Ok(bytes[start..end].to_vec())
            }
        }
    }

    async fn has(&self, hash: &ContentHash) -> bool {
        self.blob_path(hash).exists()
    }

    async fn stat(&self, hash: &ContentHash) -> Option<u64> {
        tokio::fs::metadata(self.blob_path(hash))
            .await
            .ok()
            .map(|m| m.len())
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

fn io(e: std::io::Error) -> BlobError {
    BlobError::Io(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("daemon-blobs-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    #[tokio::test]
    async fn put_get_round_trip_and_dedup() {
        let root = temp_root("rt");
        let store = FileBlobStore::open(&root).unwrap();

        let a = store.put(b"hello world").await.unwrap();
        assert_eq!(a.size, 11);
        assert!(store.has(&a.hash).await);
        assert_eq!(store.stat(&a.hash).await, Some(11));
        assert_eq!(store.get(&a.hash, None).await.unwrap(), b"hello world");

        // Dedup: the same bytes yield the same hash and a single on-disk file.
        let b = store.put(b"hello world").await.unwrap();
        assert_eq!(a.hash, b.hash);
        let count = std::fs::read_dir(&root).unwrap().count();
        assert_eq!(count, 1, "identical content must dedupe to one blob");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn ranged_read_and_missing_and_too_large() {
        let root = temp_root("range");
        let store = FileBlobStore::open(&root).unwrap();
        let r = store.put(b"0123456789").await.unwrap();

        // Ranged read returns the slice.
        let mid = store
            .get(&r.hash, Some(ByteRange { offset: 2, len: 3 }))
            .await
            .unwrap();
        assert_eq!(mid, b"234");
        // Out-of-bounds range errors.
        assert!(matches!(
            store
                .get(&r.hash, Some(ByteRange { offset: 8, len: 5 }))
                .await,
            Err(BlobError::Range(..))
        ));
        // Missing hash.
        let missing = ContentHash::new([0u8; 32]);
        assert!(!store.has(&missing).await);
        assert!(matches!(
            store.get(&missing, None).await,
            Err(BlobError::NotFound(_))
        ));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn full_read_detects_corruption() {
        let root = temp_root("integrity");
        let store = FileBlobStore::open(&root).unwrap();
        let r = store.put(b"trustworthy").await.unwrap();

        // Tamper with the stored file behind the store's back.
        let path = root.join(format!("{}.bin", r.hash.to_hex()));
        std::fs::write(&path, b"tampered!!!").unwrap();

        assert!(matches!(
            store.get(&r.hash, None).await,
            Err(BlobError::Integrity(_))
        ));

        let _ = std::fs::remove_dir_all(&root);
    }
}
