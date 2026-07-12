// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-net` — the swarm transport.
//!
//! The [`SwarmTransport`](transport) seam (spec §7.1): one control plane
//! ([`ControlPlane`] — publish/subscribe of already-signed message bytes, with the in-process
//! [`LoopbackGossip`] implementation) and one payload plane ([`PayloadStore`] — opaque objects by
//! `(run, round, peer)` key + content hash, with the filesystem [`FsPayloadStore`] implementation
//! and its retention window). The [`ReceiptProducer`] turns store availability into signed
//! `StorageReceipt` evidence (§6.4 I6). Artifact fetch ([`ArtifactResolver`]) resolves `file://`
//! (blake3-verified); `r2`/`hf`/`https` are reserved for the egress plane.
//!
//! Engine-agnostic; consumed by `daemon-swarm-run` (§10.1). Outbound HTTP must route through
//! `daemon_egress::EgressClient` (raw `reqwest::Client` is banned workspace-wide by clippy); no HTTP
//! client is constructed this wave.
//!
//! Merge-1 note: the shared identity/hash vocabulary in [`seam`] is now the canonical
//! `daemon-swarm-proto` types (blake3 `Hash`, `PeerId`); the [`ReceiptProducer`] emits proto's
//! signed `StorageReceipt` control message (ed25519 over canonical CBOR).

#![forbid(unsafe_code)]

pub mod artifact;
pub mod gossip;
pub mod receipt;
pub mod seam;
pub mod store;
pub mod transport;

pub use artifact::{ArtifactRef, ArtifactResolver, ArtifactScheme};
pub use gossip::LoopbackGossip;
pub use receipt::ReceiptProducer;
pub use seam::{ContentHash, PayloadKey, PeerId, RoundId, RunId};
pub use store::FsPayloadStore;
pub use transport::{ControlPlane, ControlSubscription, PayloadStat, PayloadStore};

/// Errors surfaced by the swarm transport.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SwarmNetError {
    /// A control-plane or payload-plane transport step failed.
    #[error("swarm transport error: {0}")]
    Transport(String),
    /// An artifact fetch (`file`, and later `r2` / `hf` / `https`) failed.
    #[error("artifact fetch failed: {0}")]
    Fetch(String),
    /// A content hash did not match the expected digest (payload GET or artifact verify) — the
    /// tamper/corruption reject path (§12).
    #[error("content hash mismatch: expected {expected}, got {actual}")]
    HashMismatch {
        /// The hash the caller expected (hex).
        expected: String,
        /// The hash actually computed (hex).
        actual: String,
    },
    /// A payload object was absent or had fallen outside the retention window — the typed miss the
    /// §6.4 stall ladder consumes (NET-8).
    #[error("payload miss: {0}")]
    PayloadMiss(String),
    /// An artifact URL used a scheme not wired this wave (`r2` / `hf` / `https` await the egress
    /// plane; only `file://` is resolved in Wave 1).
    #[error("artifact scheme unsupported this wave: {0}")]
    SchemeUnsupported(String),
    /// An artifact URL could not be parsed.
    #[error("malformed artifact url: {0}")]
    BadUrl(String),
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Test-only scratch directories, cleaned up on drop via `daemon_core::ContainedRoot` (so no
    //! raw-fs remove is needed and the crate takes no `tempfile` dependency).

    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use daemon_core::ContainedRoot;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A unique temp directory that removes itself (and its contents) on drop.
    pub struct TempRoot {
        path: PathBuf,
    }

    impl TempRoot {
        /// The directory path (created lazily by whichever consumer opens it).
        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            if let (Some(parent), Some(name)) = (self.path.parent(), self.path.file_name()) {
                if let Ok(root) = ContainedRoot::open(parent) {
                    let _ = root.remove_dir_all_sync(Path::new(name));
                }
            }
        }
    }

    /// Allocate a unique temp-directory handle tagged with `tag` (not yet created on disk).
    pub fn temp_root(tag: &str) -> TempRoot {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "daemon-swarm-net-{tag}-{pid}-{n}-{nanos}",
            pid = std::process::id()
        ));
        TempRoot { path }
    }
}
