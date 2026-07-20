// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-net` — the vhc transport.
//!
//! The [`VhcTransport`](transport) seam (spec §7.1): one control plane
//! ([`ControlPlane`] — publish/subscribe of already-signed message bytes, with the in-process
//! [`LoopbackGossip`] implementation) and one payload plane. The payload plane is split by era:
//! the **production** plane is content-addressed ([`ContentStore`] — opaque objects keyed by
//! blake3 alone, with the [`FsContentStore`](content_store::FsContentStore) filesystem seat and
//! the `R2Store` presigned seat); the coordinate-keyed [`PayloadStore`] (objects by
//! `(run, round, peer)` key, [`FsPayloadStore`] + retention window) is HARNESS-ERA — it predates
//! that seam and remains for the retained `RoundEngine` orbit only. Artifact fetch
//! ([`ArtifactResolver`]) resolves `file://` (blake3-verified); `r2`/`hf`/`https` are reserved
//! for the egress plane.
//!
//! **Opaque by construction:** this crate carries already-signed frame BYTES and content-addressed
//! payload objects; it defines no consensus message and decodes none (the round message schemas
//! are SDK vocabulary — `daemon_vhc_sdk_consensus::messages` — that hosts never link;
//! dep-check-enforced). Receipt production (store availability → signed evidence) is a
//! coordinator-seat function and lives with the coordinator harness drive in the session crate.
//!
//! Engine-agnostic; consumed by `daemon-vhc-session` (§10.1). Outbound HTTP must route through
//! `daemon_egress::EgressClient` (raw `reqwest::Client` is banned workspace-wide by clippy); no HTTP
//! client is constructed here yet.
//!
//! Merge-1 note: the shared identity/hash vocabulary in [`seam`] is the canonical
//! `daemon-vhc-proto` types (blake3 `Hash`, `PeerId`).
//!
//! Additions: [`Deduper`] — the reusable content-hash dedupe [`LoopbackGossip`]
//! composes (NET-6); and [`fetch_with_fallback`] — payload fetch with bounded [`RetryPolicy`]
//! backoff + fallback sources (NET-4), the miss-or-verified-bytes path the §6.4 stall ladder
//! drives.

#![forbid(unsafe_code)]

pub mod artifact;
/// A blake3-keyed, on-disk, size-bounded content cache (spec §8/§10.6) — the persistent
/// half of the artifact/shard cache the fleet warms once and never re-downloads.
pub mod content_cache;
pub mod content_store;
pub mod dedupe;
/// Multiplex several [`ControlPlane`]s (WS + iroh gossip) with cross-plane content-hash dedupe
/// (spec §7.1; A1) — the run survives one plane degrading.
pub mod dual_plane;
pub mod fetch;
pub mod gossip;
/// The real iroh-gossip control plane (spec §7.1; B2). Behind the off-default `iroh` feature so the
/// default workspace build never compiles the iroh/QUIC/relay tree.
#[cfg(feature = "iroh")]
pub mod iroh_gossip;
pub mod presign;
pub mod r2_store;
/// Run discovery + envelope fetch against the coordinator registry (spec §6.1/§11.1; A1).
pub mod registry;
pub mod seam;
pub mod seat_registry;
pub mod store;
pub mod transport;
/// The node WS coordinator client as a [`ControlPlane`] (spec §11.2; A1). Behind the off-default
/// `ws` feature so the default workspace build never compiles the WS/TLS tree.
#[cfg(feature = "ws")]
pub mod ws_client;
// The reusable harness fixtures (never production): the mock R2 presign/object server serves the
// crate's own suites too (`cfg(test)`); the loopback WS relay needs the `ws` server stack, which
// only the `harness` feature pulls.
#[cfg(any(test, feature = "harness"))]
pub mod mock_r2;
#[cfg(feature = "harness")]
pub mod ws_relay;

pub use artifact::{ArtifactCache, ArtifactRef, ArtifactResolver, ArtifactScheme};
pub use content_cache::ContentCache;
pub use content_store::FsContentStore;
pub use dedupe::Deduper;
pub use dual_plane::DualPlane;
pub use fetch::{
    fetch_with_fallback, fetch_with_fallback_dyn, DownloadScheduler, ReadyRetry, RetryConfig,
    RetryPolicy, RetryQueueResult,
};
pub use gossip::LoopbackGossip;
#[cfg(feature = "iroh")]
pub use iroh_gossip::{IrohGossip, IrohGossipConfig, IrohPeer, RebroadcastConfig};
pub use presign::{
    HttpPresignClient, ObjectKind, PresignClient, PresignOp, PresignRequest, PresignResponse,
};
pub use r2_store::{r2_object_key, R2Store};
pub use registry::{
    CheckpointPointer, RegistryClient, RunArtifact, RunDescriptor, RunState, SeatClaimOutcome,
    CHECKPOINT_KIND_DRAIN, CHECKPOINT_KIND_LIVE,
};
pub use seam::{ContentHash, PayloadKey, PeerId, RoundId, RunId};
pub use seat_registry::FakeSeatRegistry;
pub use store::FsPayloadStore;
pub use transport::{
    ContentStore, ControlPlane, ControlSubscription, MemoryContentStore, PayloadStat, PayloadStore,
};
#[cfg(feature = "ws")]
pub use ws_client::{ReconnectConfig, WsAuth, WsConfig, WsControlPlane};

/// Errors surfaced by the vhc transport.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VhcNetError {
    /// A control-plane or payload-plane transport step failed.
    #[error("vhc transport error: {0}")]
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
    /// A minted presigned URL was already past its `expires_at` (clock skew / a stale cache entry).
    /// Distinct from [`VhcNetError::PayloadMiss`]: the object may well exist — the *credential*
    /// expired, so the caller must re-request a fresh presign rather than treat the object as gone
    /// (NET-1 `store_presign_expired_rejected`).
    #[error("presigned url expired: {0}")]
    PresignExpired(String),
    /// An `hf://` artifact reference did not pin a revision (commit SHA). Unpinned HF refs are
    /// rejected: only a pinned revision is as immutable as a content-addressed object (spec §8,
    /// NET-3 `unpinned_hf_rejected`).
    #[error("hf:// reference must pin a revision (hf://<repo>@<rev>/<path>): {0}")]
    UnpinnedRevision(String),
    /// An artifact URL used a scheme the resolver does not serve yet (`r2` / `hf` / `https` await
    /// the egress plane; only `file://` is resolved).
    #[error("artifact scheme not yet supported: {0}")]
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
            "daemon-vhc-net-{tag}-{pid}-{n}-{nanos}",
            pid = std::process::id()
        ));
        TempRoot { path }
    }
}
