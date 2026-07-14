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
//!
//! Wave-2 (R2) additions: [`Deduper`] — the reusable content-hash dedupe [`LoopbackGossip`]
//! composes (NET-6); and [`fetch_with_fallback`] — payload fetch with bounded [`RetryPolicy`]
//! backoff + fallback sources (NET-4), the miss-or-verified-bytes path the §6.4 stall ladder
//! drives.

#![forbid(unsafe_code)]

pub mod artifact;
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
pub mod receipt;
/// Run discovery + envelope fetch against the coordinator registry (spec §6.1/§11.1; A1).
pub mod registry;
pub mod seam;
pub mod store;
pub mod transport;
/// The node WS coordinator client as a [`ControlPlane`] (spec §11.2; A1). Behind the off-default
/// `ws` feature so the default workspace build never compiles the WS/TLS tree.
#[cfg(feature = "ws")]
pub mod ws_client;

pub use artifact::{ArtifactCache, ArtifactRef, ArtifactResolver, ArtifactScheme};
pub use dedupe::Deduper;
pub use dual_plane::DualPlane;
pub use fetch::{
    fetch_record_set, fetch_with_fallback, fetch_with_fallback_dyn, DownloadScheduler, ReadyRetry,
    RetryConfig, RetryPolicy, RetryQueueResult,
};
pub use gossip::LoopbackGossip;
#[cfg(feature = "iroh")]
pub use iroh_gossip::{IrohGossip, IrohGossipConfig, IrohPeer, RebroadcastConfig};
pub use presign::{
    HttpPresignClient, ObjectKind, PresignClient, PresignOp, PresignRequest, PresignResponse,
};
pub use r2_store::{r2_object_key, R2Store};
pub use receipt::ReceiptProducer;
pub use registry::{RegistryClient, RunArtifact, RunDescriptor};
pub use seam::{ContentHash, PayloadKey, PeerId, RoundId, RunId};
pub use store::FsPayloadStore;
pub use transport::{ControlPlane, ControlSubscription, PayloadStat, PayloadStore};
#[cfg(feature = "ws")]
pub use ws_client::{ReconnectConfig, WsAuth, WsConfig, WsControlPlane};

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
    /// A minted presigned URL was already past its `expires_at` (clock skew / a stale cache entry).
    /// Distinct from [`SwarmNetError::PayloadMiss`]: the object may well exist — the *credential*
    /// expired, so the caller must re-request a fresh presign rather than treat the object as gone
    /// (NET-1 `store_presign_expired_rejected`).
    #[error("presigned url expired: {0}")]
    PresignExpired(String),
    /// An `hf://` artifact reference did not pin a revision (commit SHA). Unpinned HF refs are
    /// rejected: only a pinned revision is as immutable as a content-addressed object (spec §8,
    /// NET-3 `unpinned_hf_rejected`).
    #[error("hf:// reference must pin a revision (hf://<repo>@<rev>/<path>): {0}")]
    UnpinnedRevision(String),
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

#[cfg(test)]
pub(crate) mod mock_r2 {
    //! An in-process mock of the coordinator presign endpoint + the R2 object store (NET-1/3/8),
    //! built on `wiremock` (a dev-dep; no live network). Mirrors what BC's `apps/swarm` worker does:
    //! `POST /api/v1/swarm/runs/:id/presign` returns a URL into a stateful `/obj/*` PUT/GET store at
    //! the spec §11.3 object key. It can mint expired presigns (`with_expiry`) and drop objects
    //! (`evict`) for the negative cases.

    use std::collections::{BTreeMap, HashMap};
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    use daemon_egress::{EgressClient, EgressConfig};
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    use crate::presign::{HttpPresignClient, PresignRequest, PresignResponse};
    use crate::r2_store::r2_object_key;
    use crate::seam::RunId;

    type Objects = Arc<Mutex<HashMap<String, Vec<u8>>>>;

    fn now_s() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    /// The presign responder: parses the request body, computes the §11.3 object key, and returns a
    /// URL into this server's `/obj/*` store with the configured expiry.
    struct Presigner {
        base: String,
        expiry_offset_s: i64,
    }

    impl Respond for Presigner {
        fn respond(&self, req: &Request) -> ResponseTemplate {
            let run = req
                .url
                .path()
                .trim_start_matches('/')
                .strip_prefix("api/v1/swarm/runs/")
                .and_then(|s| s.split('/').next())
                .unwrap_or_default()
                .to_string();
            let preq: PresignRequest = match serde_json::from_slice(&req.body) {
                Ok(p) => p,
                Err(e) => return ResponseTemplate::new(400).set_body_string(e.to_string()),
            };
            let key = match r2_object_key(&RunId::new(run), &preq) {
                Ok(k) => k,
                Err(e) => return ResponseTemplate::new(400).set_body_string(e.to_string()),
            };
            let expires_at = (now_s() + self.expiry_offset_s).max(0) as u64;
            let resp = PresignResponse {
                url: format!("{}/obj/{key}?sig=mock", self.base),
                expires_at,
                headers: BTreeMap::new(),
            };
            ResponseTemplate::new(200).set_body_json(serde_json::to_value(&resp).expect("json"))
        }
    }

    struct PutObj {
        objects: Objects,
    }
    impl Respond for PutObj {
        fn respond(&self, req: &Request) -> ResponseTemplate {
            self.objects
                .lock()
                .expect("objects mutex")
                .insert(req.url.path().to_string(), req.body.clone());
            ResponseTemplate::new(200)
        }
    }

    struct GetObj {
        objects: Objects,
    }
    impl Respond for GetObj {
        fn respond(&self, req: &Request) -> ResponseTemplate {
            match self
                .objects
                .lock()
                .expect("objects mutex")
                .get(req.url.path())
            {
                Some(bytes) => ResponseTemplate::new(200).set_body_bytes(bytes.clone()),
                None => ResponseTemplate::new(404),
            }
        }
    }

    /// A running mock coordinator + object store.
    pub(crate) struct MockR2 {
        server: MockServer,
        objects: Objects,
    }

    impl MockR2 {
        /// Start with a healthy 15-minute presign expiry.
        pub async fn start() -> Self {
            Self::with_expiry(900).await
        }

        /// Start with presigns that expire `expiry_offset_s` seconds from now (negative = already
        /// expired — the `store_presign_expired_rejected` case).
        pub async fn with_expiry(expiry_offset_s: i64) -> Self {
            let server = MockServer::start().await;
            let base = server.uri();
            let objects: Objects = Arc::new(Mutex::new(HashMap::new()));
            Mock::given(method("POST"))
                .and(path_regex(r"^/api/v1/swarm/runs/[^/]+/presign$"))
                .respond_with(Presigner {
                    base: base.clone(),
                    expiry_offset_s,
                })
                .mount(&server)
                .await;
            Mock::given(method("PUT"))
                .and(path_regex(r"^/obj/"))
                .respond_with(PutObj {
                    objects: objects.clone(),
                })
                .mount(&server)
                .await;
            Mock::given(method("GET"))
                .and(path_regex(r"^/obj/"))
                .respond_with(GetObj {
                    objects: objects.clone(),
                })
                .mount(&server)
                .await;
            Self { server, objects }
        }

        /// The swarm coordinator base URL (`{uri}/api/v1/swarm`).
        pub fn coordinator_base(&self) -> String {
            format!("{}/api/v1/swarm", self.server.uri())
        }

        /// A fresh SSRF-safe egress client (the initial hop to the loopback mock is not re-checked).
        pub fn egress(&self) -> EgressClient {
            EgressClient::new(EgressConfig::default()).expect("egress client")
        }

        /// An [`HttpPresignClient`] pointed at this mock.
        pub fn presign_client(&self) -> HttpPresignClient {
            HttpPresignClient::new(self.egress(), self.coordinator_base())
        }

        /// Seed an object directly at its §11.3 key (bypassing PUT) — GET-only / artifact cases.
        pub fn seed(&self, object_key: &str, bytes: Vec<u8>) {
            self.objects
                .lock()
                .expect("objects mutex")
                .insert(format!("/obj/{object_key}"), bytes);
        }

        /// Drop a stored object (simulate lifecycle/retention expiry).
        pub fn evict(&self, object_key: &str) {
            self.objects
                .lock()
                .expect("objects mutex")
                .remove(&format!("/obj/{object_key}"));
        }
    }
}
