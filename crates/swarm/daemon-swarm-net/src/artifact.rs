// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Artifact fetch: scheme-dispatch resolution + blake3 verification (spec §8, §12).
//!
//! Everything a run references externally goes through the envelope's artifact map: a name →
//! `(url, blake3)` table the *host* fetches, verifies, and caches (the module has no I/O and
//! addresses artifacts by name). [`ArtifactResolver::new`] resolves **`file://` only** (no egress);
//! [`ArtifactResolver::with_egress`] additionally wires the network schemes through the SSRF-safe
//! [`EgressClient`] (raw `reqwest::Client` is clippy-banned outside `daemon-egress`):
//!
//! - **`https://`** → `EgressClient` GET with `Redirects::FollowValidated` (§8 "data host").
//! - **`hf://<repo>@<rev>/<path>`** → mapped to the pinned HF resolve URL
//!   `https://huggingface.co/<repo>/resolve/<rev>/<path>` and GET through egress. **Unpinned refs
//!   are rejected** ([`SwarmNetError::UnpinnedRevision`]) — only a pinned revision is as immutable
//!   as a content-addressed object (§8). Mirrors the revision-pin shape of daemon-models
//!   `crates/providers/daemon-models/src/acquire.rs:395` (`Repo::with_revision`), but over
//!   `EgressClient` (hf-hub does its own HTTP, which would bypass the egress gate).
//! - **`r2://<path>`** → a presigned GET via [`PresignClient`](crate::PresignClient) (enable with
//!   [`ArtifactResolver::with_presign`]), then GET through egress.
//!
//! Every scheme blake3-verifies the fetched bytes against the artifact map's hash before returning
//! (tamper/corruption reject, §12). [`ArtifactCache`] is the RUN-4 LRU that bounds resolved-artifact
//! bytes by `[swarm].data_cache_gb`.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use daemon_egress::{EgressClient, Redirects};
use daemon_swarm_proto::blake3_hash;

use crate::presign::{PresignClient, PresignOp, PresignRequest};
use crate::seam::{ContentHash, RunId};
use crate::SwarmNetError;

/// The transport scheme of an artifact URL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactScheme {
    /// `file://` — a local absolute path (wired this wave).
    File,
    /// `r2://` — a presigned R2/S3 object (reserved; awaits the egress plane).
    R2,
    /// `hf://` — a Hugging Face repo artifact, revision-pinned (reserved; awaits the egress plane).
    Hf,
    /// `https://` — a plain static host (reserved; awaits the egress plane).
    Https,
}

impl ArtifactScheme {
    /// Split `url` into its scheme + the remainder after `scheme://`.
    fn parse(url: &str) -> Result<(Self, &str), SwarmNetError> {
        let (scheme, rest) = url
            .split_once("://")
            .ok_or_else(|| SwarmNetError::BadUrl(format!("missing scheme separator: {url}")))?;
        let scheme = match scheme {
            "file" => Self::File,
            "r2" => Self::R2,
            "hf" => Self::Hf,
            "https" => Self::Https,
            other => {
                return Err(SwarmNetError::BadUrl(format!("unknown scheme: {other}")));
            }
        };
        Ok((scheme, rest))
    }
}

/// One entry of the envelope artifact map: a URL plus the blake3 it must hash to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactRef {
    /// The source URL (`file://…` this wave).
    pub url: String,
    /// The blake3 the fetched bytes must match (content addressing, §8/§12).
    pub blake3: ContentHash,
}

impl ArtifactRef {
    /// Construct an artifact reference.
    pub fn new(url: impl Into<String>, blake3: ContentHash) -> Self {
        Self {
            url: url.into(),
            blake3,
        }
    }
}

/// Resolves artifact references, verifying each against its blake3.
///
/// [`ArtifactResolver::new`] is file-only (no egress); [`ArtifactResolver::with_egress`] wires the
/// `https`/`hf` schemes, and [`ArtifactResolver::with_presign`] additionally enables `r2://`.
#[derive(Clone)]
pub struct ArtifactResolver {
    egress: Option<EgressClient>,
    presign: Option<Arc<dyn PresignClient>>,
    run: Option<RunId>,
    /// The HF Hub base (default `https://huggingface.co`); overridable for a private mirror / tests.
    hf_base: String,
}

impl Default for ArtifactResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// The public HF Hub base — where `hf://` refs resolve unless a mirror is configured.
const HF_HUB_BASE: &str = "https://huggingface.co";

impl std::fmt::Debug for ArtifactResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArtifactResolver")
            .field("egress", &self.egress.is_some())
            .field("presign", &self.presign.is_some())
            .field("run", &self.run)
            .field("hf_base", &self.hf_base)
            .finish()
    }
}

impl ArtifactResolver {
    /// A **file-only** resolver (`file://`, blake3-verified). Network schemes return
    /// [`SwarmNetError::SchemeUnsupported`] until [`ArtifactResolver::with_egress`] is used.
    #[must_use]
    pub fn new() -> Self {
        Self {
            egress: None,
            presign: None,
            run: None,
            hf_base: HF_HUB_BASE.to_string(),
        }
    }

    /// A resolver that additionally fetches `https://` and `hf://` (pinned-revision) artifacts
    /// through the SSRF-safe [`EgressClient`]. `file://` still works; `r2://` needs
    /// [`ArtifactResolver::with_presign`].
    #[must_use]
    pub fn with_egress(egress: EgressClient) -> Self {
        Self {
            egress: Some(egress),
            presign: None,
            run: None,
            hf_base: HF_HUB_BASE.to_string(),
        }
    }

    /// Override the HF Hub base (default `https://huggingface.co`) — for a private HF mirror or a
    /// hermetic test server. Mirrors daemon-models `HfClient::with_endpoint`.
    #[must_use]
    pub fn with_hf_endpoint(mut self, base: impl Into<String>) -> Self {
        self.hf_base = base.into().trim_end_matches('/').to_string();
        self
    }

    /// Enable `r2://` resolution: presign a GET for the object through `presign` (scoped to `run`),
    /// then fetch it via egress. Chain after [`ArtifactResolver::with_egress`] (r2 needs both).
    #[must_use]
    pub fn with_presign(mut self, presign: Arc<dyn PresignClient>, run: RunId) -> Self {
        self.presign = Some(presign);
        self.run = Some(run);
        self
    }

    /// Fetch `artifact` and verify its blake3. A mismatch is a typed
    /// [`SwarmNetError::HashMismatch`] (tamper/corruption reject, §12).
    pub async fn fetch(&self, artifact: &ArtifactRef) -> Result<Vec<u8>, SwarmNetError> {
        let bytes = self.fetch_raw(&artifact.url).await?;
        let actual = blake3_hash(&bytes);
        if actual != artifact.blake3 {
            return Err(SwarmNetError::HashMismatch {
                expected: artifact.blake3.to_hex(),
                actual: actual.to_hex(),
            });
        }
        Ok(bytes)
    }

    /// Fetch the raw bytes for `url`, dispatching on scheme (no verification).
    async fn fetch_raw(&self, url: &str) -> Result<Vec<u8>, SwarmNetError> {
        let (scheme, rest) = ArtifactScheme::parse(url)?;
        match scheme {
            ArtifactScheme::File => read_file_uri(rest).await,
            ArtifactScheme::Https => self.fetch_https(url).await,
            ArtifactScheme::Hf => self.fetch_hf(rest).await,
            ArtifactScheme::R2 => self.fetch_r2(rest).await,
        }
    }

    /// The configured egress client, or a typed "scheme needs egress" error.
    fn egress(&self, scheme: &str) -> Result<&EgressClient, SwarmNetError> {
        self.egress.as_ref().ok_or_else(|| {
            SwarmNetError::SchemeUnsupported(format!(
                "{scheme} needs an egress client — build with ArtifactResolver::with_egress"
            ))
        })
    }

    /// `https://` → egress GET, browser-style validated redirects (data hosts 302 to a CDN).
    async fn fetch_https(&self, url: &str) -> Result<Vec<u8>, SwarmNetError> {
        egress_get_bytes(self.egress("https://")?, url).await
    }

    /// `hf://<repo>@<rev>/<path>` → the pinned HF resolve URL, via egress. Unpinned → reject (§8).
    async fn fetch_hf(&self, rest: &str) -> Result<Vec<u8>, SwarmNetError> {
        let url = hf_resolve_url(&self.hf_base, rest)?;
        egress_get_bytes(self.egress("hf://")?, &url).await
    }

    /// `r2://<path>` → presigned GET (artifact object key = run-relative `path`), via egress.
    async fn fetch_r2(&self, rest: &str) -> Result<Vec<u8>, SwarmNetError> {
        let egress = self.egress("r2://")?;
        let presign = self.presign.as_ref().ok_or_else(|| {
            SwarmNetError::SchemeUnsupported(
                "r2:// needs a presign client — chain ArtifactResolver::with_presign".into(),
            )
        })?;
        let run = self.run.as_ref().ok_or_else(|| {
            SwarmNetError::SchemeUnsupported(
                "r2:// needs a run — chain with_presign(.., run)".into(),
            )
        })?;
        let path = rest.trim_start_matches('/');
        let req = PresignRequest::artifact(PresignOp::Get, path);
        let resp = presign.presign(run, &req).await?;
        egress_get_bytes(egress, &resp.url).await
    }
}

/// Resolve the local path of a `file://` URI's remainder (`<host>/<abs-path>`), accepting an empty
/// or `localhost` host per RFC 8089.
fn file_uri_path(rest: &str) -> Result<PathBuf, SwarmNetError> {
    // `file:///abs/path` -> rest = "/abs/path"; `file://localhost/abs` -> rest = "localhost/abs".
    let path = if let Some(stripped) = rest.strip_prefix('/') {
        // Empty host: rest began with the leading slash of the absolute path.
        format!("/{stripped}")
    } else if let Some((host, path)) = rest.split_once('/') {
        if !host.is_empty() && host != "localhost" {
            return Err(SwarmNetError::BadUrl(format!(
                "file:// host must be empty or localhost, got {host:?}"
            )));
        }
        format!("/{path}")
    } else {
        return Err(SwarmNetError::BadUrl(format!(
            "file:// url has no path: file://{rest}"
        )));
    };
    Ok(PathBuf::from(path))
}

/// Read a `file://` artifact's bytes.
///
/// `file://` artifact URLs come from the run **envelope's** artifact map — authored and signed
/// (§4.3/§8), not attacker-influenced relative paths — and the bytes are blake3-verified by
/// [`ArtifactResolver::fetch`] immediately after read. `ContainedRoot`'s relative-containment model
/// does not apply to an absolute, operator-pinned path, so this is the one sanctioned raw-fs read in
/// the crate. (Network schemes route through `daemon-egress`, never raw fs.)
#[allow(clippy::disallowed_methods)]
async fn read_file_uri(rest: &str) -> Result<Vec<u8>, SwarmNetError> {
    let path = file_uri_path(rest)?;
    tokio::fs::read(&path)
        .await
        .map_err(|e| SwarmNetError::Fetch(format!("read {}: {e}", path.display())))
}

/// GET `url` through the egress client and return the body, mapping non-2xx + transport failures to
/// [`SwarmNetError::Fetch`]. Validated redirects are followed (data hosts / HF `resolve` 302 to CDN).
async fn egress_get_bytes(egress: &EgressClient, url: &str) -> Result<Vec<u8>, SwarmNetError> {
    let resp = egress
        .get(url, Redirects::DEFAULT)
        .await
        .map_err(|e| SwarmNetError::Fetch(format!("egress GET {url}: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(SwarmNetError::Fetch(format!("GET {url} returned {status}")));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| SwarmNetError::Fetch(format!("read body {url}: {e}")))?;
    Ok(bytes.to_vec())
}

/// Map an `hf://<repo>@<rev>/<path>` remainder to the pinned HF resolve URL
/// `<base>/<repo>/resolve/<rev>/<path>` (`base` is normally `https://huggingface.co`).
///
/// `<repo>` may itself contain `/` (`org/name`); the revision pin is the segment after the **first**
/// `@`. A remainder with no `@` (unpinned) is rejected — only a pinned revision is immutable (§8).
/// Mirrors daemon-models `acquire.rs:395` (`Repo::with_revision(repo, Model, rev)`), which resolves
/// into the same `/resolve/<rev>/` URL space.
fn hf_resolve_url(base: &str, rest: &str) -> Result<String, SwarmNetError> {
    let (repo, right) = rest
        .split_once('@')
        .ok_or_else(|| SwarmNetError::UnpinnedRevision(format!("hf://{rest}")))?;
    let (rev, path) = right.split_once('/').ok_or_else(|| {
        SwarmNetError::BadUrl(format!("hf:// reference has no path: hf://{rest}"))
    })?;
    if repo.is_empty() || rev.is_empty() || path.is_empty() {
        return Err(SwarmNetError::BadUrl(format!(
            "hf:// reference must be hf://<repo>@<rev>/<path>: hf://{rest}"
        )));
    }
    Ok(format!("{base}/{repo}/resolve/{rev}/{path}"))
}

/// A bounded LRU cache of resolved artifact bytes (spec §8, §10.6; TDD RUN-4).
///
/// Artifacts (module `.wasm`, tokenizer, shards) download lazily ahead-of-need into a workspace
/// cache bounded by `[swarm].data_cache_gb`. This is that bound: an LRU over the resolved bytes,
/// keyed by the artifact URL, evicting least-recently-used entries until a new insert fits. Bytes
/// are already blake3-verified by [`ArtifactResolver::fetch`] before they land here.
///
/// The cache is `&mut`-driven (single owner); B3/M2 wire it around the resolver at call sites.
pub struct ArtifactCache {
    max_bytes: u64,
    used: u64,
    entries: HashMap<String, Vec<u8>>,
    /// Recency order: front = least-recently-used, back = most-recently-used.
    order: VecDeque<String>,
}

impl ArtifactCache {
    /// A cache bounded by `max_bytes` total resolved-artifact bytes.
    #[must_use]
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            used: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// A cache bounded by `[swarm].data_cache_gb` gibibytes (the §10.6 knob, RUN-4).
    #[must_use]
    pub fn from_gb(data_cache_gb: u32) -> Self {
        Self::new(u64::from(data_cache_gb) * (1 << 30))
    }

    /// The byte budget.
    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Bytes currently held.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    /// Number of cached artifacts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `url` is currently cached (does not affect recency).
    #[must_use]
    pub fn contains(&self, url: &str) -> bool {
        self.entries.contains_key(url)
    }

    /// Fetch a cached artifact by `url`, marking it most-recently-used on a hit.
    pub fn get(&mut self, url: &str) -> Option<&[u8]> {
        if self.entries.contains_key(url) {
            self.touch(url);
            self.entries.get(url).map(Vec::as_slice)
        } else {
            None
        }
    }

    /// Insert `bytes` under `url`, evicting least-recently-used entries until it fits. An object
    /// larger than the whole budget is not cached (returns without storing).
    pub fn insert(&mut self, url: impl Into<String>, bytes: Vec<u8>) {
        let url = url.into();
        let len = bytes.len() as u64;
        // Replacing an existing key: drop its old size + order slot first.
        if let Some(old) = self.entries.remove(&url) {
            self.used -= old.len() as u64;
            self.order.retain(|k| k != &url);
        }
        if len > self.max_bytes {
            return; // too big to ever cache — leave it uncached rather than evict everything.
        }
        while self.used + len > self.max_bytes {
            let Some(lru) = self.order.pop_front() else {
                break;
            };
            if let Some(evicted) = self.entries.remove(&lru) {
                self.used -= evicted.len() as u64;
            }
        }
        self.used += len;
        self.entries.insert(url.clone(), bytes);
        self.order.push_back(url);
    }

    /// Move `url` to the most-recently-used position.
    fn touch(&mut self, url: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == url) {
            let key = self.order.remove(pos).expect("position just found");
            self.order.push_back(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_root;
    use daemon_core::ContainedRoot;
    use std::path::Path;

    /// The canonical blake3 test vector for the empty input (pinned golden, NET-2).
    const BLAKE3_EMPTY_HEX: &str =
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";

    #[test]
    fn blake3_empty_golden() {
        assert_eq!(blake3_hash(b"").to_hex(), BLAKE3_EMPTY_HEX);
    }

    #[test]
    fn scheme_parsing() {
        assert_eq!(
            ArtifactScheme::parse("file:///a/b").unwrap(),
            (ArtifactScheme::File, "/a/b")
        );
        assert_eq!(
            ArtifactScheme::parse("hf://repo@rev/f").unwrap().0,
            ArtifactScheme::Hf
        );
        assert!(matches!(
            ArtifactScheme::parse("no-scheme"),
            Err(SwarmNetError::BadUrl(_))
        ));
    }

    /// Write a file into a temp dir via `ContainedRoot` and return its absolute path + a `file://`
    /// URL for it.
    async fn write_artifact(dir: &Path, name: &str, bytes: &[u8]) -> (PathBuf, String) {
        let root = ContainedRoot::open(dir).unwrap();
        root.write(Path::new(name), bytes).await.unwrap();
        let abs = dir.join(name);
        let url = format!("file://{}", abs.display());
        (abs, url)
    }

    #[tokio::test]
    async fn fetch_file_verifies_blake3() {
        let dir = temp_root("artifact-ok");
        let (_abs, url) = write_artifact(dir.path(), "module.wasm", b"wasm-bytes").await;
        let art = ArtifactRef::new(url, blake3_hash(b"wasm-bytes"));

        let got = ArtifactResolver::new().fetch(&art).await.unwrap();
        assert_eq!(got, b"wasm-bytes");
    }

    #[tokio::test]
    async fn fetch_file_rejects_tamper() {
        let dir = temp_root("artifact-tamper");
        let (_abs, url) = write_artifact(dir.path(), "module.wasm", b"tampered").await;
        // The artifact map claims a different blake3 than the file actually has.
        let art = ArtifactRef::new(url, blake3_hash(b"expected-original"));

        let err = ArtifactResolver::new().fetch(&art).await.unwrap_err();
        assert!(
            matches!(err, SwarmNetError::HashMismatch { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn missing_file_is_fetch_error() {
        let art = ArtifactRef::new("file:///no/such/daemon-swarm/artifact", blake3_hash(b""));
        let err = ArtifactResolver::new().fetch(&art).await.unwrap_err();
        assert!(matches!(err, SwarmNetError::Fetch(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn network_schemes_unsupported_without_egress() {
        for url in [
            "r2://bucket/obj",
            "hf://org/repo@abcdef/file",
            "https://host/x",
        ] {
            let art = ArtifactRef::new(url, blake3_hash(b""));
            let err = ArtifactResolver::new().fetch(&art).await.unwrap_err();
            assert!(
                matches!(err, SwarmNetError::SchemeUnsupported(_)),
                "{url}: got {err:?}"
            );
        }
    }

    #[test]
    fn hf_pinned_maps_to_resolve_url() {
        assert_eq!(
            hf_resolve_url(HF_HUB_BASE, "org/repo@abcdef0123/tokenizer.json").unwrap(),
            "https://huggingface.co/org/repo/resolve/abcdef0123/tokenizer.json"
        );
        // A single-segment repo is fine too.
        assert_eq!(
            hf_resolve_url(HF_HUB_BASE, "gpt2@main/config.json").unwrap(),
            "https://huggingface.co/gpt2/resolve/main/config.json"
        );
    }

    #[test]
    fn hf_unpinned_url_rejected() {
        // No `@rev` at all — the reject the NET-3 `unpinned_hf_rejected` case asserts (unit level).
        let err = hf_resolve_url(HF_HUB_BASE, "org/repo/file.json").unwrap_err();
        assert!(
            matches!(err, SwarmNetError::UnpinnedRevision(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn hf_pinned_without_path_is_bad_url() {
        let err = hf_resolve_url(HF_HUB_BASE, "org/repo@rev").unwrap_err();
        assert!(matches!(err, SwarmNetError::BadUrl(_)), "got {err:?}");
    }

    #[test]
    fn artifact_cache_lru_evicts() {
        // Budget = 3 objects of 100 bytes each.
        let mut cache = ArtifactCache::new(300);
        cache.insert("r2://a", vec![0u8; 100]);
        cache.insert("r2://b", vec![0u8; 100]);
        cache.insert("r2://c", vec![0u8; 100]);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.used_bytes(), 300);

        // Touch `a` so it is most-recently-used; `b` is now the LRU.
        assert!(cache.get("r2://a").is_some());

        // Inserting `d` must evict the LRU (`b`), not `a`.
        cache.insert("r2://d", vec![0u8; 100]);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.used_bytes(), 300);
        assert!(cache.contains("r2://a"), "recently-used survives");
        assert!(!cache.contains("r2://b"), "LRU evicted");
        assert!(cache.contains("r2://c"));
        assert!(cache.contains("r2://d"));
    }

    #[test]
    fn artifact_cache_skips_oversize_object() {
        let mut cache = ArtifactCache::new(100);
        cache.insert("r2://big", vec![0u8; 200]);
        assert!(
            cache.is_empty(),
            "an object larger than the budget is not cached"
        );
        assert_eq!(cache.used_bytes(), 0);
    }

    #[test]
    fn artifact_cache_from_gb() {
        let cache = ArtifactCache::from_gb(50);
        assert_eq!(cache.capacity_bytes(), 50u64 * (1 << 30));
    }

    // --- NET-2/NET-3: egress-scheme resolution against in-process mock servers ---------------------
    //
    // wiremock speaks plaintext HTTP, so the network-scheme end-to-end tests drive the `r2://` and
    // `hf://` paths (which fetch the presigned / resolve URL through egress over http). The `https://`
    // scheme is the same `egress_get_bytes` code path; its dispatch is asserted separately (a fetch
    // is *attempted*, not `SchemeUnsupported`). NET-2's blake3 verify is exercised over `r2://`.

    use crate::mock_r2::MockR2;
    use crate::seam::RunId;
    use daemon_egress::{EgressClient, EgressConfig};
    use std::sync::Arc;
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn egress() -> EgressClient {
        EgressClient::new(EgressConfig::default()).expect("egress")
    }

    /// NET-2: an artifact whose fetched bytes hash to the artifact-map blake3 resolves cleanly
    /// (verify happens in `fetch`, scheme-independent — driven here over the hermetic `r2://` path).
    #[tokio::test]
    async fn verify_artifact_ok() {
        let mock = MockR2::start().await;
        let body = b"module-bytes".to_vec();
        mock.seed("runs/run-x/experiment.wasm", body.clone());
        let resolver = ArtifactResolver::with_egress(mock.egress())
            .with_presign(Arc::new(mock.presign_client()), RunId::new("run-x"));
        let art = ArtifactRef::new("r2://experiment.wasm", blake3_hash(&body));
        assert_eq!(resolver.fetch(&art).await.unwrap(), body);
    }

    /// NET-2: bytes that do not hash to the artifact-map blake3 are a typed tamper reject.
    #[tokio::test]
    async fn verify_artifact_tamper() {
        let mock = MockR2::start().await;
        mock.seed("runs/run-x/experiment.wasm", b"tampered".to_vec());
        let resolver = ArtifactResolver::with_egress(mock.egress())
            .with_presign(Arc::new(mock.presign_client()), RunId::new("run-x"));
        // The artifact map claims a different blake3 than the object actually has.
        let art = ArtifactRef::new("r2://experiment.wasm", blake3_hash(b"the-original"));
        let err = resolver.fetch(&art).await.unwrap_err();
        assert!(
            matches!(err, SwarmNetError::HashMismatch { .. }),
            "got {err:?}"
        );
    }

    /// NET-3: an `hf://<repo>@<rev>/<path>` ref resolves against the pinned `/resolve/<rev>/` URL.
    #[tokio::test]
    async fn resolve_hf_pinned_ok() {
        let server = MockServer::start().await;
        let body = b"pinned-weights".to_vec();
        Mock::given(method("GET"))
            .and(wm_path("/org/model/resolve/abcdef012345/model.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let resolver = ArtifactResolver::with_egress(egress()).with_hf_endpoint(server.uri());
        let art = ArtifactRef::new("hf://org/model@abcdef012345/model.bin", blake3_hash(&body));
        assert_eq!(resolver.fetch(&art).await.unwrap(), body);
    }

    /// NET-3: an unpinned `hf://` ref is rejected end-to-end (before any fetch).
    #[tokio::test]
    async fn unpinned_hf_rejected() {
        let resolver = ArtifactResolver::with_egress(egress());
        let art = ArtifactRef::new("hf://org/model/model.bin", blake3_hash(b""));
        let err = resolver.fetch(&art).await.unwrap_err();
        assert!(
            matches!(err, SwarmNetError::UnpinnedRevision(_)),
            "got {err:?}"
        );
    }

    /// NET-3: an `r2://` artifact resolves via a presigned GET through the [`crate::PresignClient`].
    #[tokio::test]
    async fn r2_to_presign() {
        let mock = MockR2::start().await;
        let body = b"experiment-wasm".to_vec();
        // The artifact object lives at `runs/<run>/experiment.wasm` (§11.3 artifact key).
        mock.seed("runs/run-x/experiment.wasm", body.clone());
        let resolver = ArtifactResolver::with_egress(mock.egress())
            .with_presign(Arc::new(mock.presign_client()), RunId::new("run-x"));
        let art = ArtifactRef::new("r2://experiment.wasm", blake3_hash(&body));
        assert_eq!(resolver.fetch(&art).await.unwrap(), body);
    }

    /// The `https://` scheme is *wired* to egress (a fetch is attempted, not `SchemeUnsupported`):
    /// an unreachable host yields a `Fetch` error, proving dispatch reaches `egress_get_bytes`.
    #[tokio::test]
    async fn https_scheme_routes_through_egress() {
        let resolver = ArtifactResolver::with_egress(egress());
        // Port 1 on loopback: connection refused immediately (no TLS handshake completes) → Fetch.
        let art = ArtifactRef::new("https://127.0.0.1:1/artifact", blake3_hash(b""));
        let err = resolver.fetch(&art).await.unwrap_err();
        assert!(
            matches!(err, SwarmNetError::Fetch(_)),
            "https must route through egress (got {err:?})"
        );
    }
}
