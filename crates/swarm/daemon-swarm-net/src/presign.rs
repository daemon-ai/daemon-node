// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`PresignClient`] — the coordinator presign seam (spec §11.1) + its HTTP implementation.
//!
//! The `r2` payload plane keeps the node **S3-SDK-free**: SigV4 lives in the coordinator app (BC's
//! `apps/swarm` worker), which mints short-lived presigned URLs on demand. A peer calls
//! `POST <coordinator_base>/runs/:id/presign` with `{kind, op, round?, peer?, path?}` and gets back
//! `{url, expires_at, headers?}`; it then PUTs/GETs the object bytes at `url` through the SSRF-safe
//! [`EgressClient`] (raw `reqwest::Client` is clippy-banned outside `daemon-egress`).
//!
//! The DTOs here are the **frozen node↔cloud HTTP contract** (program Risk 6): the byte-exact shapes
//! are pinned as `tests/fixtures/presign-*.json`, which BC's worker and B3's live client both consume
//! verbatim. Merge 1 freezes this trait + those fixtures.
//!
//! [`HttpPresignClient`] caches responses keyed to their `expires_at`, so a burst of PUT/GET ops on
//! the same object presigns once; a minted URL already past `expires_at` is rejected up-front as
//! [`SwarmNetError::PresignExpired`] (NET-1 `store_presign_expired_rejected`).

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use daemon_egress::{EgressClient, EgressRequest, Redirects};
use serde::{Deserialize, Serialize};

use crate::seam::RunId;
use crate::SwarmNetError;

/// The class of object being presigned — selects the R2 key layout (spec §11.3).
///
/// Serializes kebab-case (`"payload"`, `"record-set"`, `"checkpoint"`, `"artifact"`) — the wire
/// tokens BC matches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObjectKind {
    /// A round update object: `runs/<run>/rounds/<round>/<peer_hex>.upd` (needs `round` + `peer`).
    Payload,
    /// The committed-set object: `runs/<run>/rounds/<round>/record-set.cbor` (needs `round`).
    RecordSet,
    /// An epoch checkpoint: `runs/<run>/checkpoints/round-<round>.safetensors` (needs `round`).
    Checkpoint,
    /// An envelope artifact (module / tokenizer / shard): `runs/<run>/<path>` (needs `path`, §8).
    Artifact,
}

/// The presigned HTTP method (spec §11.1: presigned PUT/GET).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PresignOp {
    /// A presigned `PUT` (upload the object bytes).
    Put,
    /// A presigned `GET` (download the object bytes). Also the `head` path (see [`crate::R2Store`]).
    Get,
}

/// The presign **request** body (JSON) — `POST /api/v1/swarm/runs/:id/presign`.
///
/// One endpoint serves both round objects (§11.1) and `r2://` envelope artifacts (§8): `round`/`peer`
/// are set for round objects and `path` for artifacts; each is omitted from the JSON when unset
/// (`skip_serializing_if`). This is B1's generalisation of the brief's `{round, peer, kind, op}` so a
/// second endpoint is not needed — BC validates field presence per kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresignRequest {
    /// The object class (selects the §11.3 key layout).
    pub kind: ObjectKind,
    /// The HTTP method to presign.
    pub op: PresignOp,
    /// Round number — set for `payload` / `record-set` / `checkpoint`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub round: Option<u64>,
    /// Peer node-id hex — set for `payload` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer: Option<String>,
    /// Run-relative object key — set for `artifact` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

impl PresignRequest {
    /// A round-payload request (needs `round` + `peer`).
    #[must_use]
    pub fn payload(op: PresignOp, round: u64, peer_hex: impl Into<String>) -> Self {
        Self {
            kind: ObjectKind::Payload,
            op,
            round: Some(round),
            peer: Some(peer_hex.into()),
            path: None,
        }
    }

    /// A record-set request (needs `round`).
    #[must_use]
    pub fn record_set(op: PresignOp, round: u64) -> Self {
        Self {
            kind: ObjectKind::RecordSet,
            op,
            round: Some(round),
            peer: None,
            path: None,
        }
    }

    /// A checkpoint request (needs `round`).
    #[must_use]
    pub fn checkpoint(op: PresignOp, round: u64) -> Self {
        Self {
            kind: ObjectKind::Checkpoint,
            op,
            round: Some(round),
            peer: None,
            path: None,
        }
    }

    /// An artifact request (needs the run-relative `path`).
    #[must_use]
    pub fn artifact(op: PresignOp, path: impl Into<String>) -> Self {
        Self {
            kind: ObjectKind::Artifact,
            op,
            round: None,
            peer: None,
            path: Some(path.into()),
        }
    }
}

/// The presign **response** body (JSON).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresignResponse {
    /// The presigned URL to `PUT`/`GET`.
    pub url: String,
    /// Expiry (unix seconds). The [`HttpPresignClient`] cache honours this; a URL already past it is
    /// rejected as [`SwarmNetError::PresignExpired`].
    pub expires_at: u64,
    /// Signed headers the caller must replay verbatim on the object request (e.g. a
    /// signature-covered `content-type`). Empty when the presign signed only the URL query.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

/// The coordinator presign seam (spec §11.1). Frozen at Merge 1; BC implements the HTTP side.
#[async_trait]
pub trait PresignClient: Send + Sync {
    /// Presign one object for `run`. Returns the URL + its expiry (+ any signed headers to replay).
    async fn presign(
        &self,
        run: &RunId,
        req: &PresignRequest,
    ) -> Result<PresignResponse, SwarmNetError>;
}

/// The cache key: a run + the fully-qualified object request. `ObjectKind`/`PresignOp` are `Hash`.
#[derive(Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    run: String,
    kind: ObjectKind,
    op: PresignOp,
    round: Option<u64>,
    peer: Option<String>,
    path: Option<String>,
}

impl CacheKey {
    fn of(run: &RunId, req: &PresignRequest) -> Self {
        Self {
            run: run.as_str().to_string(),
            kind: req.kind,
            op: req.op,
            round: req.round,
            peer: req.peer.clone(),
            path: req.path.clone(),
        }
    }
}

/// Current wall-clock in unix seconds (presign expiry is a coarse, wall-clock quantity).
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The HTTP [`PresignClient`]: `POST <coordinator_base>/runs/:id/presign` through [`EgressClient`].
///
/// `coordinator_base` is a swarm-allowlisted coordinator endpoint (spec §11.1), e.g.
/// `https://api.daemon.ai/api/v1/swarm`. An optional bearer token carries the `swarm:*` API-key scope.
pub struct HttpPresignClient {
    egress: EgressClient,
    coordinator_base: String,
    bearer: Option<String>,
    /// The internal identity headers (`x-daemon-org-id` / `x-daemon-actor`) for the
    /// direct-to-`apps/swarm` dev path (A3, additive — mirrors `RegistryClient::with_internal`).
    internal: Option<(String, String)>,
    /// Clock-skew safety margin (seconds): a cached URL is reused only while `expires_at > now + margin`.
    skew_margin_s: u64,
    cache: Mutex<HashMap<CacheKey, PresignResponse>>,
}

impl HttpPresignClient {
    /// Build a client against `coordinator_base` (a trailing `/` is trimmed).
    pub fn new(egress: EgressClient, coordinator_base: impl Into<String>) -> Self {
        Self {
            egress,
            coordinator_base: coordinator_base.into().trim_end_matches('/').to_string(),
            bearer: None,
            internal: None,
            skew_margin_s: 5,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Attach the `swarm:*`-scoped API-key bearer token (sent on every presign request).
    #[must_use]
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Attach the internal identity headers (the direct-to-`apps/swarm` dev path; A3, additive —
    /// mirrors [`crate::RegistryClient::with_internal`]). Never hardcoded: sourced from
    /// `JoinRun.credentials` / node config.
    #[must_use]
    pub fn with_internal(mut self, org_id: impl Into<String>, actor: impl Into<String>) -> Self {
        self.internal = Some((org_id.into(), actor.into()));
        self
    }

    /// The presign endpoint URL for `run`.
    fn endpoint(&self, run: &RunId) -> String {
        format!("{}/runs/{}/presign", self.coordinator_base, run.as_str())
    }

    /// A live cache hit for `key`, if one is present and comfortably unexpired.
    fn cached(&self, key: &CacheKey) -> Option<PresignResponse> {
        let cache = self.cache.lock().expect("presign cache mutex");
        cache.get(key).and_then(|resp| {
            (resp.expires_at > now_unix() + self.skew_margin_s).then(|| resp.clone())
        })
    }
}

#[async_trait]
impl PresignClient for HttpPresignClient {
    async fn presign(
        &self,
        run: &RunId,
        req: &PresignRequest,
    ) -> Result<PresignResponse, SwarmNetError> {
        let cache_key = CacheKey::of(run, req);
        if let Some(hit) = self.cached(&cache_key) {
            return Ok(hit);
        }

        let mut ereq = EgressRequest::post_json(self.endpoint(run), req)
            .map_err(|e| SwarmNetError::Transport(format!("encode presign request: {e}")))?;
        if let Some(token) = &self.bearer {
            ereq = ereq.bearer_auth(token);
        }
        if let Some((org_id, actor)) = &self.internal {
            ereq = ereq
                .header("x-daemon-org-id", org_id)
                .header("x-daemon-actor", actor);
        }
        let resp = self
            .egress
            .execute(ereq, Redirects::DEFAULT)
            .await
            .map_err(|e| SwarmNetError::Transport(format!("presign request: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| SwarmNetError::Transport(format!("read presign body: {e}")))?;
        if !status.is_success() {
            return Err(SwarmNetError::Transport(format!(
                "presign endpoint returned {status}: {}",
                String::from_utf8_lossy(&body)
            )));
        }
        let presigned: PresignResponse = serde_json::from_slice(&body)
            .map_err(|e| SwarmNetError::Transport(format!("decode presign response: {e}")))?;

        // A URL that is already expired is a fault (clock skew / a misconfigured coordinator): the
        // credential is unusable, so reject rather than hand back a dead URL. Not a `PayloadMiss`.
        if presigned.expires_at <= now_unix() {
            return Err(SwarmNetError::PresignExpired(format!(
                "{}/{:?} expires_at={} <= now",
                run.as_str(),
                req.kind,
                presigned.expires_at
            )));
        }
        // Cache only if it will outlive the skew margin (so `cached()` can always reuse it).
        if presigned.expires_at > now_unix() + self.skew_margin_s {
            self.cache
                .lock()
                .expect("presign cache mutex")
                .insert(cache_key, presigned.clone());
        }
        Ok(presigned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_json_omits_unset_fields() {
        // A payload request carries round+peer, never `path`.
        let req = PresignRequest::payload(PresignOp::Put, 7, "aabb");
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["kind"], "payload");
        assert_eq!(v["op"], "put");
        assert_eq!(v["round"], 7);
        assert_eq!(v["peer"], "aabb");
        assert!(v.get("path").is_none(), "path omitted for payload");

        // An artifact request carries only `path`.
        let art = PresignRequest::artifact(PresignOp::Get, "runs/r/experiment.wasm");
        let av: serde_json::Value = serde_json::to_value(&art).unwrap();
        assert_eq!(av["kind"], "artifact");
        assert_eq!(av["path"], "runs/r/experiment.wasm");
        assert!(av.get("round").is_none(), "round omitted for artifact");
        assert!(av.get("peer").is_none(), "peer omitted for artifact");
    }

    #[test]
    fn kind_wire_tokens_are_kebab_case() {
        assert_eq!(
            serde_json::to_value(ObjectKind::RecordSet).unwrap(),
            serde_json::Value::from("record-set")
        );
        assert_eq!(
            serde_json::to_value(ObjectKind::Checkpoint).unwrap(),
            serde_json::Value::from("checkpoint")
        );
    }

    #[test]
    fn response_headers_default_to_empty() {
        let resp: PresignResponse =
            serde_json::from_str(r#"{"url":"https://x/obj","expires_at":123}"#).unwrap();
        assert!(resp.headers.is_empty());
    }

    /// The frozen node↔cloud HTTP contract (program Risk 6): every `tests/fixtures/presign-*.json`
    /// round-trips through the DTOs byte-for-structure (order-independent `Value` equality), so BC's
    /// worker and B3's live client can pin these bytes.
    #[test]
    fn presign_fixture_contract() {
        fn round_trips_request(fixture: &str) {
            let parsed: PresignRequest =
                serde_json::from_str(fixture).expect("fixture parses as PresignRequest");
            let reser = serde_json::to_value(&parsed).unwrap();
            let original: serde_json::Value = serde_json::from_str(fixture).unwrap();
            assert_eq!(reser, original, "DTO must reproduce the fixture exactly");
        }
        round_trips_request(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/presign-request-payload-put.json"
        )));
        round_trips_request(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/presign-request-payload-get.json"
        )));
        round_trips_request(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/presign-request-record-set-get.json"
        )));
        round_trips_request(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/presign-request-checkpoint-put.json"
        )));
        round_trips_request(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/presign-request-artifact-get.json"
        )));

        let resp_fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/presign-response.json"
        ));
        let resp: PresignResponse =
            serde_json::from_str(resp_fixture).expect("fixture parses as PresignResponse");
        assert!(resp.expires_at > 0);
        assert_eq!(
            resp.headers.get("content-type").map(String::as_str),
            Some("application/octet-stream")
        );
        let reser = serde_json::to_value(&resp).unwrap();
        let original: serde_json::Value = serde_json::from_str(resp_fixture).unwrap();
        assert_eq!(reser, original);
    }

    /// The URL cache presigns once per (run, object, op) while the URL is unexpired: a second call
    /// for the same object is served from cache, so the coordinator sees exactly one POST.
    #[tokio::test]
    async fn presign_cache_reuses_within_expiry() {
        use daemon_egress::{EgressClient, EgressConfig};
        use wiremock::matchers::{method, path_regex};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = PresignResponse {
            url: format!("{}/obj/x", server.uri()),
            expires_at: now_unix() + 900,
            headers: BTreeMap::new(),
        };
        Mock::given(method("POST"))
            .and(path_regex(r"/presign$"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::to_value(&resp).unwrap()),
            )
            .expect(1) // exactly one POST — the second presign must be a cache hit
            .mount(&server)
            .await;

        let egress = EgressClient::new(EgressConfig::default()).unwrap();
        let client = HttpPresignClient::new(egress, format!("{}/api/v1/swarm", server.uri()));
        let run = RunId::new("run-c");
        let req = PresignRequest::payload(PresignOp::Get, 1, "aabb");
        let a = client.presign(&run, &req).await.unwrap();
        let b = client.presign(&run, &req).await.unwrap();
        assert_eq!(a.url, b.url);
        // `server` drop verifies the `.expect(1)`.
    }
}
