// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`R2Store`] — a [`PayloadStore`] over presigned R2/S3 URLs (spec §7.1, §11.1, §11.3).
//!
//! The `r2` baseline payload plane: the node never holds S3 credentials — it asks the coordinator
//! ([`PresignClient`]) for a short-lived presigned URL per op, then moves the bytes through the
//! SSRF-safe [`EgressClient`] (raw `reqwest::Client` is clippy-banned outside `daemon-egress`).
//!
//! - **`put`** → presigned `PUT`, returns `blake3(bytes)`.
//! - **`get`** → presigned `GET` + blake3-verify against the commitment's hash (reuses the frozen
//!   mismatch reject path); a 404/403 from the object store is the typed
//!   [`VhcNetError::PayloadMiss`] the §6.4 stall ladder consumes (matches the `FsPayloadStore`
//!   taxonomy, NET-8).
//! - **`head`** → presigned `GET` + hash the body (see the type doc: an R2 `HEAD` cannot yield the
//!   blake3 `PayloadStat` needs, and the trait's `head` takes no expected hash — so we re-fetch and
//!   hash, exactly like `FsPayloadStore::head`). Feeds the coordinator-seat receipt producer,
//!   which works unchanged over `R2Store` (NET-1 `head_emits_signed_receipt`).
//!
//! Object keys are the authoritative §11.3 layout, produced by [`r2_object_key`] (the coordinator
//! mints its presigned URLs at the same keys).

use async_trait::async_trait;
use daemon_egress::{EgressClient, EgressRequest, Redirects};
use daemon_vhc_proto::blake3_hash;

use crate::presign::{ObjectKind, PresignClient, PresignOp, PresignRequest, PresignResponse};
use crate::seam::{ContentHash, PayloadKey, RunId};
use crate::transport::{PayloadStat, PayloadStore};
use crate::VhcNetError;

/// The R2 object key for one presign request, per the spec §11.3 layout. The coordinator (BC) mints
/// its presigned URLs at exactly these keys, so this is the single source of truth both sides share.
///
/// - `payload`     → `runs/<run>/rounds/<round>/<peer_hex>.upd`
/// - `record-set`  → `runs/<run>/rounds/<round>/record-set.cbor`
/// - `checkpoint`  → `runs/<run>/checkpoints/round-<round>.safetensors`
/// - `artifact`    → `runs/<run>/<path>`
pub fn r2_object_key(run: &RunId, req: &PresignRequest) -> Result<String, VhcNetError> {
    let run = run.as_str();
    match req.kind {
        ObjectKind::Payload => {
            let round = req.round.ok_or_else(|| missing("payload", "round"))?;
            let peer = req
                .peer
                .as_deref()
                .ok_or_else(|| missing("payload", "peer"))?;
            Ok(format!("runs/{run}/rounds/{round}/{peer}.upd"))
        }
        ObjectKind::RecordSet => {
            let round = req.round.ok_or_else(|| missing("record-set", "round"))?;
            Ok(format!("runs/{run}/rounds/{round}/record-set.cbor"))
        }
        ObjectKind::Checkpoint => {
            let round = req.round.ok_or_else(|| missing("checkpoint", "round"))?;
            Ok(format!("runs/{run}/checkpoints/round-{round}.safetensors"))
        }
        ObjectKind::Artifact => {
            let path = req
                .path
                .as_deref()
                .ok_or_else(|| missing("artifact", "path"))?;
            Ok(format!("runs/{run}/{path}"))
        }
    }
}

fn missing(kind: &str, field: &str) -> VhcNetError {
    VhcNetError::Transport(format!("presign {kind} object requires `{field}`"))
}

/// A [`PayloadStore`] over presigned R2/S3 URLs (spec §7.1). Generic over the [`PresignClient`] so
/// the mock presign server (tests) and BC's real endpoint are drop-in.
pub struct R2Store<P: PresignClient> {
    presign: P,
    egress: EgressClient,
    run: RunId,
}

impl<P: PresignClient> R2Store<P> {
    /// Build a store for `run`, presigning through `presign` and moving bytes through `egress`.
    pub fn new(presign: P, egress: EgressClient, run: RunId) -> Self {
        Self {
            presign,
            egress,
            run,
        }
    }

    /// The run this store is scoped to.
    #[must_use]
    pub fn run(&self) -> &RunId {
        &self.run
    }

    /// **Live checkpoint-resync (§9; lane R).** Fetch the committed-set object bytes
    /// (`record-set.cbor`) the coordinator wrote for `round`
    /// (`runs/<run>/rounds/<round>/record-set.cbor`, spec §11.3). A rejoining peer uses this to
    /// learn which `(peer, hash, size)` payloads to stage when replaying a retained round forward
    /// from a checkpoint. Presigns a `record-set` GET (the frozen §11.1 contract) and fetches;
    /// a 404/403 → a typed [`VhcNetError::PayloadMiss`] (the stall-ladder signal — the caller
    /// falls back to fresh-state per §9). The bytes are returned OPAQUE: the record-set schema is
    /// SDK vocabulary, so decoding (and the per-payload blake3 verify, RUN-2) is the engine-side
    /// caller's job — net never interprets a consensus object.
    pub async fn fetch_record_set_bytes(
        &self,
        round: crate::seam::RoundId,
    ) -> Result<Vec<u8>, VhcNetError> {
        let req = PresignRequest::record_set(PresignOp::Get, round);
        let resp = self.presign.presign(&self.run, &req).await?;
        self.get_object(&resp).await?.ok_or_else(|| {
            VhcNetError::PayloadMiss(format!("{}@r{round}/record-set.cbor", self.run.as_str()))
        })
    }

    /// Presign one payload op for `key`.
    async fn presign_payload(
        &self,
        key: &PayloadKey,
        op: PresignOp,
    ) -> Result<PresignResponse, VhcNetError> {
        debug_assert_eq!(
            &key.run, &self.run,
            "payload key run must match the store run"
        );
        let req = PresignRequest::payload(op, key.round, key.peer.to_hex());
        self.presign.presign(&self.run, &req).await
    }

    /// One attempt of a content-addressed put: presign (`fresh` bypasses the presign cache — the
    /// expiry-retry lane, which must never re-serve a rejected credential) + presigned `PUT`.
    async fn put_content_once(
        &self,
        hash: &ContentHash,
        bytes: &[u8],
        fresh: bool,
    ) -> Result<(), VhcNetError> {
        let req = PresignRequest::artifact(PresignOp::Put, content_rel(hash));
        let resp = if fresh {
            self.presign.presign_fresh(&self.run, &req).await?
        } else {
            self.presign.presign(&self.run, &req).await?
        };
        let egress_resp = if resp.headers.is_empty() {
            self.egress
                .put(&resp.url, bytes.to_vec(), Redirects::None)
                .await
                .map_err(transport)?
        } else {
            let mut ereq = EgressRequest::put(&resp.url, bytes.to_vec());
            for (name, value) in &resp.headers {
                ereq = ereq.header(name, value);
            }
            self.egress
                .execute(ereq, Redirects::None)
                .await
                .map_err(transport)?
        };
        let status = egress_resp.status();
        if status == reqwest::StatusCode::FORBIDDEN {
            let body = egress_resp.bytes().await.unwrap_or_default();
            return Err(forbidden_error("PUT", &resp.url, &body));
        }
        if !status.is_success() {
            return Err(status_error(
                status,
                format!("presigned content PUT {} returned {status}", resp.url),
            ));
        }
        Ok(())
    }

    /// Issue a presigned `GET`, returning `Some(bytes)` on 2xx, `None` on a 404/403 miss, or a hard
    /// transport error otherwise. Signed headers (if any) are replayed verbatim.
    async fn get_object(&self, resp: &PresignResponse) -> Result<Option<Vec<u8>>, VhcNetError> {
        let egress_resp = if resp.headers.is_empty() {
            self.egress
                .get(&resp.url, Redirects::DEFAULT)
                .await
                .map_err(transport)?
        } else {
            let mut req = EgressRequest::get(&resp.url);
            for (name, value) in &resp.headers {
                req = req.header(name, value);
            }
            self.egress
                .execute(req, Redirects::DEFAULT)
                .await
                .map_err(transport)?
        };
        let status = egress_resp.status();
        if status.is_success() {
            let bytes = egress_resp.bytes().await.map_err(read_body)?;
            return Ok(Some(bytes.to_vec()));
        }
        // 404 (never stored / lifecycle-expired) is the availability miss the stall ladder
        // consumes — not a hard fault.
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        // 403 is NOT a miss — the object may well exist (REL-2, reliability spec §3). A
        // recognized credential-expiry body is the typed re-presign lane; any other 403 is an
        // authoritative authorization refusal retrying cannot change.
        if status == reqwest::StatusCode::FORBIDDEN {
            let body = egress_resp.bytes().await.unwrap_or_default();
            return Err(forbidden_error("GET", &resp.url, &body));
        }
        Err(status_error(
            status,
            format!("presigned GET {} returned {status}", resp.url),
        ))
    }

    /// One attempt of a content-addressed get: presign (`fresh` bypasses the presign cache — the
    /// expiry-retry lane) + presigned `GET`; an absent object is the typed miss.
    async fn get_content_once(
        &self,
        hash: &ContentHash,
        fresh: bool,
    ) -> Result<Vec<u8>, VhcNetError> {
        let req = PresignRequest::artifact(PresignOp::Get, content_rel(hash));
        let resp = if fresh {
            self.presign.presign_fresh(&self.run, &req).await?
        } else {
            self.presign.presign(&self.run, &req).await?
        };
        self.get_object(&resp)
            .await?
            .ok_or_else(|| VhcNetError::PayloadMiss(hash.to_hex()))
    }
}

#[async_trait]
impl<P: PresignClient> PayloadStore for R2Store<P> {
    async fn put(&self, key: &PayloadKey, bytes: &[u8]) -> Result<ContentHash, VhcNetError> {
        let resp = self.presign_payload(key, PresignOp::Put).await?;
        // No forced Content-Type: a presigned PUT only validates the headers it was minted with
        // (SigV4 parity — initial `EgressClient::put`). Replay any the presign *did* sign.
        let egress_resp = if resp.headers.is_empty() {
            self.egress
                .put(&resp.url, bytes.to_vec(), Redirects::None)
                .await
                .map_err(transport)?
        } else {
            let mut req = EgressRequest::put(&resp.url, bytes.to_vec());
            for (name, value) in &resp.headers {
                req = req.header(name, value);
            }
            self.egress
                .execute(req, Redirects::None)
                .await
                .map_err(transport)?
        };
        let status = egress_resp.status();
        if !status.is_success() {
            return Err(status_error(
                status,
                format!("presigned PUT {} returned {status}", resp.url),
            ));
        }
        Ok(blake3_hash(bytes))
    }

    async fn get(&self, key: &PayloadKey, expected: &ContentHash) -> Result<Vec<u8>, VhcNetError> {
        let resp = self.presign_payload(key, PresignOp::Get).await?;
        let bytes = self.get_object(&resp).await?.ok_or_else(|| miss(key))?;
        let actual = blake3_hash(&bytes);
        if &actual != expected {
            return Err(VhcNetError::HashMismatch {
                expected: expected.to_hex(),
                actual: actual.to_hex(),
            });
        }
        Ok(bytes)
    }

    async fn head(&self, key: &PayloadKey) -> Result<PayloadStat, VhcNetError> {
        // A network HEAD cannot produce the blake3 `PayloadStat.hash` (R2 exposes only size + an
        // etag/md5), and the trait's `head` takes no expected hash — so we re-fetch and hash, exactly
        // like `FsPayloadStore::head` re-reads to attest the content hash (store.rs).
        let resp = self.presign_payload(key, PresignOp::Get).await?;
        let bytes = self.get_object(&resp).await?.ok_or_else(|| miss(key))?;
        Ok(PayloadStat {
            hash: blake3_hash(&bytes),
            size: bytes.len() as u64,
        })
    }
}

/// The run-relative artifact path of one content-addressed payload object (ABI §12.6 [RS-4]):
/// `payload/<blake3-hex>`, presigned through the frozen **artifact** presign form
/// (`runs/<run>/payload/<hex>` at the bucket — the `runs/<run>/` prefix is the store's own
/// namespace/auth scoping from the per-run presign endpoint, never a module-visible coordinate).
fn content_rel(hash: &ContentHash) -> String {
    format!("payload/{}", hash.to_hex())
}

/// The content-addressed seam over the SAME presigned remote store: the module names content,
/// this impl moves bytes. This is the production payload plane a role session binds
/// ([`crate::transport::ContentStore`]); the coordinate-keyed [`PayloadStore`] impl above
/// predates it and survives for the harness-era consumers.
#[async_trait]
impl<P: PresignClient> crate::transport::ContentStore for R2Store<P> {
    async fn put_content(&self, bytes: &[u8]) -> Result<ContentHash, VhcNetError> {
        let hash = blake3_hash(bytes);
        // A content-addressed put is idempotent by construction, and this is the WAN payload
        // plane: a transient egress fault (reset upload, gateway 5xx) must not surface as a hard
        // completion failure — a trainer treats a failed commit put as fatal (fail loud) and one
        // dropped multi-hundred-MB upload would kill an otherwise healthy round. Bounded retries
        // with a fresh presign per attempt (the previous URL may have aged out during a slow
        // upload); a non-transient refusal (a 4xx status) still fails fast and loud.
        let mut expiry_retried = false;
        let mut fresh = false;
        let mut last: Option<VhcNetError> = None;
        let mut attempt = 0;
        while attempt < PUT_ATTEMPTS {
            if attempt > 0 && !fresh {
                let backoff = PUT_BACKOFF_BASE * 2u32.pow(attempt - 1);
                tracing::warn!(
                    hash = %hash.to_hex(),
                    attempt,
                    error = %last.as_ref().map_or_else(String::new, ToString::to_string),
                    "content put failed transiently; retrying after {backoff:?}"
                );
                tokio::time::sleep(backoff).await;
            }
            match self
                .put_content_once(&hash, bytes, std::mem::take(&mut fresh))
                .await
            {
                Ok(()) => return Ok(hash),
                // The object store rejected the credential as expired while the local cache may
                // still consider it live: one immediate retry on a guaranteed-fresh presign,
                // outside the transient budget. A second expiry is authoritative.
                Err(VhcNetError::PresignExpired(_)) if !expiry_retried => {
                    expiry_retried = true;
                    fresh = true;
                }
                Err(e) if is_transient(&e) => {
                    last = Some(e);
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last.expect("at least one attempt ran"))
    }

    async fn get_content(&self, hash: &ContentHash) -> Result<Vec<u8>, VhcNetError> {
        // GET-side absorption (REL-2, reliability spec §3): a transient egress fault on a
        // committed-payload read must not surface as a hard completion failure — the C2 evidence
        // class is a single mid-body TCP reset, for which bounded whole-object retry is the
        // right first knob (spec §2.1). A genuine miss (404) never retries: retention truth is
        // the stall ladder's to consume, immediately.
        let mut expiry_retried = false;
        let mut fresh = false;
        let mut attempt = 0;
        loop {
            match self
                .get_content_once(hash, std::mem::take(&mut fresh))
                .await
            {
                // Served verbatim: the requested address is not always the object's plain blake3
                // (a chunk-addressed corpus shard is keyed by its domain-separated CHUNK FOLD),
                // so verification is the PUMP's job — plain hash for payloads/checkpoints/whole
                // artifacts, covering-chunk hashes for shards. A plain-hash gate here would
                // wrongly reject every chunk-addressed range fetch (the store is untrusted by
                // construction; the pump is the arbiter).
                Ok(bytes) => return Ok(bytes),
                // The expiry-retry lane: one immediate re-fetch on a guaranteed-fresh presign
                // (cache bypassed — `cached()` could re-serve exactly the rejected URL), outside
                // the transient budget. A second expiry is authoritative (a misconfigured
                // coordinator), never a silent loop.
                Err(VhcNetError::PresignExpired(_)) if !expiry_retried => {
                    expiry_retried = true;
                    fresh = true;
                }
                Err(e) if is_transient(&e) => {
                    attempt += 1;
                    if attempt >= GET_ATTEMPTS {
                        return Err(e);
                    }
                    let backoff = GET_BACKOFF_BASE * 2u32.pow(attempt - 1);
                    tracing::warn!(
                        hash = %hash.to_hex(),
                        attempt,
                        error = %e,
                        "content get failed transiently; retrying after {backoff:?}"
                    );
                    tokio::time::sleep(backoff).await;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// How many times a content put is attempted before its transport fault becomes the caller's.
const PUT_ATTEMPTS: u32 = 4;

/// The first retry backoff; doubles per attempt (2 s, 4 s, 8 s).
const PUT_BACKOFF_BASE: std::time::Duration = std::time::Duration::from_secs(2);

/// How many times a content get is attempted before its transport fault becomes the caller's.
const GET_ATTEMPTS: u32 = 4;

/// The first GET retry backoff; doubles per attempt (1 s, 2 s, 4 s). Tuned from the C2 evidence
/// (reliability spec §2.1): the observed fault class is a single mid-body reset with no
/// saturation signature, so short flat-start pacing recovers it without stacking meaningful
/// latency onto the blocked guest completion.
const GET_BACKOFF_BASE: std::time::Duration = std::time::Duration::from_secs(1);

/// Whether an object-store `403` body carries a credential-expiry shape. Conservative on the
/// S3-compatible error vocabulary an aged presigned URL produces (`Request has expired` under
/// `AccessDenied`, `ExpiredToken`/`ExpiredRequest` codes); everything else stays an
/// authoritative refusal. The exact bodies R2 returns — and whether it enforces expiry at
/// request admission or mid-transfer — are RQ-1's empirical probe (reliability spec §16); C2's
/// record contains no expiry-shaped fault, so this lane is precautionary correctness.
fn body_is_expiry(body: &[u8]) -> bool {
    let text = String::from_utf8_lossy(body);
    ["Request has expired", "ExpiredToken", "ExpiredRequest"]
        .iter()
        .any(|marker| text.contains(marker))
}

/// Type an object-store `403`: a recognized credential-expiry body is the typed
/// [`VhcNetError::PresignExpired`] the re-presign lane consumes; anything else is an
/// authoritative semantic refusal (never a miss — the object may well exist).
fn forbidden_error(op: &str, url: &str, body: &[u8]) -> VhcNetError {
    let snippet: String = String::from_utf8_lossy(body).chars().take(200).collect();
    if body_is_expiry(body) {
        VhcNetError::PresignExpired(format!(
            "presigned {op} {url} rejected by the object store as expired: {snippet}"
        ))
    } else {
        VhcNetError::Transport(format!(
            "presigned {op} {url} returned 403 Forbidden: {snippet}"
        ))
    }
}

/// Whether a put fault is worth retrying in-attempt: exactly the TYPED transient lane
/// (egress send faults, gateway 5xx — [`VhcNetError::Transient`]). A semantic
/// [`VhcNetError::Transport`] (an authoritative 4xx: bad signature, missing object prefix,
/// over-size) is a refusal retrying cannot change.
fn is_transient(e: &VhcNetError) -> bool {
    e.is_transient_transport()
}

/// A typed availability miss for `key` (the stall-ladder signal; mirrors `store.rs`'s taxonomy).
fn miss(key: &PayloadKey) -> VhcNetError {
    VhcNetError::PayloadMiss(format!(
        "{}@r{}/{}",
        key.run.as_str(),
        key.round,
        key.peer.to_hex()
    ))
}

/// Map an [`EgressError`](daemon_egress::EgressError) onto the typed taxonomy: send-level
/// transport faults stay typed transient; policy/encode refusals stay semantic.
fn transport(e: daemon_egress::EgressError) -> VhcNetError {
    crate::classify_egress(&e, "egress")
}

/// Map a response-body read failure (`reqwest::Error`) onto the typed taxonomy — a body that
/// dies mid-read is a reset-class transient fault (the request itself already succeeded).
fn read_body(e: reqwest::Error) -> VhcNetError {
    VhcNetError::Transient {
        kind: crate::TransportFaultKind::Reset,
        detail: format!("read object body: {e}"),
    }
}

/// Map a non-success HTTP status onto the typed taxonomy: `5xx`/`429` is a typed transient
/// server fault; any other status is an authoritative semantic refusal.
fn status_error(status: reqwest::StatusCode, detail: String) -> VhcNetError {
    if crate::status_is_transient(status) {
        VhcNetError::Transient {
            kind: crate::TransportFaultKind::ServerFault,
            detail,
        }
    } else {
        VhcNetError::Transport(detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::{PeerId, RoundId};

    fn run() -> RunId {
        RunId::new("run-x")
    }

    #[test]
    fn object_keys_match_spec_11_3() {
        let r = run();
        let peer = PeerId([0xAB; 32]);
        assert_eq!(
            r2_object_key(
                &r,
                &PresignRequest::payload(PresignOp::Put, 7, peer.to_hex())
            )
            .unwrap(),
            format!("runs/run-x/rounds/7/{}.upd", peer.to_hex())
        );
        assert_eq!(
            r2_object_key(&r, &PresignRequest::record_set(PresignOp::Get, 7)).unwrap(),
            "runs/run-x/rounds/7/record-set.cbor"
        );
        assert_eq!(
            r2_object_key(&r, &PresignRequest::checkpoint(PresignOp::Put, 3)).unwrap(),
            "runs/run-x/checkpoints/round-3.safetensors"
        );
        assert_eq!(
            r2_object_key(
                &r,
                &PresignRequest::artifact(PresignOp::Get, "experiment.wasm")
            )
            .unwrap(),
            "runs/run-x/experiment.wasm"
        );
    }

    #[test]
    fn missing_required_field_is_typed_error() {
        let r = run();
        // A payload request with no peer is malformed.
        let bad = PresignRequest {
            kind: ObjectKind::Payload,
            op: PresignOp::Get,
            round: Some(1),
            peer: None,
            path: None,
        };
        assert!(matches!(
            r2_object_key(&r, &bad),
            Err(VhcNetError::Transport(_))
        ));
    }

    /// A helper to bind the type parameter — proves `RoundId` is the seam type the key uses.
    #[allow(dead_code)]
    fn _round_type(_: RoundId) {}

    // --- NET-1 / NET-8: R2Store over the mock presign + object server ----------------------------

    use crate::fetch::{fetch_with_fallback_dyn, RetryPolicy};
    use crate::mock_r2::MockR2;
    use crate::store::FsPayloadStore;
    use crate::test_support::temp_root;
    use daemon_vhc_proto::blake3_hash;

    fn pkey(round: RoundId, peer: u8) -> PayloadKey {
        PayloadKey::new(RunId::new("run-x"), round, PeerId([peer; 32]))
    }

    fn store_over(mock: &MockR2) -> R2Store<crate::presign::HttpPresignClient> {
        R2Store::new(mock.presign_client(), mock.egress(), RunId::new("run-x"))
    }

    /// NET-1: PUT then GET round-trips the bytes through presigned URLs (hash-verified on GET).
    #[tokio::test]
    async fn store_presign_roundtrip() {
        let mock = MockR2::start().await;
        let store = store_over(&mock);
        let k = pkey(3, 0x11);

        let hash = store.put(&k, b"update-bytes").await.unwrap();
        assert_eq!(hash, blake3_hash(b"update-bytes"));
        let got = store.get(&k, &hash).await.unwrap();
        assert_eq!(got, b"update-bytes");
    }

    /// NET-1: a presigned URL already past `expires_at` is rejected (not treated as a miss).
    #[tokio::test]
    async fn store_presign_expired_rejected() {
        // Every presign this mock mints is already 60s expired.
        let mock = MockR2::with_expiry(-60).await;
        let store = store_over(&mock);
        let k = pkey(3, 0x12);
        let err = store.put(&k, b"x").await.unwrap_err();
        assert!(matches!(err, VhcNetError::PresignExpired(_)), "got {err:?}");
    }

    /// NET-1: HEAD over the presign plane attests the stored object's `(hash, size)` — the stat
    /// the coordinator-seat receipt producer folds into signed availability evidence (that
    /// producer lives with the coordinator harness drive; here we pin the R2 stat half).
    #[tokio::test]
    async fn head_attests_hash_and_size() {
        let mock = MockR2::start().await;
        let store = store_over(&mock);
        let k = pkey(2, 0x01);
        let hash = store.put(&k, b"peer-update").await.unwrap();

        let stat = store.head(&k).await.unwrap();
        assert_eq!(stat.hash, hash);
        assert_eq!(stat.size, b"peer-update".len() as u64);
    }

    /// NET-8: an object within the retention window is fetchable.
    #[tokio::test]
    async fn retained_object_fetchable() {
        let mock = MockR2::start().await;
        let store = store_over(&mock);
        let k = pkey(4, 0x44);
        let hash = store.put(&k, b"recent").await.unwrap();
        assert_eq!(store.get(&k, &hash).await.unwrap(), b"recent");
        // HEAD attests it too.
        let stat = store.head(&k).await.unwrap();
        assert_eq!(stat.hash, hash);
        assert_eq!(stat.size, 6);
    }

    /// NET-8: a lifecycle-expired (evicted) object is a typed [`VhcNetError::PayloadMiss`] — the
    /// stall-ladder signal.
    #[tokio::test]
    async fn expired_object_typed_miss() {
        let mock = MockR2::start().await;
        let store = store_over(&mock);
        let k = pkey(0, 0x55);
        let hash = store.put(&k, b"stale").await.unwrap();
        // Retention pruned it server-side.
        mock.evict(&format!("runs/run-x/rounds/0/{}.upd", k.peer.to_hex()));

        let err = store.get(&k, &hash).await.unwrap_err();
        assert!(matches!(err, VhcNetError::PayloadMiss(_)), "got {err:?}");
    }

    /// NET-1: a GET whose bytes do not match the commitment hash is a tamper reject.
    #[tokio::test]
    async fn get_rejects_hash_mismatch() {
        let mock = MockR2::start().await;
        let store = store_over(&mock);
        let k = pkey(1, 0x66);
        store.put(&k, b"honest").await.unwrap();
        let err = store.get(&k, &blake3_hash(b"different")).await.unwrap_err();
        assert!(
            matches!(err, VhcNetError::HashMismatch { .. }),
            "got {err:?}"
        );
    }

    // --- the content-addressed seam (ABI §12.6 [RS-4]) over the same presigned store ------------

    use crate::transport::ContentStore as _;

    /// Content put/get round-trips by blake3 address over the presigned artifact form
    /// (`runs/<run>/payload/<hex>` at the bucket); a re-put of identical bytes is idempotent.
    #[tokio::test]
    async fn content_presign_roundtrip_is_idempotent() {
        let mock = MockR2::start().await;
        let store = store_over(&mock);

        let hash = store.put_content(b"sealed-object").await.unwrap();
        assert_eq!(hash, blake3_hash(b"sealed-object"));
        assert_eq!(store.get_content(&hash).await.unwrap(), b"sealed-object");
        assert_eq!(store.put_content(b"sealed-object").await.unwrap(), hash);
    }

    /// A never-stored / evicted content object is a typed miss keyed by its hex address.
    #[tokio::test]
    async fn content_miss_is_typed() {
        let mock = MockR2::start().await;
        let store = store_over(&mock);
        let hash = store.put_content(b"short-lived").await.unwrap();
        mock.evict(&format!("runs/run-x/payload/{}", hash.to_hex()));

        let err = store.get_content(&hash).await.unwrap_err();
        assert!(matches!(err, VhcNetError::PayloadMiss(_)), "got {err:?}");
    }

    /// The content seam serves keyed bytes VERBATIM: its key is not always the object's plain
    /// blake3 (a chunk-addressed corpus shard is keyed by its CHUNK FOLD), so verification is the
    /// pump's job (plain hash for payloads/whole artifacts, covering-chunk hashes for shards). A
    /// plain-hash gate here would reject every chunk-addressed range fetch — the store is untrusted
    /// by construction and the pump is the sole arbiter (matching `MemoryContentStore`).
    #[tokio::test]
    async fn get_content_serves_keyed_bytes_verbatim() {
        let mock = MockR2::start().await;
        let store = store_over(&mock);
        let hash = store.put_content(b"honest").await.unwrap();
        mock.corrupt(
            &format!("runs/run-x/payload/{}", hash.to_hex()),
            b"fold-keyed-bytes",
        );

        assert_eq!(
            store.get_content(&hash).await.unwrap(),
            b"fold-keyed-bytes",
            "the untrusted store serves the keyed bytes; the pump verifies"
        );
    }

    // --- Gate C: the typed transient transport lane at the object store --------------------------

    /// A presign stub that mints non-expiring URLs at a fixed base — lets a test point the
    /// store's object traffic anywhere (a `5xx` mock, a closed port).
    struct StaticPresign {
        base: String,
    }

    #[async_trait]
    impl PresignClient for StaticPresign {
        async fn presign(
            &self,
            _run: &RunId,
            req: &PresignRequest,
        ) -> Result<PresignResponse, VhcNetError> {
            Ok(PresignResponse {
                url: format!("{}/{}", self.base, r2_object_key(&run(), req).unwrap()),
                expires_at: u64::MAX,
                headers: std::collections::BTreeMap::new(),
            })
        }
    }

    /// Gate C: an object store answering `5xx` on a presigned GET is a TYPED transient server
    /// fault (the budget-free deferral lane) — never a semantic transport refusal and never a
    /// miss.
    #[tokio::test]
    async fn a_5xx_object_store_is_a_typed_transient_fault() {
        use daemon_egress::{EgressClient, EgressConfig};
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(502))
            .mount(&server)
            .await;
        let store = R2Store::new(
            StaticPresign { base: server.uri() },
            EgressClient::new(EgressConfig::default()).unwrap(),
            run(),
        );

        let err = store
            .get_content(&blake3_hash(b"whatever"))
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                VhcNetError::Transient {
                    kind: crate::TransportFaultKind::ServerFault,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    /// Gate C: a connect-refused object store (the R2 outage shape) is a TYPED transient
    /// connect fault, preserved from the egress client — the classification the defect-10
    /// strings erased.
    #[tokio::test]
    async fn a_connect_refused_object_store_is_a_typed_transient_fault() {
        use daemon_egress::{EgressClient, EgressConfig};

        // A port nothing listens on: the OS refuses the dial.
        let store = R2Store::new(
            StaticPresign {
                base: "http://127.0.0.1:9".to_string(),
            },
            EgressClient::new(EgressConfig::default()).unwrap(),
            run(),
        );

        let err = store
            .get_content(&blake3_hash(b"whatever"))
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                VhcNetError::Transient {
                    kind: crate::TransportFaultKind::Connect,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    // --- REL-2: GET-side absorption + the 403 taxonomy (reliability spec §3) ---------------------

    /// A presign stub that mints a DISTINCT URL per mint (`/mint-<n>/...`) and counts mints —
    /// makes the fresh-credential lane observable (a re-served cached URL would repeat `mint-0`).
    struct CountingPresign {
        base: String,
        mints: std::sync::atomic::AtomicU32,
    }

    impl CountingPresign {
        fn new(base: impl Into<String>) -> Self {
            Self {
                base: base.into(),
                mints: std::sync::atomic::AtomicU32::new(0),
            }
        }

        fn minted(&self) -> u32 {
            self.mints.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl PresignClient for &CountingPresign {
        async fn presign(
            &self,
            _run: &RunId,
            req: &PresignRequest,
        ) -> Result<PresignResponse, VhcNetError> {
            let n = self.mints.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(PresignResponse {
                url: format!(
                    "{}/mint-{n}/{}",
                    self.base,
                    r2_object_key(&run(), req).unwrap()
                ),
                expires_at: u64::MAX,
                headers: std::collections::BTreeMap::new(),
            })
        }
    }

    const EXPIRY_BODY: &str =
        "<Error><Code>AccessDenied</Code><Message>Request has expired</Message></Error>";

    /// REL-2: a transient server fault on a presigned GET is absorbed by bounded retry — the
    /// C2 mid-body-reset class must not surface as a hard completion failure.
    #[tokio::test]
    async fn get_content_retries_a_transient_server_fault() {
        use daemon_egress::{EgressClient, EgressConfig};
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(502))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"recovered".to_vec()))
            .mount(&server)
            .await;

        let presign = CountingPresign::new(server.uri());
        let store = R2Store::new(
            &presign,
            EgressClient::new(EgressConfig::default()).unwrap(),
            run(),
        );
        let got = store
            .get_content(&blake3_hash(b"whatever"))
            .await
            .expect("one 502 is absorbed");
        assert_eq!(got, b"recovered");
    }

    /// REL-2: a genuine miss (404) NEVER retries — retention truth belongs to the stall ladder,
    /// immediately. The mock's `expect(1)` verifies exactly one GET was issued.
    #[tokio::test]
    async fn get_content_miss_never_retries() {
        use daemon_egress::{EgressClient, EgressConfig};
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;

        let presign = CountingPresign::new(server.uri());
        let store = R2Store::new(
            &presign,
            EgressClient::new(EgressConfig::default()).unwrap(),
            run(),
        );
        let err = store
            .get_content(&blake3_hash(b"whatever"))
            .await
            .unwrap_err();
        assert!(matches!(err, VhcNetError::PayloadMiss(_)), "got {err:?}");
        assert_eq!(presign.minted(), 1, "a miss must not re-presign");
    }

    /// REL-2: a recognized expiry-shaped 403 re-fetches ONCE on a guaranteed-fresh credential
    /// (observable: the second GET rides a `mint-1` URL, so a re-served cached URL would fail).
    #[tokio::test]
    async fn get_content_expiry_403_refetches_on_fresh_presign() {
        use daemon_egress::{EgressClient, EgressConfig};
        use wiremock::matchers::{method, path_regex};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/mint-0/"))
            .respond_with(ResponseTemplate::new(403).set_body_string(EXPIRY_BODY))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/mint-1/"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"fresh".to_vec()))
            .expect(1)
            .mount(&server)
            .await;

        let presign = CountingPresign::new(server.uri());
        let store = R2Store::new(
            &presign,
            EgressClient::new(EgressConfig::default()).unwrap(),
            run(),
        );
        let got = store
            .get_content(&blake3_hash(b"whatever"))
            .await
            .expect("the fresh-credential re-fetch recovers");
        assert_eq!(got, b"fresh");
        assert_eq!(presign.minted(), 2, "exactly one fresh re-mint");
    }

    /// REL-2: a SECOND expiry-shaped 403 is authoritative — the lane never loops silently.
    #[tokio::test]
    async fn get_content_second_expiry_is_authoritative() {
        use daemon_egress::{EgressClient, EgressConfig};
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(403).set_body_string(EXPIRY_BODY))
            .expect(2)
            .mount(&server)
            .await;

        let presign = CountingPresign::new(server.uri());
        let store = R2Store::new(
            &presign,
            EgressClient::new(EgressConfig::default()).unwrap(),
            run(),
        );
        let err = store
            .get_content(&blake3_hash(b"whatever"))
            .await
            .unwrap_err();
        assert!(matches!(err, VhcNetError::PresignExpired(_)), "got {err:?}");
        assert_eq!(
            presign.minted(),
            2,
            "initial + one fresh re-mint, never more"
        );
    }

    /// REL-2: a 403 WITHOUT an expiry shape is an authoritative semantic refusal — never a miss
    /// (the object may well exist), never retried.
    #[tokio::test]
    async fn get_content_plain_403_is_semantic_not_miss() {
        use daemon_egress::{EgressClient, EgressConfig};
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(403)
                    .set_body_string("<Error><Code>AccessDenied</Code></Error>"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let presign = CountingPresign::new(server.uri());
        let store = R2Store::new(
            &presign,
            EgressClient::new(EgressConfig::default()).unwrap(),
            run(),
        );
        let err = store
            .get_content(&blake3_hash(b"whatever"))
            .await
            .unwrap_err();
        assert!(matches!(err, VhcNetError::Transport(_)), "got {err:?}");
    }

    /// REL-2 (the PUT loop shares the caveat): an expiry-shaped 403 on a presigned PUT re-puts
    /// ONCE on a guaranteed-fresh credential, outside the transient budget.
    #[tokio::test]
    async fn put_content_expiry_403_reputs_on_fresh_presign() {
        use daemon_egress::{EgressClient, EgressConfig};
        use wiremock::matchers::{method, path_regex};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path_regex("^/mint-0/"))
            .respond_with(ResponseTemplate::new(403).set_body_string(EXPIRY_BODY))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path_regex("^/mint-1/"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let presign = CountingPresign::new(server.uri());
        let store = R2Store::new(
            &presign,
            EgressClient::new(EgressConfig::default()).unwrap(),
            run(),
        );
        let hash = store
            .put_content(b"sealed-object")
            .await
            .expect("the fresh-credential re-put recovers");
        assert_eq!(hash, blake3_hash(b"sealed-object"));
        assert_eq!(presign.minted(), 2, "exactly one fresh re-mint");
    }

    /// NET-4: cross-store dyn fallback — an `R2Store` primary that misses falls through to an
    /// `FsPayloadStore` mirror that has the object.
    #[tokio::test]
    async fn dyn_fallback_r2_miss_to_fs() {
        let mock = MockR2::start().await;
        let r2 = store_over(&mock); // empty — every GET is a 404 miss
        let dir = temp_root("r2-dyn-fs");
        let fs = FsPayloadStore::open(dir.path(), 8).unwrap();
        let k = pkey(5, 0x77);
        let hash = fs.put(&k, b"mirrored").await.unwrap();

        let stores: [&dyn PayloadStore; 2] = [&r2, &fs];
        let got = fetch_with_fallback_dyn(&stores, &k, &hash, RetryPolicy::none())
            .await
            .unwrap();
        assert_eq!(got, b"mirrored");
    }
}
