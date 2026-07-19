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
        // 404 (never stored / lifecycle-expired) and 403 (SignatureExpired at the object store) are
        // the availability misses the stall ladder consumes — not hard faults.
        if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::FORBIDDEN {
            return Ok(None);
        }
        Err(VhcNetError::Transport(format!(
            "presigned GET {} returned {status}",
            resp.url
        )))
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
            return Err(VhcNetError::Transport(format!(
                "presigned PUT {} returned {status}",
                resp.url
            )));
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
        let req = PresignRequest::artifact(PresignOp::Put, content_rel(&hash));
        let resp = self.presign.presign(&self.run, &req).await?;
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
        if !status.is_success() {
            return Err(VhcNetError::Transport(format!(
                "presigned content PUT {} returned {status}",
                resp.url
            )));
        }
        Ok(hash)
    }

    async fn get_content(&self, hash: &ContentHash) -> Result<Vec<u8>, VhcNetError> {
        let req = PresignRequest::artifact(PresignOp::Get, content_rel(hash));
        let resp = self.presign.presign(&self.run, &req).await?;
        let bytes = self
            .get_object(&resp)
            .await?
            .ok_or_else(|| VhcNetError::PayloadMiss(hash.to_hex()))?;
        // The requested address is not always the object's plain blake3: a chunk-addressed corpus
        // shard is keyed by its domain-separated CHUNK FOLD, which never equals `blake3(bytes)`.
        // The PUMP verifies every fetched object (plain hash for payloads/checkpoints/whole
        // artifacts, covering-chunk hashes for chunk-addressed shards), so this seat serves the
        // keyed bytes verbatim — a plain-hash gate here would wrongly reject every chunk-addressed
        // range fetch (the store is untrusted by construction; the pump is the arbiter).
        Ok(bytes)
    }
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

/// Map an [`EgressError`](daemon_egress::EgressError) onto a transport error.
fn transport(e: daemon_egress::EgressError) -> VhcNetError {
    VhcNetError::Transport(format!("egress: {e}"))
}

/// Map a response-body read failure (`reqwest::Error`) onto a transport error.
fn read_body(e: reqwest::Error) -> VhcNetError {
    VhcNetError::Transport(format!("read object body: {e}"))
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
