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
//!   [`SwarmNetError::PayloadMiss`] the §6.4 stall ladder consumes (matches the `FsPayloadStore`
//!   taxonomy, NET-8).
//! - **`head`** → presigned `GET` + hash the body (see the type doc: an R2 `HEAD` cannot yield the
//!   blake3 `PayloadStat` needs, and the trait's `head` takes no expected hash — so we re-fetch and
//!   hash, exactly like `FsPayloadStore::head`). Feeds [`ReceiptProducer`](crate::ReceiptProducer),
//!   which works unchanged over `R2Store` (NET-1 `head_emits_signed_receipt`).
//!
//! Object keys are the authoritative §11.3 layout, produced by [`r2_object_key`] (the coordinator
//! mints its presigned URLs at the same keys).

use async_trait::async_trait;
use daemon_egress::{EgressClient, EgressRequest, Redirects};
use daemon_swarm_proto::blake3_hash;

use crate::presign::{ObjectKind, PresignClient, PresignOp, PresignRequest, PresignResponse};
use crate::seam::{ContentHash, PayloadKey, RunId};
use crate::transport::{PayloadStat, PayloadStore};
use crate::SwarmNetError;

/// The R2 object key for one presign request, per the spec §11.3 layout. The coordinator (BC) mints
/// its presigned URLs at exactly these keys, so this is the single source of truth both sides share.
///
/// - `payload`     → `runs/<run>/rounds/<round>/<peer_hex>.upd`
/// - `record-set`  → `runs/<run>/rounds/<round>/record-set.cbor`
/// - `checkpoint`  → `runs/<run>/checkpoints/round-<round>.safetensors`
/// - `artifact`    → `runs/<run>/<path>`
pub fn r2_object_key(run: &RunId, req: &PresignRequest) -> Result<String, SwarmNetError> {
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

fn missing(kind: &str, field: &str) -> SwarmNetError {
    SwarmNetError::Transport(format!("presign {kind} object requires `{field}`"))
}

/// A [`PayloadStore`] over presigned R2/S3 URLs (spec §7.1). Generic over the [`PresignClient`] so
/// the mock presign server (tests) and BC's real endpoint (Wave 3) are drop-in.
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

    /// Presign one payload op for `key`.
    async fn presign_payload(
        &self,
        key: &PayloadKey,
        op: PresignOp,
    ) -> Result<PresignResponse, SwarmNetError> {
        debug_assert_eq!(
            &key.run, &self.run,
            "payload key run must match the store run"
        );
        let req = PresignRequest::payload(op, key.round, key.peer.to_hex());
        self.presign.presign(&self.run, &req).await
    }

    /// Issue a presigned `GET`, returning `Some(bytes)` on 2xx, `None` on a 404/403 miss, or a hard
    /// transport error otherwise. Signed headers (if any) are replayed verbatim.
    async fn get_object(&self, resp: &PresignResponse) -> Result<Option<Vec<u8>>, SwarmNetError> {
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
        Err(SwarmNetError::Transport(format!(
            "presigned GET {} returned {status}",
            resp.url
        )))
    }
}

#[async_trait]
impl<P: PresignClient> PayloadStore for R2Store<P> {
    async fn put(&self, key: &PayloadKey, bytes: &[u8]) -> Result<ContentHash, SwarmNetError> {
        let resp = self.presign_payload(key, PresignOp::Put).await?;
        // No forced Content-Type: a presigned PUT only validates the headers it was minted with
        // (SigV4 parity — Wave-0 `EgressClient::put`). Replay any the presign *did* sign.
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
            return Err(SwarmNetError::Transport(format!(
                "presigned PUT {} returned {status}",
                resp.url
            )));
        }
        Ok(blake3_hash(bytes))
    }

    async fn get(
        &self,
        key: &PayloadKey,
        expected: &ContentHash,
    ) -> Result<Vec<u8>, SwarmNetError> {
        let resp = self.presign_payload(key, PresignOp::Get).await?;
        let bytes = self.get_object(&resp).await?.ok_or_else(|| miss(key))?;
        let actual = blake3_hash(&bytes);
        if &actual != expected {
            return Err(SwarmNetError::HashMismatch {
                expected: expected.to_hex(),
                actual: actual.to_hex(),
            });
        }
        Ok(bytes)
    }

    async fn head(&self, key: &PayloadKey) -> Result<PayloadStat, SwarmNetError> {
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

/// A typed availability miss for `key` (the stall-ladder signal; mirrors `store.rs`'s taxonomy).
fn miss(key: &PayloadKey) -> SwarmNetError {
    SwarmNetError::PayloadMiss(format!(
        "{}@r{}/{}",
        key.run.as_str(),
        key.round,
        key.peer.to_hex()
    ))
}

/// Map an [`EgressError`](daemon_egress::EgressError) onto a transport error.
fn transport(e: daemon_egress::EgressError) -> SwarmNetError {
    SwarmNetError::Transport(format!("egress: {e}"))
}

/// Map a response-body read failure (`reqwest::Error`) onto a transport error.
fn read_body(e: reqwest::Error) -> SwarmNetError {
    SwarmNetError::Transport(format!("read object body: {e}"))
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
            Err(SwarmNetError::Transport(_))
        ));
    }

    /// A helper to bind the type parameter — proves `RoundId` is the seam type the key uses.
    #[allow(dead_code)]
    fn _round_type(_: RoundId) {}

    // --- NET-1 / NET-8: R2Store over the mock presign + object server ----------------------------

    use crate::fetch::{fetch_with_fallback_dyn, RetryPolicy};
    use crate::mock_r2::MockR2;
    use crate::receipt::ReceiptProducer;
    use crate::store::FsPayloadStore;
    use crate::test_support::temp_root;
    use daemon_swarm_proto::{blake3_hash, SigningKey, SwarmMessage, SWARM_PROTO_VERSION};

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
        assert!(
            matches!(err, SwarmNetError::PresignExpired(_)),
            "got {err:?}"
        );
    }

    /// NET-1: `ReceiptProducer<R2Store>` compiles + works **unchanged** — HEAD → signed
    /// `StorageReceipt`.
    #[tokio::test]
    async fn head_emits_signed_receipt() {
        let mock = MockR2::start().await;
        let store = store_over(&mock);
        let k = pkey(2, 0x01);
        let hash = store.put(&k, b"peer-update").await.unwrap();

        let producer = ReceiptProducer::new(
            store,
            SigningKey::from_bytes(&[0x42; 32]),
            SWARM_PROTO_VERSION,
        );
        let signed = producer.produce(&k).await.unwrap();
        assert!(signed.verify().is_ok());

        let SwarmMessage::StorageReceipt(receipt) = &signed.payload else {
            panic!("expected StorageReceipt, got {:?}", signed.payload);
        };
        assert_eq!(receipt.round, 2);
        assert_eq!(receipt.verified.len(), 1);
        assert_eq!(receipt.verified[0].peer, k.peer);
        assert_eq!(receipt.verified[0].hash, hash);
        assert_eq!(receipt.verified[0].size, b"peer-update".len() as u64);
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

    /// NET-8: a lifecycle-expired (evicted) object is a typed [`SwarmNetError::PayloadMiss`] — the
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
        assert!(matches!(err, SwarmNetError::PayloadMiss(_)), "got {err:?}");
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
            matches!(err, SwarmNetError::HashMismatch { .. }),
            "got {err:?}"
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
