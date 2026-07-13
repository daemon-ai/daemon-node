// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Payload-plane fetch with retry + fallback locators (spec §7.1; TDD NET-4).
//!
//! A committed payload named by a `Commitment` may be fetchable from more than one place — the
//! presigned `r2` store key and/or an iroh-blobs ticket (`Locator`, §6.4). The ingest barrier
//! needs *some* verified copy; the retention floor (`payload_retention_rounds`, §7.4) guarantees
//! the bytes outlive a bounded retry. [`fetch_with_fallback`] encodes that policy: try the primary
//! store with bounded backoff, then fall through to alternate sources in cost order, returning the
//! first blake3-verified copy or a typed [`SwarmNetError::PayloadMiss`] once every source is
//! exhausted (the miss the §6.4 stall ladder consumes).
//!
//! [`fetch_with_fallback_dyn`] is the same policy over heterogeneous stores behind a trait object
//! (`&[&dyn PayloadStore]`) — an [`R2Store`](crate::R2Store) primary with an
//! [`FsPayloadStore`](crate::store::FsPayloadStore) mirror fallback (NET-4). [`fetch_record_set`]
//! fetches + decodes + content-verifies a `record-set.cbor` object (RUN-2 net half). The
//! [`DownloadScheduler`] is the concurrency + retry layer ported from Psyche's download scheduler
//! (capacity gate, FIFO waiters, per-class expo-backoff retry) — see the port note below.
//!
//! The alternate-locator hook (`BlobTicket` → iroh-blobs) slots behind the same shape once the
//! network plane lands (P4 / Wave 3).

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use daemon_swarm_proto::RecordSet;
use tokio::sync::{mpsc, oneshot};

use crate::seam::{ContentHash, PayloadKey};
use crate::transport::PayloadStore;
use crate::SwarmNetError;

/// Bounded retry/backoff policy for a single payload source (NET-4, §7.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Attempts per source (≥ 1). `1` means "no retry".
    pub max_attempts: u32,
    /// Delay before the first retry.
    pub base_backoff: Duration,
    /// Cap on the (doubling) backoff between attempts.
    pub max_backoff: Duration,
}

impl RetryPolicy {
    /// A no-wait, single-attempt policy (tests / synchronous stores).
    #[must_use]
    pub fn none() -> Self {
        Self {
            max_attempts: 1,
            base_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        }
    }

    /// The backoff before attempt `attempt` (0-indexed): `base * 2^(attempt-1)`, capped.
    #[must_use]
    fn backoff_for(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        let shift = (attempt - 1).min(16);
        let scaled = self.base_backoff.saturating_mul(1u32 << shift);
        scaled.min(self.max_backoff)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_backoff: Duration::from_millis(20),
            max_backoff: Duration::from_millis(200),
        }
    }
}

/// Attempt a hash-verified `get` of `key` from `store` under `policy` (bounded backoff).
///
/// Returns `Ok(Some(bytes))` on the first verified fetch, `Ok(None)` if the object was a typed
/// miss on every attempt (so the caller can fall through to another source), or `Err` for a hard
/// failure that must not be masked — a [`SwarmNetError::HashMismatch`] (tamper/corruption, §12) or
/// a non-miss transport error.
async fn try_source(
    store: &dyn PayloadStore,
    key: &PayloadKey,
    expected: &ContentHash,
    policy: RetryPolicy,
) -> Result<Option<Vec<u8>>, SwarmNetError> {
    let attempts = policy.max_attempts.max(1);
    for attempt in 0..attempts {
        let backoff = policy.backoff_for(attempt);
        if !backoff.is_zero() {
            tokio::time::sleep(backoff).await;
        }
        match store.get(key, expected).await {
            Ok(bytes) => return Ok(Some(bytes)),
            // A miss is retryable (the object may still be landing) and, once exhausted, falls
            // through to the next source.
            Err(SwarmNetError::PayloadMiss(_)) => continue,
            // A hash mismatch is never masked by a retry — the source served the wrong bytes.
            Err(e @ SwarmNetError::HashMismatch { .. }) => return Err(e),
            // Any other transport error is surfaced.
            Err(e) => return Err(e),
        }
    }
    Ok(None)
}

/// Fetch a committed payload, trying `primary` first (with `policy` backoff) then each of
/// `fallbacks` in order, returning the first blake3-verified copy.
///
/// A [`SwarmNetError::PayloadMiss`] is returned only when **every** source is exhausted — the typed
/// miss the §6.4 stall ladder consumes. A tamper ([`SwarmNetError::HashMismatch`]) from any source
/// aborts immediately (it is a correctness fault, not an availability one).
pub async fn fetch_with_fallback<P: PayloadStore>(
    primary: &P,
    fallbacks: &[&P],
    key: &PayloadKey,
    expected: &ContentHash,
    policy: RetryPolicy,
) -> Result<Vec<u8>, SwarmNetError> {
    if let Some(bytes) = try_source(primary, key, expected, policy).await? {
        return Ok(bytes);
    }
    for source in fallbacks {
        if let Some(bytes) = try_source(*source, key, expected, policy).await? {
            return Ok(bytes);
        }
    }
    Err(exhausted(key, 1 + fallbacks.len()))
}

/// Like [`fetch_with_fallback`], but over **heterogeneous** stores behind a trait object
/// (`&[&dyn PayloadStore]`) — e.g. an [`R2Store`](crate::R2Store) primary with an
/// [`FsPayloadStore`](crate::store::FsPayloadStore) mirror fallback (NET-4, the monomorphic-fallback
/// gap). The `PayloadStore` trait is `async_trait` and object-safe, so this needs no wrapper type.
///
/// Sources are tried in the given (cost) order under `policy` backoff; returns the first
/// blake3-verified copy, a tamper ([`SwarmNetError::HashMismatch`]) aborts immediately, and a typed
/// [`SwarmNetError::PayloadMiss`] surfaces only once **every** source is exhausted.
pub async fn fetch_with_fallback_dyn(
    stores: &[&dyn PayloadStore],
    key: &PayloadKey,
    expected: &ContentHash,
    policy: RetryPolicy,
) -> Result<Vec<u8>, SwarmNetError> {
    for store in stores {
        if let Some(bytes) = try_source(*store, key, expected, policy).await? {
            return Ok(bytes);
        }
    }
    Err(exhausted(key, stores.len()))
}

/// Fetch and decode a `record-set.cbor` object (spec §6.4, §11.3; TDD RUN-2 net half).
///
/// Fetches the object via [`PayloadStore::get`] (which blake3-verifies the raw bytes equal
/// `expected` — the locator hash), decodes [`RecordSet`], and re-verifies the decoded set's content
/// address equals `expected` (so a non-canonical re-encoding is also rejected). Root verification
/// against the `RoundRecord`'s signed commitment stays **engine-side** (B3 wires this into
/// `engine.rs::verify_record_set` in Wave 3); this is the net-side fetch+decode+content-verify half.
pub async fn fetch_record_set<P: PayloadStore>(
    store: &P,
    key: &PayloadKey,
    expected: &ContentHash,
) -> Result<RecordSet, SwarmNetError> {
    let bytes = store.get(key, expected).await?;
    let set = RecordSet::from_canonical_slice(&bytes)
        .map_err(|e| SwarmNetError::Fetch(format!("decode record-set: {e}")))?;
    let actual = set
        .content_hash()
        .map_err(|e| SwarmNetError::Fetch(format!("hash record-set: {e}")))?;
    if &actual != expected {
        return Err(SwarmNetError::HashMismatch {
            expected: expected.to_hex(),
            actual: actual.to_hex(),
        });
    }
    Ok(set)
}

/// The typed "exhausted every source" miss the §6.4 stall ladder consumes.
fn exhausted(key: &PayloadKey, sources: usize) -> SwarmNetError {
    SwarmNetError::PayloadMiss(format!(
        "{}@r{}/{} unavailable from {sources} source(s)",
        key.run.as_str(),
        key.round,
        key.peer.to_hex(),
    ))
}

// ---------------------------------------------------------------------------------------------
// Download scheduler — a DIRECT port of Psyche's `download_scheduler_actor`
// (`psyche/shared/network/src/download/scheduler.rs:189-375`; its DIRECT tests at
// `scheduler.rs:411-675`). Deltas from upstream (recorded in `swarm-ledger-b1.md`):
//
//   * Upstream keys retry entries by an `iroh_blobs::Hash` (a blob ticket). We key by the blake3
//     `ContentHash` and carry the `PayloadKey` — no iroh-blobs this program (P4).
//   * Upstream has three retry classes (DistroResult + ModelSharing Parameter/Config). We keep
//     **one** class — the payload fetch (= upstream's `DistroResult`: expo backoff
//     `backoff_base * 2^prev_retries`, capped `max_payload_retries`, time-gated `due_retries`).
//     The ModelSharing classes are P2P model sharing (P4), dropped.
//   * The capacity gate + FIFO waiters + release-transfers-a-slot are ported 1:1.
// ---------------------------------------------------------------------------------------------

/// Per-class retry policy for the payload fetch class (Psyche `RetryConfig`, scheduler.rs:14-26).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryConfig {
    /// Base backoff; the retry after `n` prior attempts waits `backoff_base * 2^n`.
    pub backoff_base: Duration,
    /// Cap on payload-fetch retries before [`RetryQueueResult::MaxRetriesExceeded`] (Psyche's
    /// `max_distro_retries`, default 3).
    pub max_payload_retries: usize,
}

impl Default for RetryConfig {
    fn default() -> Self {
        // scheduler.rs:19-25 — 2s base, 3 max.
        Self {
            backoff_base: Duration::from_secs(2),
            max_payload_retries: 3,
        }
    }
}

/// The outcome of queuing a failed download for retry (Psyche `RetryQueueResult`, scheduler.rs:48-52).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryQueueResult {
    /// The retry was queued (its backoff is now ticking).
    Queued,
    /// The object exceeded `max_payload_retries` and was dropped from the retry set.
    MaxRetriesExceeded,
}

/// A retry that is due to be re-attempted (Psyche `ReadyRetry`, scheduler.rs:54-61).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadyRetry {
    /// The object's blake3 content hash (the retry-set key).
    pub hash: ContentHash,
    /// The payload locator to re-fetch.
    pub key: PayloadKey,
    /// How many attempts have been made so far.
    pub retries: usize,
}

/// One pending retry (Psyche `RetryEntry`, scheduler.rs:28-46).
struct RetryEntry {
    retries: usize,
    retry_time: Option<Instant>,
    key: PayloadKey,
}

/// Actor mailbox messages (Psyche `SchedulerMessage`, scheduler.rs:63-88).
enum SchedulerMessage {
    WaitForCapacity {
        response: oneshot::Sender<()>,
    },
    ReleaseCapacity,
    QueueFailed {
        hash: ContentHash,
        key: PayloadKey,
        response: oneshot::Sender<RetryQueueResult>,
    },
    RemoveRetry {
        hash: ContentHash,
        response: oneshot::Sender<bool>,
    },
    DueRetries {
        response: oneshot::Sender<Vec<ReadyRetry>>,
    },
}

/// A concurrency + retry scheduler for payload fetches (Psyche `DownloadSchedulerHandle`,
/// scheduler.rs:90-187). A `max_concurrent` capacity gate with FIFO waiters bounds in-flight
/// fetches; failed fetches are queued for time-gated, capped-retry re-attempts.
#[derive(Clone)]
pub struct DownloadScheduler {
    tx: mpsc::UnboundedSender<SchedulerMessage>,
}

impl DownloadScheduler {
    /// Spawn a scheduler granting up to `max_concurrent` in-flight fetches, retrying per `retry`.
    pub fn new(max_concurrent: usize, retry: RetryConfig) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(scheduler_actor(rx, max_concurrent, retry));
        Self { tx }
    }

    /// Send a request and await its reply, returning `default` if the actor has shut down
    /// (scheduler.rs:108-120).
    async fn request<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> SchedulerMessage,
        default: T,
    ) -> T {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(make(tx)).is_err() {
            return default;
        }
        rx.await.unwrap_or(default)
    }

    /// Acquire one capacity slot, awaiting until one frees (FIFO). Errors iff the actor shut down
    /// (scheduler.rs:122-132). The caller must [`release_capacity`](Self::release_capacity) when done.
    pub async fn wait_for_capacity(&self) -> Result<(), SwarmNetError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(SchedulerMessage::WaitForCapacity { response: tx })
            .map_err(|_| SwarmNetError::Transport("download scheduler actor shut down".into()))?;
        rx.await
            .map_err(|_| SwarmNetError::Transport("download scheduler dropped before reply".into()))
    }

    /// Release one capacity slot, handing it to the next FIFO waiter if any (scheduler.rs:134-136).
    pub fn release_capacity(&self) {
        let _ = self.tx.send(SchedulerMessage::ReleaseCapacity);
    }

    /// Queue a failed fetch of `key`/`hash` for retry. Returns [`RetryQueueResult::MaxRetriesExceeded`]
    /// once the object has failed more than `max_payload_retries` times (scheduler.rs:138-154).
    pub async fn queue_failed(&self, key: PayloadKey, hash: ContentHash) -> RetryQueueResult {
        self.request(
            |response| SchedulerMessage::QueueFailed {
                hash,
                key,
                response,
            },
            RetryQueueResult::MaxRetriesExceeded,
        )
        .await
    }

    /// Drop a queued retry (e.g. the object arrived by another path). Returns whether it was present
    /// (scheduler.rs:156-162).
    pub async fn remove_retry(&self, hash: ContentHash) -> bool {
        self.request(
            |response| SchedulerMessage::RemoveRetry { hash, response },
            false,
        )
        .await
    }

    /// Take every retry whose backoff has elapsed (draining them from the set — Psyche
    /// `get_due_distro_retries`, scheduler.rs:180-186 + 322-329).
    pub async fn due_retries(&self) -> Vec<ReadyRetry> {
        self.request(
            |response| SchedulerMessage::DueRetries { response },
            Vec::new(),
        )
        .await
    }
}

/// Actor state (Psyche `DownloadSchedulerActor`, scheduler.rs:189-206).
struct SchedulerActor {
    active: usize,
    max_concurrent: usize,
    waiting: VecDeque<oneshot::Sender<()>>,
    entries: HashMap<ContentHash, RetryEntry>,
    retry: RetryConfig,
}

impl SchedulerActor {
    fn handle(&mut self, message: SchedulerMessage) {
        match message {
            // scheduler.rs:210-217
            SchedulerMessage::WaitForCapacity { response } => {
                if self.active < self.max_concurrent {
                    self.active += 1;
                    let _ = response.send(());
                } else {
                    self.waiting.push_back(response);
                }
            }
            // scheduler.rs:219-222
            SchedulerMessage::ReleaseCapacity => {
                self.active = self.active.saturating_sub(1);
                self.notify_next_waiter();
            }
            // scheduler.rs:251-273 (the DistroResult branch — our single class).
            SchedulerMessage::QueueFailed {
                hash,
                key,
                response,
            } => {
                let prev = self.entries.get(&hash).map(|e| e.retries).unwrap_or(0);
                let new_retries = prev + 1;
                if new_retries > self.retry.max_payload_retries {
                    self.entries.remove(&hash);
                    let _ = response.send(RetryQueueResult::MaxRetriesExceeded);
                } else {
                    let backoff = self.retry.backoff_base.mul_f32(2_f32.powi(prev as i32));
                    self.entries.insert(
                        hash,
                        RetryEntry {
                            retries: new_retries,
                            retry_time: Some(Instant::now() + backoff),
                            key,
                        },
                    );
                    let _ = response.send(RetryQueueResult::Queued);
                }
            }
            // scheduler.rs:277-280
            SchedulerMessage::RemoveRetry { hash, response } => {
                let _ = response.send(self.entries.remove(&hash).is_some());
            }
            // scheduler.rs:322-329
            SchedulerMessage::DueRetries { response } => {
                let now = Instant::now();
                let due: Vec<ContentHash> = self
                    .entries
                    .iter()
                    .filter(|(_, e)| e.retry_time.map(|t| now >= t).unwrap_or(false))
                    .map(|(h, _)| *h)
                    .collect();
                let ready = due
                    .into_iter()
                    .filter_map(|hash| {
                        self.entries.remove(&hash).map(|e| ReadyRetry {
                            hash,
                            key: e.key,
                            retries: e.retries,
                        })
                    })
                    .collect();
                let _ = response.send(ready);
            }
        }
    }

    /// Hand a freed slot to the next live FIFO waiter (scheduler.rs:351-362).
    fn notify_next_waiter(&mut self) {
        while let Some(waiter) = self.waiting.pop_front() {
            if waiter.send(()).is_ok() {
                self.active += 1;
                return;
            }
        }
    }
}

/// The actor loop (scheduler.rs:365-375).
async fn scheduler_actor(
    mut rx: mpsc::UnboundedReceiver<SchedulerMessage>,
    max_concurrent: usize,
    retry: RetryConfig,
) {
    let mut actor = SchedulerActor {
        active: 0,
        max_concurrent,
        waiting: VecDeque::new(),
        entries: HashMap::new(),
        retry,
    };
    while let Some(message) = rx.recv().await {
        actor.handle(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::{PeerId, RunId};
    use crate::store::FsPayloadStore;
    use crate::test_support::temp_root;
    use daemon_swarm_proto::blake3_hash;

    fn key(peer: u8) -> PayloadKey {
        PayloadKey::new(RunId::new("run-f"), 4, PeerId([peer; 32]))
    }

    #[tokio::test]
    async fn primary_hit_never_touches_fallback() {
        let pdir = temp_root("fetch-primary");
        let fdir = temp_root("fetch-fallback-unused");
        let primary = FsPayloadStore::open(pdir.path(), 8).unwrap();
        let fallback = FsPayloadStore::open(fdir.path(), 8).unwrap();
        let k = key(0x01);
        let hash = primary.put(&k, b"payload").await.unwrap();

        let got = fetch_with_fallback(&primary, &[&fallback], &k, &hash, RetryPolicy::none())
            .await
            .unwrap();
        assert_eq!(got, b"payload");
    }

    #[tokio::test]
    async fn primary_miss_falls_back_to_second_store() {
        let pdir = temp_root("fetch-primary-miss");
        let fdir = temp_root("fetch-fallback-hit");
        let primary = FsPayloadStore::open(pdir.path(), 8).unwrap();
        let fallback = FsPayloadStore::open(fdir.path(), 8).unwrap();
        let k = key(0x02);
        // Only the fallback has the object.
        let hash = fallback.put(&k, b"mirrored").await.unwrap();

        let got = fetch_with_fallback(&primary, &[&fallback], &k, &hash, RetryPolicy::none())
            .await
            .unwrap();
        assert_eq!(got, b"mirrored");
    }

    #[tokio::test]
    async fn all_sources_miss_is_typed_miss() {
        let pdir = temp_root("fetch-all-miss-p");
        let fdir = temp_root("fetch-all-miss-f");
        let primary = FsPayloadStore::open(pdir.path(), 8).unwrap();
        let fallback = FsPayloadStore::open(fdir.path(), 8).unwrap();
        let k = key(0x03);

        let err = fetch_with_fallback(
            &primary,
            &[&fallback],
            &k,
            &blake3_hash(b"never-stored"),
            RetryPolicy::none(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SwarmNetError::PayloadMiss(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn retry_then_succeed_after_late_put() {
        let pdir = temp_root("fetch-retry");
        let primary = FsPayloadStore::open(pdir.path(), 8).unwrap();
        let k = key(0x04);
        let bytes = b"lands-late".to_vec();
        let hash = blake3_hash(&bytes);

        // Write the object after a short delay, so the first attempt misses and a retry succeeds.
        let writer = {
            let store = primary.clone();
            let k = k.clone();
            let bytes = bytes.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(15)).await;
                store.put(&k, &bytes).await.unwrap();
            })
        };

        let policy = RetryPolicy {
            max_attempts: 8,
            base_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(40),
        };
        let got = fetch_with_fallback(&primary, &[], &k, &hash, policy)
            .await
            .unwrap();
        assert_eq!(got, bytes);
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn tamper_aborts_and_is_not_masked_by_fallback() {
        let pdir = temp_root("fetch-tamper-p");
        let fdir = temp_root("fetch-tamper-f");
        let primary = FsPayloadStore::open(pdir.path(), 8).unwrap();
        let fallback = FsPayloadStore::open(fdir.path(), 8).unwrap();
        let k = key(0x05);
        // Primary serves bytes that do not match the expected hash.
        primary.put(&k, b"corrupt").await.unwrap();
        fallback.put(&k, b"honest").await.unwrap();

        let expected = blake3_hash(b"honest");
        let err = fetch_with_fallback(&primary, &[&fallback], &k, &expected, RetryPolicy::none())
            .await
            .unwrap_err();
        assert!(
            matches!(err, SwarmNetError::HashMismatch { .. }),
            "got {err:?}"
        );
    }

    // --- fetch_with_fallback_dyn (NET-4 monomorphic-fallback gap) ---------------------------------

    #[tokio::test]
    async fn dyn_fallback_tries_stores_in_order() {
        // Two heterogeneous-by-trait-object stores; only the second holds the object.
        let pdir = temp_root("dyn-primary");
        let fdir = temp_root("dyn-fallback");
        let primary = FsPayloadStore::open(pdir.path(), 8).unwrap();
        let fallback = FsPayloadStore::open(fdir.path(), 8).unwrap();
        let k = key(0x21);
        let hash = fallback.put(&k, b"via-dyn").await.unwrap();

        let stores: [&dyn PayloadStore; 2] = [&primary, &fallback];
        let got = fetch_with_fallback_dyn(&stores, &k, &hash, RetryPolicy::none())
            .await
            .unwrap();
        assert_eq!(got, b"via-dyn");
    }

    #[tokio::test]
    async fn dyn_fallback_all_miss_is_typed_miss() {
        let pdir = temp_root("dyn-miss-p");
        let primary = FsPayloadStore::open(pdir.path(), 8).unwrap();
        let k = key(0x22);
        let stores: [&dyn PayloadStore; 1] = [&primary];
        let err = fetch_with_fallback_dyn(&stores, &k, &blake3_hash(b"nope"), RetryPolicy::none())
            .await
            .unwrap_err();
        assert!(matches!(err, SwarmNetError::PayloadMiss(_)), "got {err:?}");
    }

    // --- fetch_record_set (RUN-2 net half) -------------------------------------------------------

    #[tokio::test]
    async fn record_set_round_trips_and_content_verifies() {
        use daemon_swarm_proto::messages::RecordEntry;

        let dir = temp_root("recordset-ok");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let set = RecordSet::new([
            RecordEntry {
                peer: PeerId([2; 32]),
                hash: blake3_hash(b"p2"),
                size: 2,
            },
            RecordEntry {
                peer: PeerId([1; 32]),
                hash: blake3_hash(b"p1"),
                size: 1,
            },
        ]);
        let bytes = set.to_canonical_vec().unwrap();
        let expected = set.content_hash().unwrap();
        let k = key(0x30);
        store.put(&k, &bytes).await.unwrap();

        let fetched = fetch_record_set(&store, &k, &expected).await.unwrap();
        assert_eq!(fetched, set);
        assert_eq!(fetched.len(), 2);
    }

    #[tokio::test]
    async fn tampered_set_object_rejected() {
        // The store holds bytes that do not hash to the locator hash → the get-side blake3 verify
        // rejects before we ever decode (RUN-2's tamper reject, net side).
        let dir = temp_root("recordset-tamper");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let k = key(0x31);
        store.put(&k, b"not-a-record-set").await.unwrap();

        let locator_hash = blake3_hash(b"the-honest-record-set-bytes");
        let err = fetch_record_set(&store, &k, &locator_hash)
            .await
            .unwrap_err();
        assert!(
            matches!(err, SwarmNetError::HashMismatch { .. }),
            "got {err:?}"
        );
    }

    // --- DownloadScheduler: DIRECT ports of Psyche scheduler.rs:411-675 ---------------------------

    fn retry_key(seed: u8) -> (PayloadKey, ContentHash) {
        let k = PayloadKey::new(RunId::new("run-s"), 1, PeerId([seed; 32]));
        (k, blake3_hash(&[seed]))
    }

    fn fast_config() -> RetryConfig {
        // scheduler.rs:385-390
        RetryConfig {
            backoff_base: Duration::from_millis(10),
            max_payload_retries: 3,
        }
    }

    /// Port of `test_capacity_grants_up_to_max` (scheduler.rs:411-421).
    #[tokio::test]
    async fn capacity_grants_up_to_max() {
        let scheduler = DownloadScheduler::new(2, RetryConfig::default());
        scheduler.wait_for_capacity().await.unwrap();
        scheduler.wait_for_capacity().await.unwrap();
        let third =
            tokio::time::timeout(Duration::from_millis(50), scheduler.wait_for_capacity()).await;
        assert!(third.is_err(), "third acquire must block past capacity");
    }

    /// Port of `test_release_unblocks_waiter` (scheduler.rs:423-436).
    #[tokio::test]
    async fn release_unblocks_waiter() {
        let scheduler = DownloadScheduler::new(1, RetryConfig::default());
        scheduler.wait_for_capacity().await.unwrap();

        let clone = scheduler.clone();
        let waiter = tokio::spawn(async move { clone.wait_for_capacity().await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        scheduler.release_capacity();

        let joined = tokio::time::timeout(Duration::from_millis(100), waiter).await;
        assert!(joined.is_ok(), "release must unblock the waiter");
    }

    /// Port of `test_waiters_are_served_fifo` (scheduler.rs:438-477).
    #[tokio::test]
    async fn waiters_are_served_fifo() {
        let scheduler = DownloadScheduler::new(1, RetryConfig::default());
        scheduler.wait_for_capacity().await.unwrap();

        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let s1 = scheduler.clone();
        let o1 = order.clone();
        let w1 = tokio::spawn(async move {
            s1.wait_for_capacity().await.unwrap();
            o1.lock().unwrap().push(1);
            tokio::time::sleep(Duration::from_millis(5)).await;
            s1.release_capacity();
        });
        tokio::time::sleep(Duration::from_millis(10)).await;

        let s2 = scheduler.clone();
        let o2 = order.clone();
        let w2 = tokio::spawn(async move {
            s2.wait_for_capacity().await.unwrap();
            o2.lock().unwrap().push(2);
            s2.release_capacity();
        });

        scheduler.release_capacity();
        tokio::time::timeout(Duration::from_millis(200), async {
            w1.await.unwrap();
            w2.await.unwrap();
        })
        .await
        .unwrap();
        assert_eq!(*order.lock().unwrap(), vec![1, 2], "FIFO order");
    }

    /// Port of `test_distro_retry_not_immediately_due` (scheduler.rs:540-557).
    #[tokio::test]
    async fn payload_retry_not_immediately_due() {
        let scheduler = DownloadScheduler::new(2, fast_config());
        let (k, h) = retry_key(1);
        assert_eq!(scheduler.queue_failed(k, h).await, RetryQueueResult::Queued);
        assert!(
            scheduler.due_retries().await.is_empty(),
            "backoff has not elapsed yet"
        );
    }

    /// Port of `test_distro_retries_returned_and_removed` (scheduler.rs:559-583).
    #[tokio::test]
    async fn payload_retries_returned_and_removed() {
        let scheduler = DownloadScheduler::new(2, fast_config());
        let (k1, h1) = retry_key(1);
        let (k2, h2) = retry_key(2);
        scheduler.queue_failed(k1, h1).await;
        scheduler.queue_failed(k2, h2).await;

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(scheduler.due_retries().await.len(), 2);
        assert!(
            scheduler.due_retries().await.is_empty(),
            "retries drained after the first take"
        );
        assert!(!scheduler.remove_retry(h1).await);
        assert!(!scheduler.remove_retry(h2).await);
    }

    /// Port of `test_remove_retry` (scheduler.rs:606-618).
    #[tokio::test]
    async fn remove_retry_reports_presence() {
        let scheduler = DownloadScheduler::new(2, RetryConfig::default());
        let (k, h) = retry_key(1);
        scheduler.queue_failed(k, h).await;
        assert!(scheduler.remove_retry(h).await);
        assert!(!scheduler.remove_retry(h).await);
    }

    /// Port of `test_distro_max_retries_exceeded` (scheduler.rs:620-650).
    #[tokio::test]
    async fn payload_max_retries_exceeded() {
        let config = RetryConfig {
            backoff_base: Duration::from_millis(1),
            max_payload_retries: 2,
        };
        let scheduler = DownloadScheduler::new(2, config);
        let (k, h) = retry_key(1);
        assert_eq!(
            scheduler.queue_failed(k.clone(), h).await,
            RetryQueueResult::Queued
        );
        assert_eq!(
            scheduler.queue_failed(k.clone(), h).await,
            RetryQueueResult::Queued
        );
        assert_eq!(
            scheduler.queue_failed(k, h).await,
            RetryQueueResult::MaxRetriesExceeded
        );
    }

    /// Port of `test_wait_for_capacity_errors_on_actor_shutdown` (scheduler.rs:675-682).
    #[tokio::test]
    async fn wait_for_capacity_errors_on_actor_shutdown() {
        let (tx, rx) = mpsc::unbounded_channel::<SchedulerMessage>();
        drop(rx);
        let dead = DownloadScheduler { tx };
        assert!(dead.wait_for_capacity().await.is_err());
    }
}
