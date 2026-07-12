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
//! Only the [`FsPayloadStore`](crate::store::FsPayloadStore) plane exists this wave, so a fallback
//! *source* is a second [`PayloadStore`] (e.g. a mirror root). The alternate-locator hook
//! (`BlobTicket` → iroh-blobs) slots behind the same shape once the network plane lands
//! (`// MERGE-2` / Wave 3).

use std::time::Duration;

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
async fn try_source<P: PayloadStore>(
    store: &P,
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
    Err(SwarmNetError::PayloadMiss(format!(
        "{}@r{}/{} unavailable from {} source(s)",
        key.run.as_str(),
        key.round,
        key.peer.to_hex(),
        1 + fallbacks.len()
    )))
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
}
