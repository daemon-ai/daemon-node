// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`PinnedArtifactStore`] — the run's **genesis-pinned artifact plane**, presented as a
//! [`ContentStore`] so the role session's `data.fetch` seat binds it unchanged.
//!
//! # The gap this closes
//!
//! The `data@2` world is an ARTIFACT fetch: "the only inputs are the committed blake3 (edge-pinned
//! in the envelope's artifact map, §5.1) and a byte range — no URL, no locator, no credential
//! crosses this boundary (the resolver + its credentials stay embedder-side)". The envelope's
//! artifact map is `name -> (url, blake3)`, and the run's publisher writes each object at the key
//! that url names: `modules/<blake3>.wasm` for a module and, for the chunk-addressed corpus,
//! `corpus/<manifest blake3>.cbor` / `corpus/<tokenizer blake3>.json` / `corpus/<fold>.bin`
//! (ABI §12.7 [CC-7]; `xtask publish-corpus`). Both sides derive those keys from
//! [`PublishedArtifact`](crate::PublishedArtifact) — the one place the scheme is spelled, so a
//! genesis cannot pin a url the publisher does not write.
//!
//! The committed-PAYLOAD plane is a different namespace on the same store: [RS-4]'s
//! `payload/<blake3-hex>`, where the module's own `payload_put` objects and the checkpoint family
//! chunks land ([`R2Store`](crate::R2Store)'s `ContentStore` impl). Binding the payload plane to
//! the artifact seat as well meant every module-driven corpus fetch looked for a genesis-pinned
//! object under a key nothing ever published it at — a typed
//! [`PayloadMiss`](VhcNetError::PayloadMiss) that the trainer guest, which fetches its
//! genesis-pinned corpus manifest before its first round, takes as a fatal init failure.
//!
//! This store is the missing half. It keeps the seam content-addressed (the caller still names only
//! a blake3; the url never crosses the guest boundary) and resolves in three steps:
//!
//! 1. **The on-disk content cache** ([`ContentCache`], the §8/§10.6 `[vhc].data_cache_gb` budget) —
//!    so a box warmed once (including by the worker's `DAEMON_TRAIN_PREFETCH` staging mode, whose
//!    whole point is "a subsequent live run on the box finds every artifact cache-warm") does not
//!    re-download a pinned object per round, per run, or across process restarts.
//! 2. **The artifact plane** ([`ArtifactResolver`]) at the url the ENVELOPE pins, under a bounded
//!    retry (a pinned object that is still landing is an availability blip, not a run failure).
//! 3. **The content plane** — the fallback for every hash that is not genesis-pinned: the
//!    committed payloads, the checkpoint documents, and the det-state family chunks a restoring
//!    peer range-reads ([SF-R2]), all of which really do live at `payload/<hex>`.
//!
//! # Verification, and why the cache is selective
//!
//! Like every other [`ContentStore`] seat, this one serves the keyed bytes **verbatim**: the run
//! pump is the arbiter (plain hash for whole artifacts, the registered covering-chunk hashes for
//! chunk-addressed shards), because the requested address is not always the object's plain blake3 —
//! a corpus shard is keyed by its domain-separated chunk FOLD. That is also why the ON-DISK cache is
//! only populated for objects whose bytes DO hash to the requested key: [`ContentCache`]
//! blake3-verifies every entry against its key by design, and a fold-keyed shard cannot honour that
//! invariant.
//!
//! Fold-keyed objects get an **in-process verbatim LRU** instead ([`FoldLru`]). Serving them
//! store-fresh on every fetch was a live-run outage, not a conservative choice: a trainer's
//! round-0 fetch plan issues thousands of tiny per-sequence range reads, each of which the session
//! seat services by fetching the WHOLE covering shard — so an uncached shard meant one full HTTP
//! round-trip (~100 ms) per ~2 KiB guest read, a ~10 ops/s crawl with the accelerator idle. The
//! trust argument is unchanged by the cache: this seat serves VERBATIM in every case and the pump
//! verifies the covering chunks against the registered chunk map, so a cached-but-wrong shard
//! fails exactly as a fetched-but-wrong one would. The LRU is bounded ([`FOLD_LRU_BUDGET`]) and
//! process-local; the durable cache-warm path for shards remains the prefetch mode's CHUNK-keyed
//! staging.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use daemon_vhc_proto::blake3_hash;

use crate::artifact::ArtifactResolver;
use crate::content_cache::ContentCache;
use crate::seam::ContentHash;
use crate::transport::ContentStore;
use crate::VhcNetError;

/// Attempts a pinned-artifact fetch is given before it is reported as a miss. The retention floor
/// (§7.4) guarantees a published object outlives a bounded retry, and the alternative — failing the
/// first attempt — turns a transient store blip into a terminated run (the trainer treats its
/// genesis-pinned corpus manifest as fatal-if-absent, correctly: it IS the run's data identity).
const PINNED_FETCH_ATTEMPTS: u32 = 3;

/// The first backoff between pinned-artifact attempts; doubles up to [`PINNED_FETCH_MAX_BACKOFF`].
const PINNED_FETCH_BACKOFF: Duration = Duration::from_millis(250);

/// The backoff ceiling between pinned-artifact attempts.
const PINNED_FETCH_MAX_BACKOFF: Duration = Duration::from_secs(2);

/// The in-process fold-keyed LRU's byte budget. Sized for shard locality, not the whole corpus:
/// a round's fetch plan strides a handful of ~2 MiB shards at a time, and the budget must stay
/// small enough never to distort a worker's admitted memory envelope on a modest box.
const FOLD_LRU_BUDGET: usize = 64 * 1024 * 1024;

/// A minimal byte-bounded LRU for pinned fold-keyed objects (see the module docs for why the
/// blake3-keyed [`ContentCache`] cannot hold them). Entries are whole shard objects; lookups
/// refresh recency; inserts evict from the least-recent end until the budget holds. An object
/// larger than the whole budget is simply not cached.
#[derive(Debug, Default)]
struct FoldLru {
    /// `(key, bytes)` in recency order — least recent at the front.
    entries: std::collections::VecDeque<(ContentHash, Vec<u8>)>,
    bytes: usize,
}

impl FoldLru {
    fn get(&mut self, hash: &ContentHash) -> Option<Vec<u8>> {
        let pos = self.entries.iter().position(|(h, _)| h == hash)?;
        let entry = self.entries.remove(pos).expect("position just located");
        let bytes = entry.1.clone();
        self.entries.push_back(entry);
        Some(bytes)
    }

    fn insert(&mut self, hash: ContentHash, bytes: &[u8]) {
        if bytes.len() > FOLD_LRU_BUDGET {
            return;
        }
        if let Some(pos) = self.entries.iter().position(|(h, _)| h == &hash) {
            let (_, old) = self.entries.remove(pos).expect("position just located");
            self.bytes -= old.len();
        }
        self.bytes += bytes.len();
        self.entries.push_back((hash, bytes.to_vec()));
        while self.bytes > FOLD_LRU_BUDGET {
            let (_, evicted) = self
                .entries
                .pop_front()
                .expect("a positive byte tally implies an entry");
            self.bytes -= evicted.len();
        }
    }
}

/// A [`ContentStore`] that resolves the run's genesis-pinned artifacts at the urls the envelope
/// commits, cache-first, and delegates every other content address to the payload plane.
///
/// Construct with [`PinnedArtifactStore::new`]; see the module docs for the resolution order.
pub struct PinnedArtifactStore {
    /// The envelope's artifact map, keyed by the content id the module names. For a whole-object
    /// artifact that id IS `blake3(bytes)`; for a chunk-addressed corpus shard it is the fold.
    pinned: BTreeMap<ContentHash, String>,
    resolver: ArtifactResolver,
    cache: Option<ContentCache>,
    /// The in-process LRU for pinned FOLD-keyed objects (chunk-addressed shards) — the class the
    /// blake3-keyed on-disk cache refuses by invariant. See the module docs.
    fold_lru: std::sync::Mutex<FoldLru>,
    /// The committed-payload plane (`payload/<blake3>`): the fallback, and the put seat.
    content: Arc<dyn ContentStore>,
}

impl std::fmt::Debug for PinnedArtifactStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedArtifactStore")
            .field("pinned", &self.pinned.len())
            .field("resolver", &self.resolver)
            .field("cache", &self.cache)
            .finish_non_exhaustive()
    }
}

impl PinnedArtifactStore {
    /// Bind the run's artifact map (`content id -> url`, from the genesis envelope) over the
    /// `content` payload plane, optionally warmed by an on-disk `cache`.
    #[must_use]
    pub fn new(
        pinned: BTreeMap<ContentHash, String>,
        resolver: ArtifactResolver,
        cache: Option<ContentCache>,
        content: Arc<dyn ContentStore>,
    ) -> Self {
        Self {
            pinned,
            resolver,
            cache,
            fold_lru: std::sync::Mutex::new(FoldLru::default()),
            content,
        }
    }

    /// How many artifacts this plane can resolve at their pinned url (the rest fall through to the
    /// content plane) — the number a session logs so an operator can see the binding took.
    #[must_use]
    pub fn pinned_len(&self) -> usize {
        self.pinned.len()
    }

    /// Fetch one pinned artifact whole, at its committed url, under the bounded retry policy.
    ///
    /// The read is the **unverified** whole-object range (`[0, end)`) rather than
    /// [`ArtifactResolver::fetch`], for the same reason every other [`ContentStore`] seat serves
    /// verbatim: the requested address is not always the object's plain blake3 (a chunk-addressed
    /// shard is keyed by its fold), and the pump is the arbiter either way — plain hash for whole
    /// artifacts, registered covering-chunk hashes for chunk-addressed ones. A blake3 gate here
    /// would reject every shard.
    async fn fetch_pinned(&self, hash: &ContentHash, url: &str) -> Result<Vec<u8>, VhcNetError> {
        let mut backoff = PINNED_FETCH_BACKOFF;
        let mut last: Option<VhcNetError> = None;
        for attempt in 0..PINNED_FETCH_ATTEMPTS {
            if attempt > 0 {
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(PINNED_FETCH_MAX_BACKOFF);
            }
            match self.resolver.fetch_range(url, 0, 0).await {
                Ok(bytes) => return Ok(bytes),
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or_else(|| VhcNetError::PayloadMiss(hash.to_hex())))
    }
}

#[async_trait]
impl ContentStore for PinnedArtifactStore {
    /// PUTs are the committed-payload plane's, always: nothing writes a genesis-pinned artifact at
    /// run time (they are published before the run exists, and their hashes are inside the run id).
    async fn put_content(&self, bytes: &[u8]) -> Result<ContentHash, VhcNetError> {
        self.content.put_content(bytes).await
    }

    async fn get_content(&self, hash: &ContentHash) -> Result<Vec<u8>, VhcNetError> {
        let Some(url) = self.pinned.get(hash) else {
            // Not genesis-pinned: a committed payload, a checkpoint document, or a det-state
            // family chunk — all of which the content plane really does hold.
            return self.content.get_content(hash).await;
        };
        if let Some(cache) = &self.cache {
            if let Some(bytes) = cache.get(hash).await? {
                return Ok(bytes);
            }
        }
        // The fold-keyed class the disk cache cannot hold: served from the in-process LRU when
        // warm — the seam that turns a round-0 plan of thousands of per-sequence reads into one
        // HTTP fetch per shard instead of one per read.
        if let Some(bytes) = self.fold_lru.lock().expect("fold lru lock").get(hash) {
            return Ok(bytes);
        }
        match self.fetch_pinned(hash, url).await {
            Ok(bytes) => {
                // The disk cache holds only what honours its blake3-keyed invariant (see the
                // module docs): a fold-keyed shard's bytes do not hash to its artifact id, so
                // that class warms the in-process LRU instead.
                if &blake3_hash(&bytes) == hash {
                    if let Some(cache) = &self.cache {
                        // A cache write failure must never fail a fetch whose verified bytes are
                        // already in hand — the next fetch simply re-downloads. (This crate carries
                        // no unconditional tracing dep; the cache's own health is an operator
                        // property of the cache dir, not of this run.)
                        let _ = cache.insert(hash, &bytes).await;
                    }
                } else {
                    self.fold_lru
                        .lock()
                        .expect("fold lru lock")
                        .insert(*hash, &bytes);
                }
                Ok(bytes)
            }
            // The pinned location did not serve it. A copy may still sit on the content plane (a
            // co-located peer's `put_content`, or an operator that staged it there), so try that
            // before reporting the miss — and report the PINNED plane's error, which names the url
            // the run actually commits.
            Err(pinned_err) => match self.content.get_content(hash).await {
                Ok(bytes) => Ok(bytes),
                Err(_) => Err(pinned_err),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_r2::MockR2;
    use crate::transport::MemoryContentStore;
    use crate::RunId;

    /// Build a store over a `MockR2` whose objects live at the PUBLISHED artifact keys.
    fn store_over(
        mock: &MockR2,
        run: &str,
        pinned: BTreeMap<ContentHash, String>,
        content: Arc<dyn ContentStore>,
    ) -> PinnedArtifactStore {
        let resolver = ArtifactResolver::with_egress(mock.egress())
            .with_presign(Arc::new(mock.presign_client()), RunId::new(run));
        PinnedArtifactStore::new(pinned, resolver, None, content)
    }

    /// THE REGRESSION: a genesis-pinned corpus manifest published where `xtask publish-corpus`
    /// publishes it (`corpus/<blake3>.cbor`) is fetchable BY CONTENT HASH, even though the
    /// committed-payload plane's key for that hash (`payload/<hex>`) holds nothing. Binding the
    /// payload plane alone to the artifact seat is exactly the miss that killed a fleet trainer at
    /// its first corpus fetch.
    #[tokio::test]
    async fn a_pinned_corpus_object_resolves_at_its_published_key() {
        let mock = MockR2::start().await;
        let manifest = b"canonical-cbor-corpus-manifest".to_vec();
        let hash = blake3_hash(&manifest);
        mock.seed(
            &format!("runs/run-x/corpus/{}.cbor", hash.to_hex()),
            manifest.clone(),
        );

        let content = Arc::new(MemoryContentStore::new());
        // The payload plane does NOT have it — only the published artifact key does.
        assert!(matches!(
            content.get_content(&hash).await,
            Err(VhcNetError::PayloadMiss(_))
        ));

        let pinned = BTreeMap::from([(hash, format!("r2://corpus/{}.cbor", hash.to_hex()))]);
        let store = store_over(&mock, "run-x", pinned, content);
        assert_eq!(store.get_content(&hash).await.unwrap(), manifest);
    }

    /// A hash the envelope does not pin is the committed-payload plane's — the guest's own
    /// `payload_put` objects, checkpoint documents, and det-state family chunks.
    #[tokio::test]
    async fn an_unpinned_hash_falls_through_to_the_content_plane() {
        let mock = MockR2::start().await;
        let content = Arc::new(MemoryContentStore::new());
        let hash = content.put_content(b"a committed payload").await.unwrap();
        let store = store_over(&mock, "run-x", BTreeMap::new(), content);
        assert_eq!(
            store.get_content(&hash).await.unwrap(),
            b"a committed payload"
        );
    }

    /// A chunk-addressed shard's artifact id is its FOLD, which never equals `blake3(bytes)`: the
    /// store must still serve it (verbatim — the pump verifies the covering chunks), and must not
    /// mistake the non-matching hash for tamper.
    #[tokio::test]
    async fn a_fold_keyed_shard_is_served_verbatim() {
        let mock = MockR2::start().await;
        let shard = vec![7u8; 4096];
        // A fold identity: deliberately NOT `blake3(shard)`.
        let fold = ContentHash([0x5a; 32]);
        mock.seed(
            &format!("runs/run-x/corpus/{}.bin", fold.to_hex()),
            shard.clone(),
        );

        let pinned = BTreeMap::from([(fold, format!("r2://corpus/{}.bin", fold.to_hex()))]);
        let store = store_over(&mock, "run-x", pinned, Arc::new(MemoryContentStore::new()));
        assert_eq!(store.get_content(&fold).await.unwrap(), shard);
    }

    /// A pinned object that is genuinely absent everywhere is a typed miss naming the pinned
    /// plane's failure — never empty bytes and never the content plane's less informative one.
    #[tokio::test]
    async fn an_absent_pinned_artifact_is_a_typed_miss() {
        let mock = MockR2::start().await;
        let hash = blake3_hash(b"never published");
        let pinned = BTreeMap::from([(hash, format!("r2://corpus/{}.cbor", hash.to_hex()))]);
        let store = store_over(&mock, "run-x", pinned, Arc::new(MemoryContentStore::new()));
        let err = store.get_content(&hash).await.unwrap_err();
        assert!(
            matches!(err, VhcNetError::PayloadMiss(_) | VhcNetError::Fetch(_)),
            "got {err:?}"
        );
    }

    /// The round-0 crawl regression: a fold-keyed shard fetched once is served from the
    /// in-process LRU afterwards — never one whole-shard HTTP round-trip per tiny guest range
    /// read. Proven by evicting the object from the store after the first fetch: a warm LRU
    /// still serves it, so the second fetch cannot have gone to the wire.
    #[tokio::test]
    async fn a_fold_keyed_shard_is_served_from_the_lru_after_the_first_fetch() {
        let mock = MockR2::start().await;
        let shard = vec![9u8; 4096];
        // A fold identity: deliberately NOT `blake3(shard)` — the disk-cacheable class this is
        // not.
        let fold = ContentHash([0x3c; 32]);
        let key = format!("runs/run-x/corpus/{}.bin", fold.to_hex());
        mock.seed(&key, shard.clone());

        let pinned = BTreeMap::from([(fold, format!("r2://corpus/{}.bin", fold.to_hex()))]);
        let store = store_over(&mock, "run-x", pinned, Arc::new(MemoryContentStore::new()));
        assert_eq!(store.get_content(&fold).await.unwrap(), shard);
        mock.evict(&key);
        assert_eq!(
            store.get_content(&fold).await.unwrap(),
            shard,
            "the warm LRU serves the shard after the store dropped it"
        );
    }

    /// The LRU stays inside its byte budget: entries evict least-recent-first, and a re-read
    /// refreshes recency.
    #[test]
    fn the_fold_lru_evicts_least_recent_within_its_budget() {
        let mut lru = FoldLru::default();
        let a = ContentHash([1; 32]);
        let b = ContentHash([2; 32]);
        let c = ContentHash([3; 32]);
        let chunk = vec![0u8; FOLD_LRU_BUDGET / 2];
        lru.insert(a, &chunk);
        lru.insert(b, &chunk);
        assert!(lru.get(&a).is_some(), "two halves fit");
        // `a` was just refreshed, so the third insert evicts `b`.
        lru.insert(c, &chunk);
        assert!(lru.get(&b).is_none(), "least-recent evicted");
        assert!(lru.get(&a).is_some() && lru.get(&c).is_some());
        // An oversized object is refused outright, disturbing nothing.
        lru.insert(ContentHash([4; 32]), &vec![0u8; FOLD_LRU_BUDGET + 1]);
        assert!(lru.get(&a).is_some() && lru.get(&c).is_some());
    }

    /// The cache-warm property: a whole-object artifact fetched once is served from the on-disk
    /// cache afterwards, even when the store has since dropped it (the fleet-staging guarantee the
    /// worker's prefetch mode advertises).
    #[tokio::test]
    async fn a_whole_object_artifact_warms_the_content_cache() {
        let dir = crate::test_support::temp_root("pinned-artifact-cache");
        let mock = MockR2::start().await;
        let manifest = b"corpus manifest that should be cached".to_vec();
        let hash = blake3_hash(&manifest);
        let key = format!("runs/run-x/corpus/{}.cbor", hash.to_hex());
        mock.seed(&key, manifest.clone());

        let resolver = ArtifactResolver::with_egress(mock.egress())
            .with_presign(Arc::new(mock.presign_client()), RunId::new("run-x"));
        let cache = ContentCache::open(dir.path(), 1 << 20).unwrap();
        let store = PinnedArtifactStore::new(
            BTreeMap::from([(hash, format!("r2://corpus/{}.cbor", hash.to_hex()))]),
            resolver,
            Some(cache),
            Arc::new(MemoryContentStore::new()),
        );

        assert_eq!(store.get_content(&hash).await.unwrap(), manifest);
        mock.evict(&key);
        assert_eq!(
            store.get_content(&hash).await.unwrap(),
            manifest,
            "the warmed cache serves the object after the store dropped it"
        );
    }
}
