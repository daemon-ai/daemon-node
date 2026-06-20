//! 5-tier semantic query cache — port of `query_cache.py` (P2).
//!
//! Tier 1 exact normalized key; Tier 2 cosine >= 0.88; Tier 3 cosine >= 0.78 + word Jaccard >= 0.15;
//! Tier 4 >= 70% word overlap (min 2); Tier 5 miss. Default `max_size=1000`, `ttl=3600s`. Scaffold.

/// Tier-2 cosine threshold (`query_cache.py` L214-L217).
pub const TIER2_COSINE: f64 = 0.88;
/// Tier-3 cosine threshold (`query_cache.py` L220-L224).
pub const TIER3_COSINE: f64 = 0.78;
/// Default cache capacity / TTL seconds (`query_cache.py` L49).
pub const DEFAULT_MAX_SIZE: usize = 1000;
/// Default TTL in seconds.
pub const DEFAULT_TTL_SECONDS: u64 = 3600;
