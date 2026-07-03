// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! 5-tier semantic query cache — port of `query_cache.py`.
//!
//! Tier 1 exact normalized key; Tier 2 cosine >= 0.88; Tier 3 cosine >= 0.78 + word Jaccard >= 0.15;
//! Tier 4 >= 70% word overlap (min 2 words); Tier 5 miss. Default `max_size=1000`, `ttl=3600s`.
//! Backed by a **separate** `query_cache.db` (`§4.7`), opened alongside the bank DB; tier-1/tier-4
//! entries persist, tier-2/3 embeddings stay in memory (mirroring Python). Invalidated wholesale on
//! every `remember` (`beam.py` L3041-L3043).

use crate::engine::MemoryRow;
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Tier-2 cosine threshold (`query_cache.py` L214-L217).
pub const TIER2_COSINE: f64 = 0.88;
/// Tier-3 cosine threshold (`query_cache.py` L220-L224).
pub const TIER3_COSINE: f64 = 0.78;
/// Tier-3 minimum word Jaccard (`query_cache.py` L222).
pub const TIER3_JACCARD: f64 = 0.15;
/// Default cache capacity.
pub const DEFAULT_MAX_SIZE: usize = 1000;
/// Default TTL in seconds.
pub const DEFAULT_TTL_SECONDS: u64 = 3600;

/// In-memory tier state (the SQLite file only persists tier-1/tier-4 result blobs).
#[derive(Default)]
struct State {
    tier1: HashMap<String, Vec<MemoryRow>>,
    tier23: HashMap<String, (Vec<f32>, Vec<MemoryRow>)>,
    tier4: HashMap<String, Vec<MemoryRow>>,
    insert_times: HashMap<String, Instant>,
    version: u64,
    hits: u64,
    misses: u64,
}

/// A 5-tier semantic recall cache (`query_cache.py` `QueryCache`).
pub struct QueryCache {
    inner: Mutex<State>,
    conn: Option<Mutex<Connection>>,
    max_size: usize,
    ttl: Duration,
}

impl QueryCache {
    /// Open a cache. `path = Some(..)` backs tier-1/tier-4 with a SQLite file (creating its parent);
    /// `None` is memory-only (used for ephemeral nodes). Cache-open failures degrade to memory-only.
    pub fn open(path: Option<&Path>) -> Self {
        let conn = path.and_then(|p| Self::open_db(p).ok().map(Mutex::new));
        let mut state = State::default();
        if let Some(c) = &conn {
            Self::load_existing(&c.lock().unwrap(), &mut state);
        }
        Self {
            inner: Mutex::new(state),
            conn,
            max_size: DEFAULT_MAX_SIZE,
            ttl: Duration::from_secs(DEFAULT_TTL_SECONDS),
        }
    }

    fn open_db(path: &Path) -> rusqlite::Result<Connection> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS query_cache (
                 normalized TEXT PRIMARY KEY,
                 embedding_json TEXT,
                 results_json TEXT,
                 hit_count INTEGER DEFAULT 0,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 last_hit TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             );
             CREATE INDEX IF NOT EXISTS idx_cache_hits ON query_cache(hit_count DESC);",
        )?;
        Ok(conn)
    }

    fn load_existing(conn: &Connection, state: &mut State) {
        let Ok(mut stmt) = conn.prepare("SELECT normalized, results_json FROM query_cache") else {
            return;
        };
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)));
        if let Ok(rows) = rows {
            for row in rows.flatten() {
                if let Ok(results) = serde_json::from_str::<Vec<MemoryRow>>(&row.1) {
                    state.tier1.insert(row.0.clone(), results.clone());
                    state.tier4.insert(row.0.clone(), results);
                    state.insert_times.insert(row.0, Instant::now());
                }
            }
        }
    }

    /// Normalize a query for the cache key (`query_cache.py` `_normalize` L133-L137): lowercase
    /// words longer than one char, sorted, space-joined. (Distinct from `synonyms::normalize_query`.)
    fn normalize(query: &str) -> String {
        let mut words: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.chars().count() > 1)
            .map(|w| w.to_lowercase())
            .collect();
        words.sort_unstable();
        words.join(" ")
    }

    /// Try to retrieve cached results for `query`, optionally using its `embedding` for the tier-2/3
    /// semantic match (`query_cache.py` `get` L167-L249).
    pub fn get(&self, query: &str, embedding: Option<&[f32]>) -> Option<Vec<MemoryRow>> {
        let normalized = Self::normalize(query);
        let mut s = self.inner.lock().unwrap();
        let now = Instant::now();

        // TTL expiry for the exact key.
        if let Some(t) = s.insert_times.get(&normalized).copied() {
            if now.duration_since(t) > self.ttl {
                s.tier1.remove(&normalized);
                s.tier23.remove(&normalized);
                s.tier4.remove(&normalized);
                s.insert_times.remove(&normalized);
                s.misses += 1;
                return None;
            }
        }

        // Tier 1: exact normalized match.
        if let Some(results) = s.tier1.get(&normalized).cloned() {
            s.hits += 1;
            return Some(results);
        }

        // Tier 2-3: embedding similarity over the in-memory entries.
        if let Some(emb) = embedding {
            let mut best_score = 0.0_f64;
            let mut best_key: Option<String> = None;
            for (key, (cached_emb, _)) in s.tier23.iter() {
                if let Some(t) = s.insert_times.get(key).copied() {
                    if now.duration_since(t) > self.ttl {
                        continue;
                    }
                }
                let cosine = cosine_padded(emb, cached_emb);
                if cosine >= TIER2_COSINE {
                    best_score = cosine;
                    best_key = Some(key.clone());
                    break; // high-confidence, take it
                }
                if cosine >= TIER3_COSINE {
                    let jaccard = jaccard_words(query, key);
                    if jaccard >= TIER3_JACCARD && cosine > best_score {
                        best_score = cosine;
                        best_key = Some(key.clone());
                    }
                }
            }
            if let Some(key) = best_key {
                let _ = best_score;
                s.hits += 1;
                return s.tier23.get(&key).map(|(_, r)| r.clone());
            }
        }

        // Tier 4: >= 70% word overlap (min 2 words).
        let query_words: std::collections::HashSet<&str> = normalized.split(' ').collect();
        let mut hit: Option<Vec<MemoryRow>> = None;
        for (key, results) in s.tier4.iter() {
            if let Some(t) = s.insert_times.get(key).copied() {
                if now.duration_since(t) > self.ttl {
                    continue;
                }
            }
            let cached_words: std::collections::HashSet<&str> = key.split(' ').collect();
            let overlap = query_words.intersection(&cached_words).count();
            if overlap as f64 >= query_words.len() as f64 * 0.7 && overlap >= 2 {
                hit = Some(results.clone());
                break;
            }
        }
        if let Some(results) = hit {
            s.hits += 1;
            return Some(results);
        }

        s.misses += 1;
        None
    }

    /// Store `results` in all applicable tiers (`query_cache.py` `put` L251-L281).
    pub fn put(&self, query: &str, results: &[MemoryRow], embedding: Option<&[f32]>) {
        let normalized = Self::normalize(query);
        let results = results.to_vec();
        let mut s = self.inner.lock().unwrap();
        s.tier1.insert(normalized.clone(), results.clone());
        s.insert_times.insert(normalized.clone(), Instant::now());
        if let Some(emb) = embedding {
            s.tier23
                .insert(normalized.clone(), (emb.to_vec(), results.clone()));
        }
        s.tier4.insert(normalized.clone(), results.clone());

        if let Some(conn) = &self.conn {
            if let Ok(results_json) = serde_json::to_string(&results) {
                let emb_json = embedding.and_then(|e| serde_json::to_string(e).ok());
                let _ = conn.lock().unwrap().execute(
                    "INSERT OR REPLACE INTO query_cache (normalized, embedding_json, results_json) \
                     VALUES (?1, ?2, ?3)",
                    rusqlite::params![normalized, emb_json, results_json],
                );
            }
        }

        self.evict_if_needed(&mut s);
    }

    /// Invalidate every cached query (`query_cache.py` `invalidate` L121-L131). Called after each
    /// `remember`.
    pub fn invalidate(&self) {
        let mut s = self.inner.lock().unwrap();
        s.version += 1;
        s.tier1.clear();
        s.tier23.clear();
        s.tier4.clear();
        s.insert_times.clear();
        if let Some(conn) = &self.conn {
            let _ = conn.lock().unwrap().execute("DELETE FROM query_cache", []);
        }
    }

    /// `(hits, misses, size, version)` snapshot (`query_cache.py` `stats` L330-L343).
    pub fn stats(&self) -> (u64, u64, usize, u64) {
        let s = self.inner.lock().unwrap();
        (s.hits, s.misses, s.tier1.len(), s.version)
    }

    /// Evict TTL-expired entries, then the oldest entries over `max_size` (`_evict_if_needed`
    /// L283-L309).
    fn evict_if_needed(&self, s: &mut State) {
        let now = Instant::now();
        let expired: Vec<String> = s
            .insert_times
            .iter()
            .filter(|(_, t)| now.duration_since(**t) > self.ttl)
            .map(|(k, _)| k.clone())
            .collect();
        for key in expired {
            s.tier1.remove(&key);
            s.tier23.remove(&key);
            s.tier4.remove(&key);
            s.insert_times.remove(&key);
        }
        let total = s.tier1.len();
        if total > self.max_size {
            let mut by_age: Vec<(String, Instant)> = s
                .insert_times
                .iter()
                .map(|(k, t)| (k.clone(), *t))
                .collect();
            by_age.sort_by_key(|(_, t)| *t);
            for (key, _) in by_age.into_iter().take(total - self.max_size) {
                s.tier1.remove(&key);
                s.tier23.remove(&key);
                s.tier4.remove(&key);
                s.insert_times.remove(&key);
            }
        }
    }
}

/// Cosine similarity, zero-padding the shorter vector (`query_cache.py` `_cosine_similarity`
/// L139-L157).
fn cosine_padded(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let n = a.len().max(b.len());
    let mut dot = 0.0_f64;
    let mut mag_a = 0.0_f64;
    let mut mag_b = 0.0_f64;
    for i in 0..n {
        let x = *a.get(i).unwrap_or(&0.0) as f64;
        let y = *b.get(i).unwrap_or(&0.0) as f64;
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a.sqrt() * mag_b.sqrt())
}

/// Word-level Jaccard similarity (`query_cache.py` `_jaccard_words` L159-L165).
fn jaccard_words(a: &str, b: &str) -> f64 {
    let wa: std::collections::HashSet<String> = a
        .to_lowercase()
        .split_whitespace()
        .map(String::from)
        .collect();
    let wb: std::collections::HashSet<String> = b
        .to_lowercase()
        .split_whitespace()
        .map(String::from)
        .collect();
    if wa.is_empty() || wb.is_empty() {
        return 0.0;
    }
    wa.intersection(&wb).count() as f64 / wa.union(&wb).count() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{MemoryRow, Tier};

    fn row(id: &str) -> MemoryRow {
        MemoryRow {
            id: id.to_string(),
            content: format!("content {id}"),
            source: "test".to_string(),
            importance: 0.5,
            veracity: "stated".to_string(),
            trust_tier: "STATED".to_string(),
            tier: Tier::Working,
            tier_level: 1,
            score: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn tier1_exact_hit() {
        let c = QueryCache::open(None);
        assert!(c.get("the auth flow", None).is_none());
        c.put("the auth flow", &[row("a")], None);
        let hit = c.get("flow auth the", None).expect("normalized exact hit");
        assert_eq!(hit[0].id, "a");
    }

    #[test]
    fn tier4_word_overlap() {
        let c = QueryCache::open(None);
        c.put("database password rotation policy", &[row("x")], None);
        // 3/4 overlap (>=70%, >=2 words) -> tier-4 hit.
        let hit = c.get("database password rotation", None);
        assert!(hit.is_some());
    }

    #[test]
    fn invalidate_clears() {
        let c = QueryCache::open(None);
        c.put("alpha beta", &[row("a")], None);
        c.invalidate();
        assert!(c.get("alpha beta", None).is_none());
    }

    #[test]
    fn tier2_embedding_hit() {
        let c = QueryCache::open(None);
        c.put("unrelated key one", &[row("e")], Some(&[1.0, 0.0, 0.0]));
        // A near-identical embedding under a different surface query -> tier-2 cosine hit.
        let hit = c.get("totally different words here", Some(&[0.99, 0.01, 0.0]));
        assert!(hit.is_some());
        assert_eq!(hit.unwrap()[0].id, "e");
    }
}
