//! The BEAM engine facade — port of `beam.py` `BeamMemory` (the `remember`/`recall`/`get_context`/
//! `sleep` surface, L2836 / L5027 / L3526 / L7576) plus the `memory.py` facade.
//!
//! As-built: `remember`/`get_context` plus a hybrid `recall` that gathers candidates across the
//! **working and episodic** tiers from FTS5 (`fts_working`/`fts_episodes`, BM25), the stored
//! embeddings (cosine), and a recency/importance fallback scan, then scores them
//! ([`crate::recall::scoring`]) with the FTS-blended lexical relevance, vector similarity, the MIB
//! `binary_bonus`, and the tier/veracity multipliers — merged, content-deduped, and MMR-diversified.
//! [`Engine::consolidate`] is a minimal WM->episodic promotion (no LLM summarization/degradation).
//! Knowledge ingestion (graph/fact bonuses) and full `sleep` remain TODO (port-spec P1).

use crate::config::MnemosyneConfig;
use crate::dynamics::typed_memory;
use crate::error::Result;
use crate::recall::{mmr, scoring};
use crate::store::Store;
use crate::{binary_vectors, sanitize, util};
use rusqlite::{params, Connection};
use std::collections::{HashMap, HashSet};

/// The vector-similarity floor that lets a vector-only hit survive the lexical gate (mirrors the
/// episodic candidate-drop rule `lexical < floor && sim < 0.65 -> drop`, `beam.py` L5720+).
const VEC_SIM_FLOOR: f64 = 0.65;

/// Which BEAM tier a row lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Hot, recent, auto-injected context.
    Working,
    /// Long-term consolidated memory.
    Episodic,
}

/// A recalled / stored memory row (the `recall` result shape, `beam.py` L5996+).
#[derive(Clone, Debug)]
pub struct MemoryRow {
    /// Memory id.
    pub id: String,
    /// Content text.
    pub content: String,
    /// Ingestion source.
    pub source: String,
    /// ISO timestamp.
    pub timestamp: String,
    /// Importance `[0, 1]`.
    pub importance: f64,
    /// Trust label (`stated`/`inferred`/`tool`/`imported`/`unknown`).
    pub veracity: String,
    /// Trust tier (`STATED`/`DERIVED`/...).
    pub trust_tier: String,
    /// Which tier the row came from.
    pub tier: Tier,
    /// The episodic tier level (`1`/`2`/`3`); working rows are always `1` (`beam.py` L5931).
    pub tier_level: i64,
    /// The recall score (0 for direct fetches).
    pub score: f64,
}

/// Arguments for [`Engine::remember`] (`beam.py` `remember` L2836).
#[derive(Clone, Debug)]
pub struct RememberArgs {
    /// Ingestion source (default `conversation`).
    pub source: String,
    /// Importance `[0, 1]` (default 0.5).
    pub importance: f64,
    /// Scope: `session` (default) or `global`. Note: the column default is `global` but
    /// `remember()` defaults to `session` (`beam.py` L2838).
    pub scope: String,
    /// Trust label (default `unknown`).
    pub veracity: String,
}

impl Default for RememberArgs {
    fn default() -> Self {
        Self {
            source: "conversation".to_string(),
            importance: 0.5,
            scope: "session".to_string(),
            veracity: "unknown".to_string(),
        }
    }
}

/// The BEAM engine over a single bank store.
pub struct Engine {
    store: Store,
    config: MnemosyneConfig,
}

impl Engine {
    /// Open the engine for the configured bank.
    pub fn open(config: MnemosyneConfig) -> Result<Self> {
        let store = Store::open(config.bank_db_path())?;
        Ok(Self { store, config })
    }

    /// Open an ephemeral in-memory engine (tests).
    pub fn open_in_memory(config: MnemosyneConfig) -> Result<Self> {
        let store = Store::open_in_memory()?;
        Ok(Self { store, config })
    }

    /// The active session id.
    pub fn session_id(&self) -> &str {
        &self.config.session_id
    }

    /// Store a memory in the working tier (`beam.py` `remember` L2836), keyword-only (no vector).
    /// Equivalent to [`Engine::remember_with_vector`] with no embedding.
    pub fn remember(&self, content: &str, args: &RememberArgs) -> Result<String> {
        self.remember_with_vector(content, args, None, "")
    }

    /// Store a memory in the working tier, optionally persisting a precomputed embedding into
    /// `memory_embeddings` (the f32-BLOB-as-JSON fallback store).
    ///
    /// The embedding is computed by the caller (the async [`MnemosyneProvider`] hooks) and passed in,
    /// so the synchronous engine never blocks on a model call. Scaffold: sanitize + classify +
    /// insert + vector write (dedup and knowledge ingestion remain TODO).
    pub fn remember_with_vector(
        &self,
        content: &str,
        args: &RememberArgs,
        vector: Option<&[f32]>,
        model: &str,
    ) -> Result<String> {
        let (content, _meta) = sanitize::sanitize_content(content);
        let id = util::memory_id(&format!("{}:{}", self.config.session_id, content));
        let memory_type = typed_memory::classify(&content).as_str();
        let now = util::now_iso();
        let conn = self.store.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO working_memory \
             (id, content, source, timestamp, session_id, importance, metadata_json, veracity, \
              memory_type, scope) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, '{}', ?7, ?8, ?9)",
            params![
                id,
                content,
                args.source,
                now,
                self.config.session_id,
                args.importance,
                args.veracity,
                memory_type,
                args.scope,
            ],
        )?;
        if let Some(vector) = vector {
            let embedding_json = serde_json::to_string(vector)?;
            conn.execute(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model) \
                 VALUES (?1, ?2, ?3)",
                params![id, embedding_json, model],
            )?;
        }
        Ok(id)
    }

    /// Auto-inject context: global then session-local working memory ordered by importance/recency
    /// (`beam.py` `get_context` L3526-L3606).
    pub fn get_context(&self, limit: usize) -> Result<Vec<MemoryRow>> {
        let conn = self.store.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier \
             FROM working_memory \
             WHERE (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?1 OR scope = 'global') \
             ORDER BY importance DESC, timestamp DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![self.config.session_id, limit as i64], |r| {
                Ok(MemoryRow {
                    id: r.get(0)?,
                    content: r.get(1)?,
                    source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    importance: r.get(4)?,
                    veracity: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    trust_tier: r.get::<_, Option<String>>(6)?.unwrap_or_default(),
                    tier: Tier::Working,
                    tier_level: 1,
                    score: 0.0,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Linear-hybrid recall over working memory (`beam.py` `recall` L5027), keyword-only. Equivalent
    /// to [`Engine::recall_with_vector`] with no query vector.
    pub fn recall(&self, query: &str, top_k: usize) -> Result<Vec<MemoryRow>> {
        self.recall_with_vector(query, top_k, None)
    }

    /// Hybrid lexical + FTS5 + vector recall across the working **and** episodic tiers (`beam.py`
    /// `recall` L5027, candidate gathering L2423-L2597 / finalize L5996-L6119).
    ///
    /// Each tier gathers candidates from three sources — an FTS5 `MATCH` (BM25-ranked), the stored
    /// embeddings (cosine), and a recency/importance fallback scan — then scores them: working rows
    /// with [`scoring::working_memory_score`] over an FTS-blended relevance ([`scoring::blend_fts`]),
    /// episodic rows with [`scoring::episodic_score`] (tier + veracity multipliers, MIB binary
    /// bonus). A row survives the candidate gate if it clears the lexical floor, is an FTS hit, **or**
    /// its vector similarity clears [`VEC_SIM_FLOOR`]. Tiers are merged, deduped by content, MMR-
    /// diversified for `>=4`-token queries, and the surviving rows have their recall stats bumped.
    /// With `query_vector = None` the vector source is skipped (pure lexical + FTS recall).
    pub fn recall_with_vector(
        &self,
        query: &str,
        top_k: usize,
        query_vector: Option<&[f32]>,
    ) -> Result<Vec<MemoryRow>> {
        let q_tokens = tokenize(query);
        let floor = scoring::lexical_floor(q_tokens.len());
        let conn = self.store.conn.lock().unwrap();

        let mut scored = self.gather_working(&conn, &q_tokens, top_k, floor, query_vector)?;
        let episodic = self.gather_episodic(&conn, &q_tokens, top_k, floor, query_vector)?;
        scored.extend(episodic);

        // Cross-tier dedup by normalized content, keeping the higher-scoring row (`beam.py` L6003).
        dedup_by_content(&mut scored);
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Diversity rerank for multi-token queries (`beam.py` L6061), else a plain top-k slice.
        let selected: Vec<MemoryRow> = if q_tokens.len() >= 4 && scored.len() > 1 {
            let items: Vec<(String, f64)> =
                scored.iter().map(|r| (r.content.clone(), r.score)).collect();
            mmr::mmr_rerank(&items, mmr::DEFAULT_LAMBDA, top_k)
                .into_iter()
                .map(|i| scored[i].clone())
                .collect()
        } else {
            scored.truncate(top_k);
            scored
        };

        self.bump_recall(&conn, &selected)?;
        Ok(selected)
    }

    /// Gather + score working-memory candidates (FTS5 ∪ vector ∪ recency fallback).
    fn gather_working(
        &self,
        conn: &Connection,
        q_tokens: &[String],
        top_k: usize,
        floor: f64,
        query_vector: Option<&[f32]>,
    ) -> Result<Vec<MemoryRow>> {
        // Base candidates: the recency/importance fallback scan (limit 2000, `beam.py` L5262), plus
        // any FTS5 hits that fall outside that window.
        let mut rows = self.scan_working(conn, 2000)?;
        let mut seen: HashSet<String> = rows.iter().map(|r| r.id.clone()).collect();
        let fts = self.fts_search(
            conn,
            "SELECT id, bm25(fts_working) FROM fts_working \
             WHERE fts_working MATCH ?1 ORDER BY bm25(fts_working) LIMIT ?2",
            q_tokens,
            (top_k * 3).max(50),
        )?;
        for id in fts.keys() {
            if seen.insert(id.clone()) {
                if let Some(row) = self.fetch_working(conn, id)? {
                    rows.push(row);
                }
            }
        }

        let stored = match query_vector {
            Some(_) => load_embeddings(conn)?,
            None => HashMap::new(),
        };
        let (_vw, _fw, iw) = scoring::DEFAULT_WEIGHTS;

        let mut scored = Vec::new();
        for mut row in rows {
            let lexical = lexical_relevance(q_tokens, &row.content);
            let nfts = fts.get(&row.id).copied().unwrap_or(0.0);
            let vec_sim = match (query_vector, stored.get(&row.id)) {
                (Some(q), Some(v)) => daemon_core::cosine(q, v) as f64,
                _ => 0.0,
            };
            if lexical < floor && vec_sim < VEC_SIM_FLOOR && nfts <= 0.0 {
                continue;
            }
            let relevance = scoring::blend_fts(lexical, nfts, floor);
            let decay = scoring::recency_decay(age_hours(&row.timestamp));
            let base = scoring::working_memory_score(relevance, row.importance, iw, vec_sim, decay);
            row.score = base * scoring::veracity_multiplier(&row.veracity);
            scored.push(row);
        }
        Ok(scored)
    }

    /// Gather + score episodic candidates (FTS5 ∪ vector ∪ recency fallback), with the MIB binary
    /// bonus and the tier/veracity post-multipliers (`beam.py` L5720-L5976).
    fn gather_episodic(
        &self,
        conn: &Connection,
        q_tokens: &[String],
        top_k: usize,
        floor: f64,
        query_vector: Option<&[f32]>,
    ) -> Result<Vec<MemoryRow>> {
        let mut rows = self.scan_episodic(conn, 2000)?;
        let mut seen: HashSet<String> = rows.iter().map(|r| r.id.clone()).collect();
        let fts = self.fts_search(
            conn,
            "SELECT e.id, bm25(fts_episodes) FROM fts_episodes f \
             JOIN episodic_memory e ON e.rowid = f.rowid \
             WHERE fts_episodes MATCH ?1 ORDER BY bm25(fts_episodes) LIMIT ?2",
            q_tokens,
            (top_k * 3).max(20),
        )?;
        for id in fts.keys() {
            if seen.insert(id.clone()) {
                if let Some(row) = self.fetch_episodic(conn, id)? {
                    rows.push(row);
                }
            }
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let stored = match query_vector {
            Some(_) => load_embeddings(conn)?,
            None => HashMap::new(),
        };
        let binaries = self.load_binary_vectors(conn)?;
        let q_bin = query_vector.map(binary_vectors::maximally_informative_binarization);
        let weights = scoring::DEFAULT_WEIGHTS;

        let mut scored = Vec::new();
        for mut row in rows {
            let lexical = lexical_relevance(q_tokens, &row.content);
            let nfts = fts.get(&row.id).copied().unwrap_or(0.0);
            let sim = match (query_vector, stored.get(&row.id)) {
                (Some(q), Some(v)) => daemon_core::cosine(q, v) as f64,
                _ => 0.0,
            };
            // Weak-signal gate (`beam.py` L5720): drop unless lexical, FTS, or vector say keep.
            if lexical < floor && sim < VEC_SIM_FLOOR && nfts <= 0.0 {
                continue;
            }
            let binary_bonus = match (&q_bin, binaries.get(&row.id)) {
                (Some(qb), Some(rb)) => {
                    let dist = binary_vectors::hamming_distance(qb, rb);
                    binary_vectors::binary_bonus(dist as f64 / binary_vectors::EMBEDDING_DIM as f64)
                }
                _ => 0.0,
            };
            let decay = scoring::recency_decay(age_hours(&row.timestamp));
            // Graph/fact bonuses are 0 until the knowledge layer lands (P1); threaded for parity.
            let base = scoring::episodic_score(
                sim,
                nfts,
                row.importance,
                lexical,
                decay,
                weights,
                0.0,
                0.0,
                binary_bonus,
            );
            row.score = base
                * scoring::tier_weight(row.tier_level)
                * scoring::veracity_multiplier(&row.veracity);
            scored.push(row);
        }
        Ok(scored)
    }

    /// Promote unconsolidated working-memory rows into the episodic tier (a minimal slice of
    /// `beam.py` `sleep`/consolidation L7576: no LLM summarization or tier degradation yet). Each
    /// promoted row is copied into `episodic_memory` at tier 1 — computing its MIB `binary_vector`
    /// from any stored embedding — its source working row is marked `consolidated_at`, and a
    /// `consolidation_log` entry is written. Returns the number of rows promoted.
    pub fn consolidate(&self) -> Result<usize> {
        let conn = self.store.conn.lock().unwrap();
        let embeddings = load_embeddings(&conn)?;
        let mut pending: Vec<EpisodicSeed> = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT id, content, source, timestamp, importance, veracity, trust_tier, scope, \
                        memory_type \
                 FROM working_memory \
                 WHERE consolidated_at IS NULL AND session_id = ?1 AND superseded_by IS NULL",
            )?;
            let rows = stmt.query_map(params![self.config.session_id], |r| {
                Ok(EpisodicSeed {
                    wm_id: r.get(0)?,
                    content: r.get(1)?,
                    source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    importance: r.get(4)?,
                    veracity: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    trust_tier: r.get::<_, Option<String>>(6)?.unwrap_or_default(),
                    scope: r.get::<_, Option<String>>(7)?.unwrap_or_else(|| "global".into()),
                    memory_type: r
                        .get::<_, Option<String>>(8)?
                        .unwrap_or_else(|| "unknown".into()),
                })
            })?;
            for row in rows {
                pending.push(row?);
            }
        }
        if pending.is_empty() {
            return Ok(0);
        }

        let now = util::now_iso();
        let mut count = 0usize;
        for seed in &pending {
            let ep_id =
                util::memory_id(&format!("episodic:{}:{}", self.config.session_id, seed.content));
            let binary = embeddings
                .get(&seed.wm_id)
                .map(|v| binary_vectors::maximally_informative_binarization(v));
            conn.execute(
                "INSERT OR IGNORE INTO episodic_memory \
                 (id, content, source, timestamp, session_id, importance, metadata_json, veracity, \
                  memory_type, tier, binary_vector, scope, trust_tier, summary_of) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, '{}', ?7, ?8, 1, ?9, ?10, ?11, '')",
                params![
                    ep_id,
                    seed.content,
                    seed.source,
                    seed.timestamp,
                    self.config.session_id,
                    seed.importance,
                    seed.veracity,
                    seed.memory_type,
                    binary,
                    seed.scope,
                    seed.trust_tier,
                ],
            )?;
            conn.execute(
                "UPDATE working_memory SET consolidated_at = ?2 WHERE id = ?1",
                params![seed.wm_id, now],
            )?;
            count += 1;
        }
        let preview: String = pending
            .iter()
            .map(|p| p.content.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
            .chars()
            .take(200)
            .collect();
        conn.execute(
            "INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview) \
             VALUES (?1, ?2, ?3)",
            params![self.config.session_id, count as i64, preview],
        )?;
        Ok(count)
    }

    /// Run an FTS5 `MATCH` query (`sql` selecting `(id, bm25)`), returning `id -> normalized BM25`
    /// for the hits. An empty token list (or a query with no usable terms) yields no hits.
    fn fts_search(
        &self,
        conn: &Connection,
        sql: &str,
        q_tokens: &[String],
        limit: usize,
    ) -> Result<HashMap<String, f64>> {
        let Some(match_str) = fts_match_string(q_tokens) else {
            return Ok(HashMap::new());
        };
        let mut stmt = conn.prepare(sql)?;
        let mut map = HashMap::new();
        let rows = stmt.query_map(params![match_str, limit as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
        })?;
        for row in rows {
            let (id, bm25) = row?;
            map.insert(id, normalize_bm25(bm25));
        }
        Ok(map)
    }

    /// Recency/importance fallback scan over working memory (the candidate floor, scope-filtered).
    fn scan_working(&self, conn: &Connection, limit: usize) -> Result<Vec<MemoryRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier \
             FROM working_memory \
             WHERE (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?1 OR scope = 'global') \
             ORDER BY importance DESC, timestamp DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![self.config.session_id, limit as i64], |r| {
                Ok(working_row(r))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Fetch a single working row by id (for FTS hits beyond the fallback window), scope-filtered.
    fn fetch_working(&self, conn: &Connection, id: &str) -> Result<Option<MemoryRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier \
             FROM working_memory \
             WHERE id = ?1 AND (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?2 OR scope = 'global')",
        )?;
        let mut rows = stmt.query_map(params![id, self.config.session_id], |r| Ok(working_row(r)))?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Recency/importance fallback scan over episodic memory, scope-filtered.
    fn scan_episodic(&self, conn: &Connection, limit: usize) -> Result<Vec<MemoryRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier, tier \
             FROM episodic_memory \
             WHERE (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?1 OR scope = 'global') \
             ORDER BY importance DESC, timestamp DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![self.config.session_id, limit as i64], |r| {
                Ok(episodic_row(r))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Fetch a single episodic row by id (for FTS hits beyond the fallback window), scope-filtered.
    fn fetch_episodic(&self, conn: &Connection, id: &str) -> Result<Option<MemoryRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier, tier \
             FROM episodic_memory \
             WHERE id = ?1 AND (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?2 OR scope = 'global')",
        )?;
        let mut rows =
            stmt.query_map(params![id, self.config.session_id], |r| Ok(episodic_row(r)))?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Load the packed MIB `binary_vector` blobs for episodic rows, keyed by memory id.
    fn load_binary_vectors(&self, conn: &Connection) -> Result<HashMap<String, Vec<u8>>> {
        let mut stmt = conn
            .prepare("SELECT id, binary_vector FROM episodic_memory WHERE binary_vector IS NOT NULL")?;
        let mut map = HashMap::new();
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
        for row in rows {
            let (id, blob) = row?;
            map.insert(id, blob);
        }
        Ok(map)
    }

    /// Bump `recall_count` / `last_recalled` for the returned rows in their source tier (`beam.py`
    /// L6084-L6119).
    fn bump_recall(&self, conn: &Connection, rows: &[MemoryRow]) -> Result<()> {
        let now = util::now_iso();
        for row in rows {
            let table = match row.tier {
                Tier::Working => "working_memory",
                Tier::Episodic => "episodic_memory",
            };
            conn.execute(
                &format!(
                    "UPDATE {table} SET recall_count = recall_count + 1, last_recalled = ?2 \
                     WHERE id = ?1"
                ),
                params![row.id, now],
            )?;
        }
        Ok(())
    }
}

/// A working-memory row queued for promotion into the episodic tier ([`Engine::consolidate`]).
struct EpisodicSeed {
    wm_id: String,
    content: String,
    source: String,
    timestamp: String,
    importance: f64,
    veracity: String,
    trust_tier: String,
    scope: String,
    memory_type: String,
}

/// Load the stored f32 embeddings (`memory_embeddings.embedding_json`), keyed by memory id.
fn load_embeddings(conn: &Connection) -> Result<HashMap<String, Vec<f32>>> {
    let mut stmt = conn.prepare("SELECT memory_id, embedding_json FROM memory_embeddings")?;
    let mut map = HashMap::new();
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    for row in rows {
        let (id, json) = row?;
        if let Ok(vec) = serde_json::from_str::<Vec<f32>>(&json) {
            map.insert(id, vec);
        }
    }
    Ok(map)
}

/// Map a working-memory result row (`id, content, source, timestamp, importance, veracity,
/// trust_tier`) into a [`MemoryRow`] at tier [`Tier::Working`].
fn working_row(r: &rusqlite::Row<'_>) -> MemoryRow {
    MemoryRow {
        id: r.get(0).unwrap_or_default(),
        content: r.get(1).unwrap_or_default(),
        source: r.get::<_, Option<String>>(2).ok().flatten().unwrap_or_default(),
        timestamp: r.get::<_, Option<String>>(3).ok().flatten().unwrap_or_default(),
        importance: r.get(4).unwrap_or(0.5),
        veracity: r.get::<_, Option<String>>(5).ok().flatten().unwrap_or_default(),
        trust_tier: r.get::<_, Option<String>>(6).ok().flatten().unwrap_or_default(),
        tier: Tier::Working,
        tier_level: 1,
        score: 0.0,
    }
}

/// Map an episodic result row (working columns + `tier`) into a [`MemoryRow`] at tier
/// [`Tier::Episodic`], carrying the integer tier level for the post-multiplier.
fn episodic_row(r: &rusqlite::Row<'_>) -> MemoryRow {
    MemoryRow {
        id: r.get(0).unwrap_or_default(),
        content: r.get(1).unwrap_or_default(),
        source: r.get::<_, Option<String>>(2).ok().flatten().unwrap_or_default(),
        timestamp: r.get::<_, Option<String>>(3).ok().flatten().unwrap_or_default(),
        importance: r.get(4).unwrap_or(0.5),
        veracity: r.get::<_, Option<String>>(5).ok().flatten().unwrap_or_default(),
        trust_tier: r.get::<_, Option<String>>(6).ok().flatten().unwrap_or_default(),
        tier: Tier::Episodic,
        tier_level: r.get::<_, Option<i64>>(7).ok().flatten().unwrap_or(1),
        score: 0.0,
    }
}

/// Tokenize text into lowercase alphanumeric terms (the shared query/content tokenizer).
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Build an FTS5 `MATCH` expression from query tokens (`"a" OR "b" OR ...`), quoting each term so
/// punctuation/operators can never break the query. `None` when there are no usable terms.
fn fts_match_string(q_tokens: &[String]) -> Option<String> {
    if q_tokens.is_empty() {
        return None;
    }
    let parts: Vec<String> = q_tokens
        .iter()
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    Some(parts.join(" OR "))
}

/// Map SQLite FTS5 `bm25()` (more-negative = better) onto `[0, 1)` (`raw / (1 + raw)`), so a missed
/// row contributes `0` and a strong match approaches `1`.
fn normalize_bm25(bm25: f64) -> f64 {
    let raw = (-bm25).max(0.0);
    raw / (1.0 + raw)
}

/// Normalized content key for cross-tier dedup (lowercased alphanumeric tokens, space-joined).
fn normalized_content(content: &str) -> String {
    tokenize(content).join(" ")
}

/// Drop duplicate rows sharing the same [`normalized_content`], keeping the highest-scoring one
/// (`beam.py` cross-tier summary dedup L6003).
fn dedup_by_content(rows: &mut Vec<MemoryRow>) {
    let mut best: HashMap<String, usize> = HashMap::new();
    let mut keep = vec![true; rows.len()];
    for i in 0..rows.len() {
        let key = normalized_content(&rows[i].content);
        match best.get(&key).copied() {
            Some(j) => {
                if rows[i].score > rows[j].score {
                    keep[j] = false;
                    best.insert(key, i);
                } else {
                    keep[i] = false;
                }
            }
            None => {
                best.insert(key, i);
            }
        }
    }
    let mut idx = 0;
    rows.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Lexical relevance (`beam.py` L1573-L1638): `(exact_token_hits + partial + full_match)/len`, where
/// a `>=4`-char substring overlap adds `0.4` and a whole-query substring adds `1.0`. Clamped to
/// `[0, 1]`. (Synonym `+0.75` matching is a `synonyms.rs` TODO — P2.)
fn lexical_relevance(query_tokens: &[String], content: &str) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let lc = content.to_lowercase();
    let content_tokens: HashSet<String> = tokenize(content).into_iter().collect();
    let mut num = 0.0;
    for t in query_tokens {
        if content_tokens.contains(t) {
            num += 1.0;
        } else if t.len() >= 4 && lc.contains(t.as_str()) {
            num += 0.4;
        }
    }
    if lc.contains(&query_tokens.join(" ")) {
        num += 1.0;
    }
    (num / query_tokens.len() as f64).min(1.0)
}

/// Hours since an ISO timestamp (`None` if unparseable -> decay falls back to 0.5).
fn age_hours(timestamp: &str) -> Option<f64> {
    let parsed = chrono::DateTime::parse_from_rfc3339(timestamp).ok()?;
    let now = chrono::Utc::now();
    let delta = now.signed_duration_since(parsed.with_timezone(&chrono::Utc));
    Some(delta.num_seconds().max(0) as f64 / 3600.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        Engine::open_in_memory(MnemosyneConfig::default()).expect("engine")
    }

    #[test]
    fn remember_then_recall() {
        let e = engine();
        e.remember(
            "the authentication flow uses JWT tokens",
            &RememberArgs::default(),
        )
        .unwrap();
        e.remember("lunch was pizza", &RememberArgs::default())
            .unwrap();
        let hits = e.recall("authentication flow", 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].content.contains("authentication"));
    }

    #[test]
    fn session_scoping_over_shared_bank() {
        // Two engines over the *same* agent-wide bank, each bound to its own session id (the
        // per-session construction the composition layer's `MnemosyneBanks` performs). Session-scoped
        // rows must not leak across sessions, while `scope='global'` rows are visible to both.
        let dir = std::env::temp_dir().join(format!("mnemosyne-scope-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = |sid: &str| MnemosyneConfig {
            data_dir: dir.clone(),
            session_id: sid.to_string(),
            ..MnemosyneConfig::default()
        };
        let s1 = Engine::open(cfg("s1")).expect("open s1");
        let s2 = Engine::open(cfg("s2")).expect("open s2");

        s1.remember("alpha private to one", &RememberArgs::default())
            .unwrap();
        s2.remember("beta private to two", &RememberArgs::default())
            .unwrap();
        s1.remember(
            "gamma shared globally",
            &RememberArgs {
                scope: "global".to_string(),
                ..Default::default()
            },
        )
        .unwrap();

        // Each session sees its own session-scoped row...
        assert!(!s1.recall("alpha", 5).unwrap().is_empty());
        assert!(!s2.recall("beta", 5).unwrap().is_empty());
        // ...but not the other session's.
        assert!(
            s1.recall("beta", 5).unwrap().is_empty(),
            "s1 must not see s2's row"
        );
        assert!(
            s2.recall("alpha", 5).unwrap().is_empty(),
            "s2 must not see s1's row"
        );
        // The global row is visible to both.
        assert!(!s1.recall("gamma", 5).unwrap().is_empty());
        assert!(
            !s2.recall("gamma", 5).unwrap().is_empty(),
            "global row visible across sessions"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn vector_recall_surfaces_semantic_match_lexical_misses() {
        let e = engine();
        // A query vector, one near-parallel memory vector (cos ~0.96) and one orthogonal — with
        // content that shares NO tokens with the query, so lexical recall finds nothing.
        let q = [1.0f32, 0.0, 0.0];
        let near = [0.96f32, 0.28, 0.0];
        let far = [0.0f32, 0.0, 1.0];
        e.remember_with_vector("alpha apple", &RememberArgs::default(), Some(&near), "mock")
            .unwrap();
        e.remember_with_vector("beta banana", &RememberArgs::default(), Some(&far), "mock")
            .unwrap();

        // Lexical-only recall for a disjoint query returns nothing.
        assert!(e.recall("zzz", 5).unwrap().is_empty());

        // Vector recall surfaces the semantically-close memory and ranks it first.
        let hits = e.recall_with_vector("zzz", 5, Some(&q)).unwrap();
        assert!(!hits.is_empty(), "vector recall should surface a match");
        assert_eq!(hits[0].content, "alpha apple");
        assert!(
            hits.iter().all(|h| h.content != "beta banana"),
            "orthogonal memory must not pass the vector gate"
        );
    }

    #[test]
    fn get_context_orders_by_importance() {
        let e = engine();
        e.remember(
            "low",
            &RememberArgs {
                importance: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        e.remember(
            "high",
            &RememberArgs {
                importance: 0.9,
                ..Default::default()
            },
        )
        .unwrap();
        let ctx = e.get_context(10).unwrap();
        assert_eq!(ctx[0].content, "high");
    }

    #[test]
    fn lexical_relevance_scores() {
        let q = vec!["auth".to_string(), "flow".to_string()];
        // Both tokens present as whole words + full-query substring -> clamped to 1.0.
        assert!((lexical_relevance(&q, "the auth flow uses jwt") - 1.0).abs() < 1e-9);
        // One exact token of two -> 0.5.
        assert!((lexical_relevance(&q, "the auth subsystem") - 0.5).abs() < 1e-9);
        // A >=4-char substring (no whole-word match, and the full query is not a substring)
        // contributes the 0.4 partial: one of two tokens at 0.4 -> 0.2.
        let q2 = vec!["serialize".to_string(), "absent".to_string()];
        assert!((lexical_relevance(&q2, "the deserializer ran") - 0.2).abs() < 1e-9);
        // Disjoint query -> 0.0; empty query -> 0.0.
        assert_eq!(lexical_relevance(&q, "completely unrelated"), 0.0);
        assert_eq!(lexical_relevance(&[], "anything"), 0.0);
    }

    #[test]
    fn fts_surfaces_row_beyond_recency_window() {
        // Fill the recency/importance window (limit 2000) with high-importance filler that does NOT
        // contain the marker, then add one low-importance row that does. The marker row ranks 2001st
        // by importance, so it is *outside* the fallback scan — only the FTS5 candidate path can
        // surface it. (Under the old full-scan recall this row was unreachable.)
        let e = engine();
        for i in 0..2000 {
            e.remember(
                &format!("filler row number {i}"),
                &RememberArgs {
                    importance: 0.9,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        e.remember(
            "a unique zqxj marker lives here",
            &RememberArgs {
                importance: 0.1,
                ..Default::default()
            },
        )
        .unwrap();

        let hits = e.recall("zqxj", 5).unwrap();
        assert!(
            hits.iter().any(|h| h.content.contains("zqxj")),
            "FTS5 must surface a row outside the recency window"
        );
    }

    #[test]
    fn consolidation_populates_episodic_and_is_idempotent() {
        let e = engine();
        e.remember("blue-green deployment rollout strategy", &RememberArgs::default())
            .unwrap();
        e.remember("margherita pizza for lunch", &RememberArgs::default())
            .unwrap();

        assert_eq!(e.consolidate().unwrap(), 2, "both WM rows promoted");
        assert_eq!(e.consolidate().unwrap(), 0, "already-consolidated rows are skipped");

        let conn = e.store.conn.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT count(*) FROM episodic_memory", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2);
        let logged: i64 = conn
            .query_row(
                "SELECT count(*) FROM consolidation_log WHERE items_consolidated = 2",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(logged, 1);
    }

    #[test]
    fn episodic_recall_after_consolidation_dedups_cross_tier() {
        let e = engine();
        e.remember("the deployment uses a blue-green rollout", &RememberArgs::default())
            .unwrap();
        e.consolidate().unwrap();

        // The content now lives in BOTH tiers; recall must surface it exactly once (cross-tier dedup).
        let hits = e.recall("deployment rollout", 5).unwrap();
        let matches: Vec<_> = hits
            .iter()
            .filter(|h| h.content.contains("blue-green"))
            .collect();
        assert_eq!(matches.len(), 1, "cross-tier duplicate collapsed to one row");
    }

    #[test]
    fn episodic_vector_recall_uses_binary_and_cosine() {
        // Promote two memories with stored embeddings (consolidate also packs MIB binary vectors),
        // then recall by a query vector parallel to one of them with NO lexical overlap. Only the
        // episodic vector + binary path can surface it.
        let e = engine();
        let near = [0.96f32, 0.28, 0.0];
        let far = [0.0f32, 0.0, 1.0];
        e.remember_with_vector("alpha apple", &RememberArgs::default(), Some(&near), "mock")
            .unwrap();
        e.remember_with_vector("beta banana", &RememberArgs::default(), Some(&far), "mock")
            .unwrap();
        e.consolidate().unwrap();

        let q = [1.0f32, 0.0, 0.0];
        let hits = e.recall_with_vector("zzz", 5, Some(&q)).unwrap();
        assert!(
            hits.iter().any(|h| h.content == "alpha apple"),
            "episodic vector recall should surface the semantically-close memory"
        );
        assert!(
            hits.iter().all(|h| h.content != "beta banana"),
            "the orthogonal memory must not pass the vector gate"
        );
    }
}
