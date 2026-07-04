// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Hybrid recall for the BEAM [`Engine`] — the faithful port of `beam.py` `recall` (L5027-L6210)
//! plus `get_context`, `fact_recall`, and the enhanced/polyphonic pipelines.
//!
//! The linear path follows Python's exact stage order: WM FTS5 + vector candidates (with the
//! recency fallback), the WM scoring loop, entity-aware then fact-aware boosts, the episodic
//! vec+FTS hybrid loop (graph/fact/binary bonuses) with its own fallback, tier + veracity
//! post-multipliers, cross-tier summary dedup, the MEMORIA supplement, multi-aspect greedy
//! selection, the scoped recall-count bump, and the C4 provenance diagnostics.

use super::*;
use crate::config::{RecallFilters, RecallMode, RecallScope};
use crate::dynamics::weibull;
use crate::knowledge::{annotations, entities, episodic_graph};
use crate::recall::lexical::{
    cjk_chars, cyrillic_score, cyrillic_words, expanded_query_tokens, fts_query_terms, has_cjk,
    has_cyrillic, lexical_relevance, recall_tokens, round4, strict_fact_matches, truncate_chars,
};
use crate::recall::{mmr, polyphonic, query_intent, scoring, synonyms};
use crate::{memoria, util};
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};

/// The shared candidate projection for the WM / entity / fact SELECTs (`beam.py` L5259).
const CAND_COLS: &str = "id, content, source, timestamp, importance, recall_count, \
     last_recalled, scope, author_id, author_type, channel_id, veracity, valid_until";

/// A fetched candidate row (the columns the scoring loops read).
struct Cand {
    rowid: i64,
    id: String,
    content: String,
    source: String,
    timestamp: String,
    importance: f64,
    recall_count: i64,
    last_recalled: Option<String>,
    scope: String,
    author_id: Option<String>,
    author_type: Option<String>,
    channel_id: Option<String>,
    veracity: String,
    valid_until: Option<String>,
    binary_vector: Option<Vec<u8>>,
}

impl Cand {
    fn from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            rowid: 0,
            id: r.get(0)?,
            content: r.get(1)?,
            source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
            importance: r.get::<_, Option<f64>>(4)?.unwrap_or(0.5),
            recall_count: r.get::<_, Option<i64>>(5)?.unwrap_or(0),
            last_recalled: r.get(6)?,
            scope: r.get::<_, Option<String>>(7)?.unwrap_or_default(),
            author_id: r.get(8)?,
            author_type: r.get(9)?,
            channel_id: r.get(10)?,
            veracity: r
                .get::<_, Option<String>>(11)?
                .unwrap_or_else(|| "unknown".to_string()),
            valid_until: r.get(12)?,
            binary_vector: None,
        })
    }

    /// The episodic projection adds `rowid` (index 13) and `binary_vector` (14).
    fn from_row_episodic(r: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        let mut c = Self::from_row(r)?;
        c.rowid = r.get(13)?;
        c.binary_vector = r.get(14)?;
        Ok(c)
    }

    /// The base recall result row for this candidate (`beam.py` L5344-L5366): identity, provenance,
    /// and bookkeeping columns; the caller fills the per-path score fields.
    fn to_row(&self, tier: Tier) -> MemoryRow {
        MemoryRow {
            id: self.id.clone(),
            content: truncate_chars(&self.content, 500),
            source: self.source.clone(),
            timestamp: self.timestamp.clone(),
            importance: self.importance,
            veracity: self.veracity.clone(),
            tier,
            tier_level: 1,
            recall_count: self.recall_count,
            last_recalled: self.last_recalled.clone(),
            scope: self.scope.clone(),
            author_id: self.author_id.clone(),
            author_type: self.author_type.clone(),
            channel_id: self.channel_id.clone(),
            valid_until: self.valid_until.clone(),
            ..Default::default()
        }
    }
}

/// The per-call C4 kept-row accumulators (`beam.py` L5152-L5166).
#[derive(Default)]
struct DiagCounts {
    wm_fts: usize,
    wm_vec: usize,
    wm_fallback: usize,
    em_fts: usize,
    em_vec: usize,
    em_fallback: usize,
    wm_fallback_used: bool,
    em_fallback_used: bool,
}

impl Engine {
    /// Auto-inject context: global then session-local working memory, each ordered by
    /// importance/recency, with the capped recall bump (`beam.py` `get_context` L3526-L3606).
    pub fn get_context(&self, limit: usize) -> Result<Vec<MemoryRow>> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();
        let select = "SELECT id, content, source, timestamp, importance, scope, last_recalled, \
             veracity, trust_tier FROM working_memory";
        let common = "(valid_until IS NULL OR valid_until > ?) AND superseded_by IS NULL";
        let map = |r: &rusqlite::Row<'_>| -> rusqlite::Result<(MemoryRow, Option<String>)> {
            Ok((
                MemoryRow {
                    id: r.get(0)?,
                    content: r.get(1)?,
                    source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    importance: r.get::<_, Option<f64>>(4)?.unwrap_or(0.5),
                    scope: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    veracity: r.get::<_, Option<String>>(7)?.unwrap_or_default(),
                    trust_tier: r.get::<_, Option<String>>(8)?.unwrap_or_default(),
                    tier: Tier::Working,
                    tier_level: 1,
                    ..Default::default()
                },
                r.get(6)?,
            ))
        };

        // Global rows first, then session-local up to the remaining budget (`beam.py` L3547-L3569).
        let mut stmt = conn.prepare(&format!(
            "{select} WHERE scope = 'global' AND {common} \
             ORDER BY importance DESC, timestamp DESC LIMIT ?"
        ))?;
        let mut rows: Vec<(MemoryRow, Option<String>)> = stmt
            .query_map(params![now, limit as i64], map)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if rows.len() < limit {
            let session_limit = (limit - rows.len()) as i64;
            let mut stmt = conn.prepare(&format!(
                "{select} WHERE session_id = ? AND (scope IS NULL OR scope != 'global') \
                 AND {common} ORDER BY importance DESC, timestamp DESC LIMIT ?"
            ))?;
            let session_rows = stmt
                .query_map(params![self.config.session_id, now, session_limit], map)?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows.extend(session_rows);
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Batch recall bump, capped per row at `WM_BUMP_CAP_HOURS` so a single read cannot extend
        // the effective clock indefinitely (`beam.py` L3576-L3605).
        let now_dt = chrono::Utc::now();
        let cap = chrono::Duration::seconds((self.config.wm_bump_cap_hours * 3600.0) as i64);
        let mut updates: HashMap<String, Vec<String>> = HashMap::new();
        for (row, last_recalled) in &rows {
            let new_last = match last_recalled.as_deref().and_then(parse_ts) {
                Some(parsed) => (parsed + cap).min(now_dt),
                None => now_dt,
            };
            updates
                .entry(new_last.to_rfc3339())
                .or_default()
                .push(row.id.clone());
        }
        for (ts, ids) in updates {
            let ph = vec!["?"; ids.len()].join(",");
            let mut bind: Vec<Value> = vec![Value::Text(ts)];
            bind.extend(ids.into_iter().map(Value::Text));
            conn.execute(
                &format!(
                    "UPDATE working_memory SET recall_count = recall_count + 1, \
                     last_recalled = ? WHERE id IN ({ph})"
                ),
                params_from_iter(bind),
            )?;
        }
        Ok(rows.into_iter().map(|(row, _)| row).collect())
    }

    /// Linear-hybrid recall over both tiers (`beam.py` `recall` L5027), keyword-only. Equivalent
    /// to [`Engine::recall_with_vector`] with no query vector.
    pub fn recall(&self, query: &str, top_k: usize) -> Result<Vec<MemoryRow>> {
        self.recall_with_vector(query, top_k, None)
    }

    /// Hybrid lexical + FTS5 + vector recall across the working **and** episodic tiers with the
    /// engine's configured scope and no filters (`beam.py` `recall` L5027 defaults).
    pub fn recall_with_vector(
        &self,
        query: &str,
        top_k: usize,
        query_vector: Option<&[f32]>,
    ) -> Result<Vec<MemoryRow>> {
        let scope = self.config_scope();
        self.recall_with_scope(&RecallReq {
            query,
            top_k,
            query_vector,
            scope: &scope,
            filters: RecallFilters::default(),
        })
    }

    /// The recall scope derived from the engine config (`beam.py` instance `author_id`/`channel_id`).
    /// The provider seam recalls with this; the `mnemosyne_recall` tool may override it per call.
    pub fn config_scope(&self) -> RecallScope {
        RecallScope {
            author_id: self.config.author_id.clone(),
            author_type: self.config.author_type.clone(),
            channel_id: self.config.channel_id.clone(),
        }
    }

    /// As [`Engine::recall_with_vector`], but with an explicit [`RecallReq`] carrying the
    /// multi-agent [`RecallScope`] and the per-call [`RecallFilters`].
    pub fn recall_with_scope(&self, req: &RecallReq) -> Result<Vec<MemoryRow>> {
        let rows = match self.config.recall_mode {
            RecallMode::Base => {
                // Per-call weight overrides (`beam.py` `recall(vec_weight=..., ...)` kwargs);
                // unset components fall back to the configured defaults.
                let (dv, df, di) = self.config.recall_weights;
                let weights = (
                    req.filters.vec_weight.unwrap_or(dv),
                    req.filters.fts_weight.unwrap_or(df),
                    req.filters.importance_weight.unwrap_or(di),
                );
                self.recall_base(req, weights)
            }
            RecallMode::Enhanced => self.recall_enhanced(req),
            RecallMode::Polyphonic => self.recall_polyphonic(req),
        }?;
        if let Some(pm) = self.plugins_if_active() {
            for row in &rows {
                pm.notify_recall(&serde_json::json!({"id": row.id, "content": row.content}));
            }
        }
        Ok(rows)
    }

    /// The recall scope *branch* predicate + params (`beam.py` L5182-L5192): channel-widened,
    /// author-only (unrestricted), or the default `session OR global`.
    fn scope_branch(&self, scope: &RecallScope) -> (&'static str, Vec<Value>) {
        if let Some(channel) = &scope.channel_id {
            (
                "(session_id = ? OR scope = 'global' OR channel_id = ?)",
                vec![
                    Value::Text(self.config.session_id.clone()),
                    Value::Text(channel.clone()),
                ],
            )
        } else if scope.author_id.is_some() || scope.author_type.is_some() {
            ("(1=1)", Vec::new())
        } else {
            (
                "(session_id = ? OR scope = 'global')",
                vec![Value::Text(self.config.session_id.clone())],
            )
        }
    }

    /// The full per-tier recall WHERE body: validity + scope branch + the per-call row filters +
    /// exact identity filters, in Python's clause order (`beam.py` L5176-L5225).
    fn tier_where(&self, scope: &RecallScope, f: &RecallFilters) -> (String, Vec<Value>) {
        let mut clauses: Vec<String> = vec![
            "(valid_until IS NULL OR valid_until > ?)".to_string(),
            "superseded_by IS NULL".to_string(),
        ];
        let mut p: Vec<Value> = vec![Value::Text(util::now_iso())];
        let (branch, branch_params) = self.scope_branch(scope);
        clauses.push(branch.to_string());
        p.extend(branch_params);
        if let Some(d) = &f.from_date {
            clauses.push("timestamp >= ?".to_string());
            p.push(Value::Text(format!("{d}T00:00:00")));
        }
        if let Some(d) = &f.to_date {
            clauses.push("timestamp <= ?".to_string());
            p.push(Value::Text(format!("{d}T23:59:59")));
        }
        if let Some(s) = &f.source {
            clauses.push("source = ?".to_string());
            p.push(Value::Text(s.clone()));
        }
        if let Some(t) = &f.topic {
            clauses.push("source = ?".to_string());
            p.push(Value::Text(t.clone()));
        }
        if let Some(v) = &f.veracity {
            clauses.push("veracity = ?".to_string());
            p.push(Value::Text(v.clone()));
        }
        if let Some(m) = &f.memory_type {
            clauses.push("memory_type = ?".to_string());
            p.push(Value::Text(m.clone()));
        }
        if let Some(a) = &scope.author_id {
            clauses.push("author_id = ?".to_string());
            p.push(Value::Text(a.clone()));
        }
        if let Some(a) = &scope.author_type {
            clauses.push("author_type = ?".to_string());
            p.push(Value::Text(a.clone()));
        }
        if let Some(c) = &scope.channel_id {
            clauses.push("channel_id = ?".to_string());
            p.push(Value::Text(c.clone()));
        }
        (clauses.join(" AND "), p)
    }

    /// The base linear recall (`beam.py` `recall` L5027-L6210) with explicit raw
    /// `(vec, fts, importance)` weights (normalized here, `beam.py` L5115).
    fn recall_base(&self, req: &RecallReq, weights: (f64, f64, f64)) -> Result<Vec<MemoryRow>> {
        let query = req.query;
        let top_k = req.top_k;
        let query_lower = query.to_lowercase();
        let query_words = recall_tokens(&query_lower);
        let (vw, fw, iw) = scoring::normalize_weights(weights.0, weights.1, weights.2);
        let halflife = self.config.recency_halflife_hours;

        // ---- Temporal scoring setup (`beam.py` L5137-L5141) ----
        let temporal_weight = req.filters.temporal_weight;
        let parsed_query_time = parse_query_time(req.filters.query_time.as_deref());
        let th_halflife = req
            .filters
            .temporal_halflife
            .unwrap_or(self.config.temporal_halflife_hours);
        let t_boost = |ts: &str, score: f64| -> f64 {
            if temporal_weight > 0.0 {
                score * (1.0 + temporal_weight * temporal_boost(ts, parsed_query_time, th_halflife))
            } else {
                score
            }
        };

        let mut diag = DiagCounts::default();
        let mut results: Vec<MemoryRow> = Vec::new();
        let conn = self.store.conn.lock().unwrap();

        // ---- Working memory: FTS5 fast path (`beam.py` L5169-L5173) ----
        let wm_fts = fts_search_working(&conn, query, (top_k * 3).max(50));
        let wm_ranks: HashMap<String, f64> = wm_fts.iter().cloned().collect();
        let mut wm_ids: Vec<String> = wm_fts.iter().map(|(id, _)| id.clone()).collect();
        let mut wm_id_set: HashSet<String> = wm_ids.iter().cloned().collect();

        let (wm_where, wm_params) = self.tier_where(req.scope, &req.filters);

        // ---- Working memory: vector search (`beam.py` L5227-L5242) ----
        let mut wm_vec_sims: HashMap<String, f64> = HashMap::new();
        if let Some(qv) = req.query_vector {
            for (id, sim) in wm_vec_search(&conn, qv, (top_k * 3).max(50), &wm_where, &wm_params)? {
                wm_vec_sims.insert(id.clone(), sim);
                if wm_id_set.insert(id.clone()) {
                    wm_ids.push(id);
                }
            }
        }

        diag.wm_fallback_used = wm_ids.is_empty();
        if diag.wm_fallback_used {
            self.recall_diag.record_fallback_used(true, false);
        }

        let rows: Vec<Cand> = if !wm_ids.is_empty() {
            let ph = vec!["?"; wm_ids.len()].join(",");
            let mut bind: Vec<Value> = wm_ids.iter().cloned().map(Value::Text).collect();
            bind.extend(wm_params.iter().cloned());
            let mut stmt = conn.prepare(&format!(
                "SELECT {CAND_COLS} FROM working_memory WHERE id IN ({ph}) AND {wm_where}"
            ))?;
            let rows = stmt
                .query_map(params_from_iter(bind), Cand::from_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows
        } else {
            // Fallback: recent rows, scored lexically (`beam.py` L5257-L5268).
            let limit = self.config.episodic_recall_limit.min(2000);
            let mut bind = wm_params.clone();
            bind.push(Value::Integer(limit as i64));
            let mut stmt = conn.prepare(&format!(
                "SELECT {CAND_COLS} FROM working_memory WHERE {wm_where} \
                 ORDER BY timestamp DESC LIMIT ?"
            ))?;
            let rows = stmt
                .query_map(params_from_iter(bind), Cand::from_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows
        };

        // Rank normalization bounds (`beam.py` L5271-L5277).
        let (wm_min_rank, wm_rng) = rank_bounds(wm_ranks.values());

        let min_relevance = scoring::lexical_floor(query_words.len());
        let single_token_relevance = 1.0 / query_words.len().max(1) as f64;
        // Broad multi-hit queries admit one-token-per-row hits (`beam.py` L5285-L5299).
        let broad_multi_hit_query = if query_words.len() >= 4 {
            let query_word_set: HashSet<&String> = query_words.iter().collect();
            let mut matched: HashSet<String> = HashSet::new();
            for row in &rows {
                for token in recall_tokens(&row.content.to_lowercase()) {
                    if query_word_set.contains(&token) {
                        matched.insert(token);
                    }
                }
            }
            matched.len() >= 2
        } else {
            false
        };

        // ---- Working-memory scoring loop (`beam.py` L5299-L5366) ----
        for row in &rows {
            let lexical = lexical_relevance(&query_words, &row.content, &query_lower);
            let row_min_relevance = if broad_multi_hit_query {
                single_token_relevance
            } else {
                min_relevance
            };
            let relevance = if let Some(rank) = wm_ranks.get(&row.id) {
                let normalized = 1.0 - ((rank - wm_min_rank) / wm_rng);
                if lexical >= row_min_relevance {
                    lexical.max(0.75 * lexical + 0.25 * normalized)
                } else {
                    0.0
                }
            } else {
                lexical
            };
            if relevance >= row_min_relevance
                || (!wm_ranks.is_empty() && query_words.len() <= 1 && relevance > 0.0)
            {
                let decay = scoring::recency_decay_hl(age_hours(&row.timestamp), halflife);
                let vec_sim = wm_vec_sims.get(&row.id).copied().unwrap_or(0.0);
                let score =
                    scoring::working_memory_score(relevance, row.importance, iw, vec_sim, decay);
                let score = t_boost(&row.timestamp, score);
                if diag.wm_fallback_used {
                    diag.wm_fallback += 1;
                } else if wm_ranks.contains_key(&row.id) {
                    diag.wm_fts += 1;
                } else if wm_vec_sims.contains_key(&row.id) {
                    diag.wm_vec += 1;
                }
                let mut out = row.to_row(Tier::Working);
                out.score = round4(score);
                out.keyword_score = round4(relevance);
                out.dense_score = round4(vec_sim);
                // Parity quirk: Python reports `relevance` (not the FTS signal) whenever ANY FTS
                // hit exists (`beam.py` L5352).
                out.fts_score = if wm_ranks.is_empty() {
                    0.0
                } else {
                    round4(relevance)
                };
                out.recency_decay = round4(decay);
                results.push(out);
            }
        }
        if diag.wm_fallback_used && !rows.is_empty() && diag.wm_fallback == 0 {
            // Diagnostics contract: fallback total_hits records the scanned candidates even when
            // the relevance gate abstained (`beam.py` L5368-L5372).
            diag.wm_fallback = rows.len();
        }

        // ---- Entity-aware recall (`beam.py` L5373-L5494) ----
        let entity_memory_ids = find_memories_by_entity(&conn, query);
        if !entity_memory_ids.is_empty() {
            self.boost_or_add_matches(
                &conn,
                &mut results,
                &BoostArgs {
                    ids: &entity_memory_ids,
                    wm_where: &wm_where,
                    wm_params: &wm_params,
                    scope: req.scope,
                    multiplier: 1.3,
                    add_base: 0.6,
                    entity: true,
                    wm_vec_sims: &wm_vec_sims,
                    temporal: &t_boost,
                    halflife,
                },
            )?;
        }

        // ---- Fact-aware recall (`beam.py` L5496-L5616) ----
        let fact_memory_ids =
            find_memories_by_fact(&conn, &query_lower, self.config.lenient_fact_match);
        if !fact_memory_ids.is_empty() {
            self.boost_or_add_matches(
                &conn,
                &mut results,
                &BoostArgs {
                    ids: &fact_memory_ids,
                    wm_where: &wm_where,
                    wm_params: &wm_params,
                    scope: req.scope,
                    multiplier: 1.2,
                    add_base: 0.5,
                    entity: false,
                    wm_vec_sims: &wm_vec_sims,
                    temporal: &t_boost,
                    halflife,
                },
            )?;
        }

        // ---- Pre-compute the query binary vector (`beam.py` L5613-L5622) ----
        let query_bv = req
            .query_vector
            .map(crate::binary_vectors::maximally_informative_binarization);

        // ---- Episodic memory: vec + FTS5 hybrid (`beam.py` L5624-L5652) ----
        let mut vec_results: HashMap<i64, f64> = HashMap::new();
        if let Some(qv) = req.query_vector {
            let vec_rows = em_vec_search(&conn, qv, (top_k * 3).max(20))?;
            let max_distance = vec_rows.iter().map(|(_, d)| *d).fold(0.0f64, f64::max);
            for (rowid, distance) in vec_rows {
                let sim = if max_distance > 0.0 {
                    (1.0 - (distance / max_distance)).max(0.0)
                } else {
                    1.0
                };
                vec_results.insert(rowid, sim);
            }
        }
        let em_fts = fts_search_episodic(&conn, query, (top_k * 3).max(20));
        let mut fts_results: HashMap<i64, f64> = HashMap::new();
        if !em_fts.is_empty() {
            let (min_rank, rng) = rank_bounds(em_fts.iter().map(|(_, r)| r));
            for (rowid, rank) in &em_fts {
                fts_results.insert(*rowid, 1.0 - ((rank - min_rank) / rng));
            }
        }
        let episodic_rowids: HashSet<i64> = vec_results
            .keys()
            .chain(fts_results.keys())
            .copied()
            .collect();

        let (em_where, em_params) = self.tier_where(req.scope, &req.filters);

        if !episodic_rowids.is_empty() {
            let ph = vec!["?"; episodic_rowids.len()].join(",");
            let mut bind: Vec<Value> = episodic_rowids.iter().map(|r| Value::Integer(*r)).collect();
            bind.extend(em_params.iter().cloned());
            let mut stmt = conn.prepare(&format!(
                "SELECT {CAND_COLS}, rowid, binary_vector FROM episodic_memory \
                 WHERE rowid IN ({ph}) AND {em_where}"
            ))?;
            let em_rows = stmt
                .query_map(params_from_iter(bind), Cand::from_row_episodic)?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            // ---- Episodic scoring loop (`beam.py` L5713-L5837) ----
            for row in em_rows {
                let sim = vec_results.get(&row.rowid).copied().unwrap_or(0.0);
                let fts = fts_results.get(&row.rowid).copied().unwrap_or(0.0);
                let decay = scoring::recency_decay_hl(age_hours(&row.timestamp), halflife);
                let lexical = lexical_relevance(&query_words, &row.content, &query_lower);
                // Lexical gate: FTS rank alone doesn't admit a row for broad queries; strong
                // vector hits pass (`beam.py` L5739-L5744).
                if lexical < min_relevance && sim < VEC_SIM_FLOOR {
                    continue;
                }
                let graph_bonus = if self.config.graph_bonus {
                    scoring::graph_bonus(graph_edge_count_like(&conn, &row.id)?)
                } else {
                    0.0
                };
                let fact_bonus = if self.config.fact_bonus {
                    scoring::fact_bonus(fact_match_count(&conn, &row.id, &query_lower)?)
                } else {
                    0.0
                };
                let binary_bonus = match (&query_bv, &row.binary_vector) {
                    (Some(qb), Some(bv)) if self.config.binary_bonus => binary_bonus(qb, bv),
                    _ => 0.0,
                };
                let score = scoring::episodic_score(
                    sim,
                    fts,
                    row.importance,
                    lexical,
                    decay,
                    (vw, fw, iw),
                    graph_bonus,
                    fact_bonus,
                    binary_bonus,
                );
                let score = t_boost(&row.timestamp, score);
                if fts_results.contains_key(&row.rowid) {
                    diag.em_fts += 1;
                } else if vec_results.contains_key(&row.rowid) {
                    diag.em_vec += 1;
                }
                let mut out = row.to_row(Tier::Episodic);
                // The main episodic SELECT carries no veracity; the tier-lookup pass below
                // overwrites it (`beam.py` L5824 sets the "unknown" placeholder).
                out.veracity = "unknown".to_string();
                out.score = round4(score);
                out.keyword_score = round4(lexical);
                out.dense_score = round4(sim);
                out.fts_score = round4(fts);
                out.recency_decay = round4(decay);
                results.push(out);
            }
        } else {
            // ---- Episodic fallback: recent scan (`beam.py` L5839-L5931) ----
            diag.em_fallback_used = true;
            self.recall_diag.record_fallback_used(false, true);
            let limit = self.config.episodic_recall_limit.min(500);
            let mut bind = em_params.clone();
            bind.push(Value::Integer(limit as i64));
            let mut stmt = conn.prepare(&format!(
                "SELECT {CAND_COLS}, rowid, binary_vector FROM episodic_memory \
                 WHERE {em_where} ORDER BY timestamp DESC LIMIT ?"
            ))?;
            let em_rows = stmt
                .query_map(params_from_iter(bind), Cand::from_row_episodic)?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for row in em_rows {
                let relevance = lexical_relevance(&query_words, &row.content, &query_lower);
                if relevance < min_relevance {
                    continue;
                }
                let decay = scoring::recency_decay_hl(age_hours(&row.timestamp), halflife);
                let mut score =
                    scoring::working_memory_score(relevance, row.importance, iw, 0.0, decay);
                let graph_b = if self.config.graph_bonus {
                    scoring::graph_bonus(graph_edge_count_like(&conn, &row.id)?)
                } else {
                    0.0
                };
                let fact_b = if self.config.fact_bonus {
                    scoring::fact_bonus(fact_match_count(&conn, &row.id, &query_lower)?)
                } else {
                    0.0
                };
                // The binary bonus stays disabled on this path (`beam.py` L5895).
                score += graph_b + fact_b;
                let score = t_boost(&row.timestamp, score);
                diag.em_fallback += 1;
                let mut out = row.to_row(Tier::Episodic);
                out.score = round4(score);
                out.keyword_score = round4(relevance);
                out.recency_decay = round4(decay);
                results.push(out);
            }
        }

        // ---- Tiered degradation + veracity multipliers (`beam.py` L5933-L5983) ----
        let em_ids_for_tier: Vec<String> = results
            .iter()
            .filter(|r| r.tier == Tier::Episodic)
            .map(|r| r.id.clone())
            .collect();
        let mut ep_summary_of: HashMap<String, String> = HashMap::new();
        if !em_ids_for_tier.is_empty() {
            let ph = vec!["?"; em_ids_for_tier.len()].join(",");
            let mut stmt = conn.prepare(&format!(
                "SELECT id, tier, veracity, summary_of FROM episodic_memory WHERE id IN ({ph})"
            ))?;
            let bind: Vec<Value> = em_ids_for_tier.iter().cloned().map(Value::Text).collect();
            let lookup: Vec<(String, i64, String, String)> = stmt
                .query_map(params_from_iter(bind), |r| {
                    Ok((
                        r.get(0)?,
                        r.get::<_, Option<i64>>(1)?.unwrap_or(1),
                        r.get::<_, Option<String>>(2)?
                            .unwrap_or_else(|| "unknown".to_string()),
                        r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let tier_lookup: HashMap<&str, i64> = lookup
                .iter()
                .map(|(id, t, _, _)| (id.as_str(), *t))
                .collect();
            let veracity_lookup: HashMap<&str, &str> = lookup
                .iter()
                .map(|(id, _, v, _)| (id.as_str(), v.as_str()))
                .collect();
            for (id, _, _, summary_of) in &lookup {
                ep_summary_of.insert(id.clone(), summary_of.clone());
            }
            let [w1, w2, w3] = self.config.tier_weights;
            for r in results.iter_mut().filter(|r| r.tier == Tier::Episodic) {
                let ep_tier = tier_lookup.get(r.id.as_str()).copied().unwrap_or(1);
                r.tier_level = ep_tier;
                r.veracity = veracity_lookup
                    .get(r.id.as_str())
                    .copied()
                    .unwrap_or("unknown")
                    .to_string();
                // Python's `weight_map.get(ep_tier, 1.0)`: unknown tiers weigh 1.0 (no clamp).
                r.score *= match ep_tier {
                    1 => w1,
                    2 => w2,
                    3 => w3,
                    _ => 1.0,
                };
                if self.config.veracity_multiplier {
                    r.score *= self.config.veracity_weights.weight(&r.veracity);
                }
            }
        }
        // E4: the veracity multiplier applies to working rows too (`beam.py` L5975-L5983).
        if self.config.veracity_multiplier {
            for r in results.iter_mut().filter(|r| r.tier == Tier::Working) {
                r.score *= self.config.veracity_weights.weight(&r.veracity);
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // E3.a.3: collapse (episodic summary, working sources) duplicates (`beam.py` L5997-L6004).
        if self.config.cross_tier_dedup {
            results = dedup_cross_tier_summary_links(results, &ep_summary_of);
        }

        // ---- MEMORIA structured-fact supplement (`beam.py` L6006-L6059) ----
        self.supplement_with_memoria(&conn, query, &query_words, &query_lower, &mut results)?;

        // ---- Multi-aspect greedy selection (`beam.py` L6061-L6081) ----
        if query_words.len() >= 4 && results.len() > top_k {
            results = greedy_aspect_select(results, &query_words, top_k);
        }
        let mut final_results: Vec<MemoryRow> = results;
        final_results.truncate(top_k);

        // ---- Recall tracking (`beam.py` L6084-L6119) ----
        self.bump_recall_scoped(&conn, &final_results, req.scope)?;

        // ---- C4 diagnostics records (`beam.py` L6121-L6150) ----
        self.recall_diag.record_tier_hits("wm_fts", diag.wm_fts);
        self.recall_diag.record_tier_hits("wm_vec", diag.wm_vec);
        self.recall_diag
            .record_tier_hits("wm_fallback", diag.wm_fallback);
        self.recall_diag.record_tier_hits("em_fts", diag.em_fts);
        self.recall_diag.record_tier_hits("em_vec", diag.em_vec);
        self.recall_diag
            .record_tier_hits("em_fallback", diag.em_fallback);
        let total_kept = diag.wm_fts
            + diag.wm_vec
            + diag.wm_fallback
            + diag.em_fts
            + diag.em_vec
            + diag.em_fallback;
        self.recall_diag
            .record_call(final_results.is_empty() && total_kept == 0);

        // ---- Optional fact-recall integration (`beam.py` L6152-L6178) ----
        if self.config.fact_recall_enabled {
            let existing: HashSet<String> =
                final_results.iter().map(|r| r.content.clone()).collect();
            for hit in self.fact_recall_conn(&conn, query, top_k.max(10))? {
                if existing.contains(&hit.content) {
                    continue;
                }
                final_results.push(MemoryRow {
                    id: format!("cf_{}", hit.fact_id),
                    content: hit.content,
                    score: hit.score * 0.9, // slight discount vs direct memory
                    source: "fact_recall".to_string(),
                    tier: Tier::Fact,
                    ..Default::default()
                });
            }
            final_results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            final_results.truncate(top_k);
        }

        Ok(final_results)
    }

    /// The shared entity-aware / fact-aware boost-or-add pass (`beam.py` L5375-L5616): matched ids
    /// already in `results` get their score multiplied (capped at 1.0); new working then episodic
    /// rows enter with the floor score `(add_base + imp*0.2) * (0.7 + 0.3*decay)`.
    fn boost_or_add_matches(
        &self,
        conn: &Connection,
        results: &mut Vec<MemoryRow>,
        args: &BoostArgs<'_>,
    ) -> Result<()> {
        let mark = |r: &mut MemoryRow, entity: bool| {
            if entity {
                r.entity_match = true;
            } else {
                r.fact_match = true;
            }
        };

        // Working tier: full recall filters apply (`beam.py` L5378-L5385).
        let ph = vec!["?"; args.ids.len()].join(",");
        let mut bind: Vec<Value> = args.ids.iter().cloned().map(Value::Text).collect();
        bind.extend(args.wm_params.iter().cloned());
        let mut stmt = conn.prepare(&format!(
            "SELECT {CAND_COLS} FROM working_memory WHERE id IN ({ph}) AND {w}",
            w = args.wm_where
        ))?;
        let wm_rows = stmt
            .query_map(params_from_iter(bind), Cand::from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let existing: HashSet<String> = results.iter().map(|r| r.id.clone()).collect();
        for row in wm_rows {
            if existing.contains(&row.id) {
                if let Some(r) = results.iter_mut().find(|r| r.id == row.id) {
                    r.score = round4((r.score * args.multiplier).min(1.0));
                    mark(r, args.entity);
                }
            } else {
                let decay = scoring::recency_decay_hl(age_hours(&row.timestamp), args.halflife);
                let score = (args.add_base + row.importance * 0.2) * (0.7 + 0.3 * decay);
                let score = (args.temporal)(&row.timestamp, score);
                let mut out = row.to_row(Tier::Working);
                out.score = round4(score);
                out.dense_score = round4(args.wm_vec_sims.get(&row.id).copied().unwrap_or(0.0));
                out.recency_decay = round4(decay);
                mark(&mut out, args.entity);
                results.push(out);
            }
        }

        // Episodic tier: scope branch + validity only — no date/source filters
        // (`beam.py` L5430-L5449 parity quirk).
        let (branch, branch_params) = self.scope_branch(args.scope);
        let mut bind: Vec<Value> = args.ids.iter().cloned().map(Value::Text).collect();
        bind.extend(branch_params);
        bind.push(Value::Text(util::now_iso()));
        let mut stmt = conn.prepare(&format!(
            "SELECT {CAND_COLS} FROM episodic_memory WHERE id IN ({ph}) AND {branch} \
             AND (valid_until IS NULL OR valid_until > ?) AND superseded_by IS NULL"
        ))?;
        let em_rows = stmt
            .query_map(params_from_iter(bind), Cand::from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let existing: HashSet<String> = results.iter().map(|r| r.id.clone()).collect();
        for row in em_rows {
            if existing.contains(&row.id) {
                if let Some(r) = results.iter_mut().find(|r| r.id == row.id) {
                    r.score = round4((r.score * args.multiplier).min(1.0));
                    mark(r, args.entity);
                }
            } else {
                let decay = scoring::recency_decay_hl(age_hours(&row.timestamp), args.halflife);
                let score = (args.add_base + row.importance * 0.2) * (0.7 + 0.3 * decay);
                let score = (args.temporal)(&row.timestamp, score);
                let mut out = row.to_row(Tier::Episodic);
                out.score = round4(score);
                // C30: episodic rows never key into the WM vec map (`beam.py` L5470-L5478).
                out.dense_score = 0.0;
                out.recency_decay = round4(decay);
                mark(&mut out, args.entity);
                results.push(out);
            }
        }
        Ok(())
    }

    /// Fold a high-relevance MEMORIA structured-fact hit into the candidate set
    /// (`beam.py` L6006-L6059): the hit enters as a `memoria` row scored `min(0.6, rel*0.6)` plus
    /// its originating working rows as `memoria_source` rows scored `min(0.59, 0.2 + rel*0.8)`,
    /// then candidates re-sort. Best-effort: failures are swallowed.
    fn supplement_with_memoria(
        &self,
        conn: &Connection,
        query: &str,
        query_words: &[String],
        query_lower: &str,
        scored: &mut Vec<MemoryRow>,
    ) -> Result<()> {
        let result = match memoria::memoria_retrieve(conn, &self.config.session_id, query, 3) {
            Some(r) if r.source != "fallback" && !r.context.is_empty() => r,
            _ => return Ok(()),
        };
        let rel = lexical_relevance(query_words, &result.context, query_lower);
        if rel < 0.35 {
            return Ok(());
        }
        scored.push(MemoryRow {
            id: format!("memoria_{}", result.source),
            content: format!("[MEMORIA {}]\n{}", result.source, result.context),
            source: format!("memoria_{}", result.source),
            score: round4((rel * 0.6).min(0.6)),
            keyword_score: round4(rel),
            importance: 0.5,
            tier: Tier::Memoria,
            tier_level: 1,
            ..Default::default()
        });

        let source_ids: Vec<&String> = result
            .source_memory_ids
            .iter()
            .filter(|s| !s.is_empty())
            .collect();
        if !source_ids.is_empty() {
            let ph = vec!["?"; source_ids.len()].join(",");
            let bind: Vec<Value> = source_ids
                .iter()
                .map(|s| Value::Text((*s).clone()))
                .collect();
            let mut stmt = conn.prepare(&format!(
                "SELECT id, content, source, timestamp, importance, scope, veracity \
                 FROM working_memory WHERE id IN ({ph})"
            ))?;
            let rows = stmt
                .query_map(params_from_iter(bind), |r| {
                    Ok(MemoryRow {
                        id: format!("memoria_source_{}", r.get::<_, String>(0)?),
                        content: truncate_chars(&r.get::<_, String>(1)?, 500),
                        source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        importance: r.get::<_, Option<f64>>(4)?.unwrap_or(0.5),
                        scope: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                        veracity: r
                            .get::<_, Option<String>>(6)?
                            .unwrap_or_else(|| "unknown".to_string()),
                        score: round4((0.2 + rel * 0.8).min(0.59)),
                        keyword_score: round4(rel),
                        tier: Tier::MemoriaSource,
                        tier_level: 1,
                        ..Default::default()
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            scored.extend(rows);
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// Batch recall-count bump for the returned working/episodic rows, scoped by the recall's
    /// session/channel branch (`beam.py` L6084-L6119). Memoria/fact supplement rows are skipped.
    fn bump_recall_scoped(
        &self,
        conn: &Connection,
        rows: &[MemoryRow],
        scope: &RecallScope,
    ) -> Result<()> {
        let now = util::now_iso();
        let (branch, branch_params) = self.scope_branch(scope);
        for (tier, table) in [
            (Tier::Working, "working_memory"),
            (Tier::Episodic, "episodic_memory"),
        ] {
            let ids: Vec<&MemoryRow> = rows.iter().filter(|r| r.tier == tier).collect();
            if ids.is_empty() {
                continue;
            }
            let ph = vec!["?"; ids.len()].join(",");
            let mut bind: Vec<Value> = vec![Value::Text(now.clone())];
            bind.extend(ids.iter().map(|r| Value::Text(r.id.clone())));
            bind.extend(branch_params.iter().cloned());
            conn.execute(
                &format!(
                    "UPDATE {table} SET recall_count = recall_count + 1, last_recalled = ? \
                     WHERE id IN ({ph}) AND {branch}"
                ),
                params_from_iter(bind),
            )?;
        }
        Ok(())
    }

    /// Search the raw `facts` table (FTS5 + LIKE fallback) and `consolidated_facts` for structured
    /// knowledge (`beam.py` `fact_recall` L6874-L6990). Rank-position relevance × confidence.
    pub fn fact_recall(&self, query: &str, top_k: usize) -> Result<Vec<FactHit>> {
        let conn = self.store.conn.lock().unwrap();
        self.fact_recall_conn(&conn, query, top_k)
    }

    fn fact_recall_conn(
        &self,
        conn: &Connection,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<FactHit>> {
        let query_lower = query.to_lowercase();
        let mut results: Vec<FactHit> = Vec::new();

        // --- Source 1: raw facts table (FTS5, then a LIKE fallback) ---
        // Python passes the raw query as the MATCH expression; malformed queries fall through to
        // the LIKE path (`beam.py` L6889-L6894).
        let mut fts_rowids: Vec<i64> = {
            let fetch = || -> rusqlite::Result<Vec<i64>> {
                let mut stmt = conn.prepare(
                    "SELECT rowid FROM fts_facts WHERE fts_facts MATCH ?1 \
                     ORDER BY rank, rowid LIMIT ?2",
                )?;
                let rows = stmt
                    .query_map(params![query, (top_k * 3) as i64], |r| r.get(0))?
                    .collect::<std::result::Result<Vec<i64>, _>>()?;
                Ok(rows)
            };
            fetch().unwrap_or_default()
        };
        if fts_rowids.is_empty() {
            let mut seen: HashSet<i64> = HashSet::new();
            for word in query_lower.split_whitespace().take(6) {
                if word.chars().count() < 3 {
                    continue;
                }
                let like = format!("%{word}%");
                let mut stmt = conn.prepare(
                    "SELECT rowid FROM facts WHERE subject LIKE ?1 OR predicate LIKE ?1 \
                     OR object LIKE ?1 LIMIT ?2",
                )?;
                let rows: Vec<i64> = stmt
                    .query_map(params![like, top_k as i64], |r| r.get(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                for rowid in rows {
                    if seen.insert(rowid) {
                        fts_rowids.push(rowid);
                    }
                }
            }
        }
        if !fts_rowids.is_empty() {
            let ranked: Vec<i64> = fts_rowids.iter().take(top_k).copied().collect();
            let rank_pos: HashMap<i64, usize> =
                ranked.iter().enumerate().map(|(i, r)| (*r, i)).collect();
            let ph = vec!["?"; ranked.len()].join(",");
            let bind: Vec<Value> = ranked.iter().map(|r| Value::Integer(*r)).collect();
            let mut stmt = conn.prepare(&format!(
                "SELECT rowid, fact_id, subject, predicate, object, confidence \
                 FROM facts WHERE rowid IN ({ph})"
            ))?;
            let n = ranked.len().max(1);
            let rows = stmt
                .query_map(params_from_iter(bind), |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                        r.get::<_, Option<f64>>(5)?.unwrap_or(0.5),
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for (rowid, fact_id, subject, predicate, object, confidence) in rows {
                let fact_text = [subject.as_str(), predicate.as_str(), object.as_str()]
                    .iter()
                    .filter(|p| !p.is_empty())
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" ");
                let fact_text = if fact_text.trim().is_empty() {
                    object.clone()
                } else {
                    fact_text.trim().to_string()
                };
                // Relevance from the FTS rank position (top hit ~1.0, decaying), combined with
                // confidence (`beam.py` L6928-L6949).
                let pos = rank_pos.get(&rowid).copied().unwrap_or(n - 1);
                let relevance = 1.0 - (pos as f64 / n as f64);
                results.push(FactHit {
                    content: fact_text,
                    score: relevance * confidence,
                    fact_id,
                    subject,
                    predicate,
                });
            }
        }

        // --- Source 2: consolidated_facts (sleep-consolidated LLM triples, `beam.py` L6952+) ---
        let mut seen_subjects: HashSet<String> = HashSet::new();
        for word in query_lower.split_whitespace().take(6) {
            if word.chars().count() < 3 {
                continue;
            }
            let subj = capitalize(word);
            if !seen_subjects.insert(subj.clone()) {
                continue;
            }
            let mut stmt = conn.prepare(
                "SELECT id, subject, predicate, object, confidence FROM consolidated_facts \
                 WHERE subject = ?1 AND confidence >= 0.3 AND superseded_by IS NULL \
                 ORDER BY confidence DESC, mention_count DESC",
            )?;
            let rows = stmt
                .query_map(params![subj], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, Option<f64>>(4)?.unwrap_or(0.5),
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for (id, subject, predicate, object, confidence) in rows {
                let fact_text = format!("{subject} {predicate} {object}");
                if results.iter().any(|r| r.content == fact_text) {
                    continue;
                }
                results.push(FactHit {
                    content: fact_text,
                    score: confidence * 0.85,
                    fact_id: id,
                    subject,
                    predicate,
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        Ok(results)
    }

    /// Enhanced recall (`beam.py` `recall_enhanced` L6177-L6328): classify the query intent and bias
    /// the hybrid weights, synonym-expand the query, consult the 5-tier query cache, run base recall
    /// over the expanded query, Weibull-rescore by memory type (`score*0.7 + wb*0.3`), MMR-diversify,
    /// and cache the result. Associative graph expansion is off by default in Python, and base recall
    /// already injects entity/graph candidates, so it is not re-run here.
    fn recall_enhanced(&self, req: &RecallReq) -> Result<Vec<MemoryRow>> {
        // 1. Intent classification -> weight bias.
        let intent = query_intent::classify_intent(req.query);
        let weights = if intent == query_intent::Intent::General {
            self.config.recall_weights
        } else {
            query_intent::adjust_weights(self.config.recall_weights, intent)
        };

        // 2. Synonym expansion (broadens FTS/lexical candidate generation).
        let expanded = synonyms::expand_query(req.query);

        // 3. Query cache check (keyed on the original query).
        if let Some(mut cached) = self.query_cache().get(req.query, req.query_vector) {
            cached.truncate(req.top_k);
            return Ok(cached);
        }

        // 4. Base recall over the expanded query, gathering a wider pool.
        let mut results = self.recall_base(
            &RecallReq {
                query: &expanded,
                top_k: req.top_k * 2,
                query_vector: req.query_vector,
                scope: req.scope,
                filters: req.filters.clone(),
            },
            weights,
        )?;

        // 5. Weibull re-scoring by memory type — skipped when the caller supplied a temporal
        // boost (`beam.py` L6243-L6245).
        if req.filters.temporal_weight == 0.0 {
            self.weibull_rescore(&mut results)?;
        }

        // 6. MMR diversity rerank at the over-fetch width (`mmr_rerank(..., top_k=top_k*2)`,
        // L6281-L6282) — only drops candidates beyond `top_k*2`; the score sort below decides
        // the final order.
        if results.len() > 1 {
            let items: Vec<(String, f64)> = results
                .iter()
                .map(|r| (r.content.clone(), r.score))
                .collect();
            results = mmr::mmr_rerank(&items, mmr::DEFAULT_LAMBDA, req.top_k * 2)
                .into_iter()
                .map(|i| results[i].clone())
                .collect();
        }

        // 7. Sort by score and take the top results (L6285-L6286). Step 8 (associative graph
        // expansion) is `use_associative=False` by default in Python and not ported.
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(req.top_k);

        // 9. Cache the result for next time.
        self.query_cache()
            .put(req.query, &results, req.query_vector);
        Ok(results)
    }

    /// Blend the per-type Weibull temporal boost into each row's score (`beam.py` L6243-L6278):
    /// `score = round4(score*0.7 + weibull_boost*0.3)`. Working/episodic rows read their
    /// `memory_type` from the tier table; supplement rows (memoria/fact) fall back to `general`
    /// like Python's dict-shaped rows, so they get rescored too (a memoria row's empty timestamp
    /// yields `wb = 0.0` -> `score*0.7`).
    fn weibull_rescore(&self, rows: &mut [MemoryRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let conn = self.store.conn.lock().unwrap();
        for row in rows.iter_mut() {
            let table = match row.tier {
                Tier::Working => Some("working_memory"),
                Tier::Episodic => Some("episodic_memory"),
                Tier::Memoria | Tier::MemoriaSource | Tier::Fact => None,
            };
            let mt: Option<String> = table.and_then(|table| {
                conn.query_row(
                    &format!("SELECT memory_type FROM {table} WHERE id = ?1"),
                    params![row.id],
                    |r| r.get(0),
                )
                .ok()
            });
            let mut memory_type = mt.unwrap_or_default();
            if memory_type.is_empty() || memory_type == "unknown" {
                memory_type = "general".to_string();
            }
            let wb = weibull::weibull_boost(age_hours(&row.timestamp), &memory_type, None);
            row.score = round4(row.score * 0.7 + wb * 0.3);
        }
        Ok(())
    }

    /// Four-voice polyphonic recall (`PolyphonicRecallEngine.recall` + `beam.py`
    /// `_recall_polyphonic` L6547-L6737): gathers the vector / graph / fact / temporal voices,
    /// fuses via RRF (voice weights stay metadata-only), diversity-reranks, assembles within the
    /// context budget, then materializes rows with the linear path's isolation/validity filters,
    /// composes the veracity + tier-degradation multipliers on top of the RRF score, cross-tier
    /// dedups, bumps recall stats scoped, and prepends the MEMORIA supplement.
    fn recall_polyphonic(&self, req: &RecallReq) -> Result<Vec<MemoryRow>> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();

        // ---- The four voices (over-fetch top_k*2, `beam.py` L6599) ----
        let voices = [
            poly_vector_voice(&conn, req.query_vector, &now),
            poly_graph_voice(&conn, req.query)?,
            poly_fact_voice(&conn, req.query)?,
            poly_temporal_voice(&conn, req.query)?,
        ];
        let fused = polyphonic::combine_voices(&voices);
        let reranked = polyphonic::diversity_rerank(fused, req.top_k * 2);
        let assembled = polyphonic::assemble_context(reranked, 4000);

        // ---- Materialize with filters + multipliers (`_recall_polyphonic` L6605-L6659) ----
        let mut ep_summary_of: HashMap<String, String> = HashMap::new();
        let mut finals: Vec<MemoryRow> = Vec::new();
        for r in assembled {
            // Synthetic consolidated-fact ids can't map back to source rows; they contribute
            // ranking signal only (`beam.py` L6577-L6582, L6617).
            if r.memory_id.starts_with("cf_") {
                continue;
            }
            let Some(poly) = fetch_polyphonic_row(&conn, &r.memory_id)? else {
                continue;
            };
            if !self.polyphonic_row_passes_filters(&poly, req, &now) {
                continue;
            }
            let mut row = poly.row;
            // Compose RRF with the post-E4 veracity multiplier and the episodic tier-degradation
            // multiplier so flag=ON callers keep the linear path's rank signals (L6640-L6646).
            let mut score = r.combined_score;
            if self.config.veracity_multiplier {
                score *= self.config.veracity_weights.weight(&row.veracity);
            }
            if row.tier == Tier::Episodic {
                let [w1, w2, w3] = self.config.tier_weights;
                score *= match row.tier_level {
                    1 => w1,
                    2 => w2,
                    3 => w3,
                    _ => 1.0,
                };
                if let Some(s) = poly.summary_of {
                    ep_summary_of.insert(row.id.clone(), s);
                }
            }
            row.score = score;
            row.voice_scores = Some(r.voice_scores);
            finals.push(row);
        }

        // Re-sort post-multiplier, dedup summary<->source pairs like the linear path, truncate
        // (`beam.py` L6657-L6665).
        finals.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut finals = dedup_cross_tier_summary_links(finals, &ep_summary_of);
        finals.truncate(req.top_k);

        // Scoped recall-stat attribution from the deduped final set (`beam.py` L6666-L6714).
        self.bump_recall_scoped(&conn, &finals, req.scope)?;

        // MEMORIA structured fact supplement — prepended, no lexical gate on this path
        // (`beam.py` L6716-L6736).
        if let Some(result) =
            memoria::memoria_retrieve(&conn, &self.config.session_id, req.query, 3)
        {
            if result.source != "fallback" && !result.context.is_empty() {
                finals.insert(
                    0,
                    MemoryRow {
                        id: format!("memoria_{}", result.source),
                        content: format!("[MEMORIA {}]\n{}", result.source, result.context),
                        source: format!("memoria_{}", result.source),
                        score: 0.95,
                        tier: Tier::Memoria,
                        tier_level: 1,
                        importance: 0.9,
                        timestamp: String::new(),
                        ..Default::default()
                    },
                );
            }
        }
        Ok(finals)
    }

    /// The linear path's filter set applied to a polyphonic row post-fetch (`beam.py`
    /// `_polyphonic_row_passes_filters` L6760-L6814): always-on session/scope isolation and
    /// validity, then the caller-supplied filters.
    fn polyphonic_row_passes_filters(&self, poly: &PolyRow, req: &RecallReq, now: &str) -> bool {
        let row = &poly.row;
        let f = &req.filters;
        // Session/scope isolation: non-global rows from another session are invisible (L6779-L6785).
        if row.scope != "global"
            && poly.session_id.is_some()
            && poly.session_id.as_deref() != Some(self.config.session_id.as_str())
        {
            return false;
        }
        // Validity (L6787-L6792).
        if let Some(vu) = &row.valid_until {
            if !vu.is_empty() && vu.as_str() <= now {
                return false;
            }
        }
        if poly.superseded_by.is_some() {
            return false;
        }
        // Caller-supplied filters (L6794-L6812). Dates compare against the raw kwarg strings.
        if let Some(from) = &f.from_date {
            if row.timestamp.as_str() < from.as_str() {
                return false;
            }
        }
        if let Some(to) = &f.to_date {
            if row.timestamp.as_str() > to.as_str() {
                return false;
            }
        }
        if let Some(source) = &f.source {
            if &row.source != source {
                return false;
            }
        }
        if let Some(topic) = &f.topic {
            if !row.source.contains(topic.as_str()) {
                return false;
            }
        }
        if let Some(author) = &req.scope.author_id {
            if row.author_id.as_deref() != Some(author.as_str()) {
                return false;
            }
        }
        if let Some(author_type) = &req.scope.author_type {
            if row.author_type.as_deref() != Some(author_type.as_str()) {
                return false;
            }
        }
        if let Some(channel) = &req.scope.channel_id {
            if row.channel_id.as_deref() != Some(channel.as_str()) {
                return false;
            }
        }
        if let Some(veracity) = &f.veracity {
            if &row.veracity != veracity {
                return false;
            }
        }
        if let Some(memory_type) = &f.memory_type {
            if &poly.memory_type != memory_type {
                return false;
            }
        }
        true
    }
}

/// A materialized polyphonic candidate: the recall row plus the isolation fields the filter pass
/// needs but [`MemoryRow`] doesn't carry (`beam.py` `_polyphonic_row_to_dict` L6847-L6872).
struct PolyRow {
    row: MemoryRow,
    session_id: Option<String>,
    superseded_by: Option<String>,
    memory_type: String,
    summary_of: Option<String>,
}

/// Resolve a polyphonic memory id to its row — episodic first, then working, unscoped; the filter
/// pass enforces isolation afterwards (`beam.py` `_fetch_polyphonic_row` L6816-L6845).
fn fetch_polyphonic_row(conn: &Connection, memory_id: &str) -> Result<Option<PolyRow>> {
    let map = |tier: Tier| {
        move |r: &rusqlite::Row<'_>| -> rusqlite::Result<PolyRow> {
            Ok(PolyRow {
                row: MemoryRow {
                    id: r.get(0)?,
                    content: r.get(1)?,
                    source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    importance: r.get::<_, Option<f64>>(5)?.unwrap_or(0.5),
                    recall_count: r.get::<_, Option<i64>>(6)?.unwrap_or(0),
                    last_recalled: r.get(7)?,
                    valid_until: r.get(8)?,
                    scope: r
                        .get::<_, Option<String>>(10)?
                        .unwrap_or_else(|| "session".into()),
                    author_id: r.get(11)?,
                    author_type: r.get(12)?,
                    channel_id: r.get(13)?,
                    veracity: r
                        .get::<_, Option<String>>(14)?
                        .unwrap_or_else(|| "unknown".into()),
                    tier,
                    tier_level: if tier == Tier::Episodic {
                        r.get::<_, Option<i64>>(16)?.unwrap_or(1)
                    } else {
                        1
                    },
                    ..Default::default()
                },
                session_id: r.get(4)?,
                superseded_by: r.get(9)?,
                memory_type: r
                    .get::<_, Option<String>>(15)?
                    .unwrap_or_else(|| "unknown".into()),
                summary_of: if tier == Tier::Episodic {
                    r.get::<_, Option<String>>(17)?
                } else {
                    None
                },
            })
        }
    };
    let episodic = conn
        .query_row(
            "SELECT id, content, source, timestamp, session_id, importance, recall_count, \
                    last_recalled, valid_until, superseded_by, scope, author_id, author_type, \
                    channel_id, veracity, memory_type, tier, summary_of \
             FROM episodic_memory WHERE id = ?1",
            params![memory_id],
            map(Tier::Episodic),
        )
        .optional()?;
    if episodic.is_some() {
        return Ok(episodic);
    }
    Ok(conn
        .query_row(
            "SELECT id, content, source, timestamp, session_id, importance, recall_count, \
                    last_recalled, valid_until, superseded_by, scope, author_id, author_type, \
                    channel_id, veracity, memory_type \
             FROM working_memory WHERE id = ?1",
            params![memory_id],
            map(Tier::Working),
        )
        .optional()?)
}

/// Voice 1 — vector (`polyphonic_recall.py` `_vector_voice` L168-L493): cosine over
/// `memory_embeddings` joined to each live tier (EM then WM), normalized `(cos+1)/2`, deduped by
/// id keeping the higher-similarity occurrence, top 20. This port always takes the
/// numpy-equivalent fallback path (no sqlite-vec ANN index is populated by the Rust engine).
fn poly_vector_voice(
    conn: &Connection,
    query_vector: Option<&[f32]>,
    now: &str,
) -> Vec<polyphonic::VoiceHit> {
    let Some(q) = query_vector else {
        return Vec::new();
    };
    let norm = (q.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sqrt();
    if q.is_empty() || norm == 0.0 {
        return Vec::new();
    }
    let mut by_id: HashMap<String, polyphonic::VoiceHit> = HashMap::new();
    for (table, tier_label) in [
        ("episodic_memory", "episodic"),
        ("working_memory", "working"),
    ] {
        let sql = format!(
            "SELECT t.id, me.embedding_json FROM memory_embeddings me \
             JOIN {table} t ON me.memory_id = t.id \
             WHERE t.superseded_by IS NULL AND (t.valid_until IS NULL OR t.valid_until > ?1) \
             LIMIT 50000"
        );
        let Ok(mut stmt) = conn.prepare(&sql) else {
            continue;
        };
        let rows = stmt
            .query_map(params![now], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map(|rows| rows.flatten().collect::<Vec<_>>())
            .unwrap_or_default();
        for (memory_id, embedding_json) in rows {
            let Ok(vec) = serde_json::from_str::<Vec<f32>>(&embedding_json) else {
                continue;
            };
            // Python skips zero-norm and dimension-mismatched rows (np.dot raises, the except
            // continues); `daemon_core::cosine` would silently return 0.0 -> sim 0.5 instead.
            if vec.len() != q.len() || vec.iter().all(|v| *v == 0.0) {
                continue;
            }
            let cos = daemon_core::cosine(q, &vec) as f64;
            let sim = (cos + 1.0) / 2.0;
            let better = by_id.get(&memory_id).map(|h| sim > h.score).unwrap_or(true);
            if better {
                let mut metadata = serde_json::Map::new();
                metadata.insert("similarity".into(), sim.into());
                metadata.insert("cosine_similarity".into(), cos.into());
                metadata.insert("embedding_tier".into(), tier_label.into());
                metadata.insert("backend".into(), "memory_embeddings".into());
                by_id.insert(
                    memory_id.clone(),
                    polyphonic::VoiceHit {
                        memory_id,
                        score: sim,
                        voice: "vector",
                        metadata,
                    },
                );
            }
        }
    }
    let mut hits: Vec<polyphonic::VoiceHit> = by_id.into_values().collect();
    // Python sorts by score over dict values (insertion-ordered); a HashMap has no such order, so
    // tie-break on id for determinism.
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.memory_id.cmp(&b.memory_id))
    });
    hits.truncate(20);
    hits
}

/// Voice 2 — graph (`polyphonic_recall.py` `_graph_voice` L495-L565): capitalized-phrase entities
/// seed gists (`0.6`) and `facts`-table subject rows (`confidence*0.5`, resolved to the fact id's
/// trailing `_` segment), then a `ctx`-edge BFS from every seed adds traversal rows at
/// `0.4/depth` under the separate `graph_traversal` voice label. No intra-voice dedup of the
/// entity-seeded rows, faithful to Python.
fn poly_graph_voice(conn: &Connection, query: &str) -> Result<Vec<polyphonic::VoiceHit>> {
    let entities = poly_extract_entities(query);
    let mut hits: Vec<polyphonic::VoiceHit> = Vec::new();
    let mut seed_ids: HashSet<String> = HashSet::new();
    for entity in &entities {
        for (mid, gist_text) in episodic_graph::find_gists_by_participant(conn, entity)? {
            seed_ids.insert(mid.clone());
            let mut metadata = serde_json::Map::new();
            metadata.insert("entity".into(), entity.as_str().into());
            metadata.insert("gist".into(), gist_text.into());
            hits.push(polyphonic::VoiceHit {
                memory_id: mid,
                score: 0.6,
                voice: "graph",
                metadata,
            });
        }
        let mut stmt = conn.prepare(
            "SELECT fact_id, subject, predicate, object, confidence FROM facts \
             WHERE subject = ?1 ORDER BY confidence DESC, timestamp DESC",
        )?;
        let facts = stmt
            .query_map(params![entity], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (fact_id, subject, predicate, object, confidence) in facts {
            // `fact.id.split("_")[-1]` (L532): beam fact ids are plain hashes, so this is
            // usually the id itself — a synthetic id that won't materialize, contributing
            // ranking signal only.
            let fact_mid = fact_id.rsplit('_').next().unwrap_or(&fact_id).to_string();
            seed_ids.insert(fact_mid.clone());
            let mut metadata = serde_json::Map::new();
            metadata.insert("entity".into(), entity.as_str().into());
            metadata.insert(
                "fact".into(),
                format!("{subject} {predicate} {object}").into(),
            );
            hits.push(polyphonic::VoiceHit {
                memory_id: fact_mid,
                score: confidence * 0.5,
                voice: "graph",
                metadata,
            });
        }
    }
    // Traversal from the seeds (depth 2, ctx edges, min weight 0.3), depth-decayed.
    let mut traversed: HashSet<String> = HashSet::new();
    let mut seeds: Vec<&String> = seed_ids.iter().collect();
    seeds.sort(); // deterministic order (Python iterates a set)
    for seed in seeds {
        for rel in episodic_graph::find_related_memories(conn, seed, 2, "ctx", 0.3)? {
            if traversed.contains(&rel.memory_id) || seed_ids.contains(&rel.memory_id) {
                continue;
            }
            traversed.insert(rel.memory_id.clone());
            let mut metadata = serde_json::Map::new();
            metadata.insert("seed".into(), seed.as_str().into());
            metadata.insert("edge_type".into(), rel.edge_type.into());
            metadata.insert("depth".into(), rel.depth.into());
            metadata.insert("weight".into(), rel.weight.into());
            hits.push(polyphonic::VoiceHit {
                memory_id: rel.memory_id,
                score: 0.4 / rel.depth as f64,
                voice: "graph_traversal",
                metadata,
            });
        }
    }
    Ok(hits)
}

/// Voice 3 — fact (`polyphonic_recall.py` `_fact_voice` L567-L614): whitespace-split query words
/// (>=3 chars), each capitalized and matched against consolidated fact subjects at confidence
/// `>= 0.5`. Hit ids are the consolidated `cf_...` ids — ranking signal that never materializes.
fn poly_fact_voice(conn: &Connection, query: &str) -> Result<Vec<polyphonic::VoiceHit>> {
    let mut hits: Vec<polyphonic::VoiceHit> = Vec::new();
    for word in query.to_lowercase().split_whitespace() {
        if word.chars().count() < 3 {
            continue;
        }
        let subject = capitalize(word);
        for fact in crate::knowledge::veracity::get_consolidated_facts(conn, Some(&subject), 0.5)? {
            let mut metadata = serde_json::Map::new();
            metadata.insert("subject".into(), fact.subject.into());
            metadata.insert("predicate".into(), fact.predicate.into());
            metadata.insert("object".into(), fact.object.into());
            metadata.insert("mentions".into(), fact.mention_count.into());
            hits.push(polyphonic::VoiceHit {
                memory_id: fact.id,
                score: fact.confidence,
                voice: "fact",
                metadata,
            });
        }
    }
    Ok(hits)
}

/// Voice 4 — temporal (`polyphonic_recall.py` `_temporal_voice` L616-L685): only on temporal
/// keywords; the 20 most recent working rows of the last 7 days (unscoped, matching Python),
/// scored `exp(-age_days/7) * importance`.
fn poly_temporal_voice(conn: &Connection, query: &str) -> Result<Vec<polyphonic::VoiceHit>> {
    if !has_temporal_keyword(query) {
        return Ok(Vec::new());
    }
    // Python: `(datetime.now() - timedelta(days=7)).isoformat()` — same clock the write path
    // stamps rows with. This port stamps rows with `util::now_iso()` (UTC RFC3339), so the cutoff
    // uses the same format for a well-ordered string comparison.
    let week_ago = (chrono::Utc::now() - chrono::Duration::days(7)).to_rfc3339();
    let mut stmt = conn.prepare(
        "SELECT id, timestamp, importance FROM working_memory \
         WHERE timestamp > ?1 ORDER BY timestamp DESC LIMIT 20",
    )?;
    let rows = stmt
        .query_map(params![week_ago], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<f64>>(2)?.unwrap_or(0.5),
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut hits = Vec::new();
    for (id, ts, importance) in rows {
        let Some(hours) = age_hours(&ts) else {
            continue;
        };
        let age_days = hours / 24.0;
        let mut metadata = serde_json::Map::new();
        metadata.insert("age_days".into(), age_days.into());
        metadata.insert("importance".into(), importance.into());
        hits.push(polyphonic::VoiceHit {
            memory_id: id,
            score: (-age_days / 7.0).exp() * importance,
            voice: "temporal",
            metadata,
        });
    }
    Ok(hits)
}

/// Query entity extraction for the graph voice (`polyphonic_recall.py` `_extract_entities`
/// L687-L692): capitalized word runs, deduped (first-seen order here; Python's `list(set(...))`
/// order is unspecified).
fn poly_extract_entities(text: &str) -> Vec<String> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b").unwrap());
    let mut seen = HashSet::new();
    re.find_iter(text)
        .map(|m| m.as_str().to_string())
        .filter(|e| seen.insert(e.clone()))
        .collect()
}

/// A structured `fact_recall` hit (`beam.py` L6874: content/score/fact_id/subject/predicate).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct FactHit {
    /// The full `subject predicate object` text.
    pub content: String,
    /// Rank-position relevance × stored confidence.
    pub score: f64,
    /// The `facts.fact_id` / `consolidated_facts.id`.
    pub fact_id: String,
    /// The fact subject.
    pub subject: String,
    /// The fact predicate.
    pub predicate: String,
}

/// The shared knobs of the entity-aware / fact-aware boost pass.
struct BoostArgs<'a> {
    ids: &'a [String],
    wm_where: &'a str,
    wm_params: &'a [Value],
    scope: &'a RecallScope,
    /// Score multiplier for ids already in results (`1.3` entity / `1.2` fact).
    multiplier: f64,
    /// New-row floor base (`0.6` entity / `0.5` fact).
    add_base: f64,
    /// Marks `entity_match` (true) or `fact_match` (false).
    entity: bool,
    wm_vec_sims: &'a HashMap<String, f64>,
    temporal: &'a dyn Fn(&str, f64) -> f64,
    halflife: f64,
}

/// `(min, range)` normalization bounds over FTS ranks (`beam.py` L5271-L5277 / L5645-L5649).
fn rank_bounds<'a>(ranks: impl Iterator<Item = &'a f64>) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for r in ranks {
        min = min.min(*r);
        max = max.max(*r);
    }
    if !min.is_finite() {
        return (0.0, 1.0);
    }
    let rng = if max != min { max - min } else { 1.0 };
    (min, rng)
}

/// FTS5 search over `fts_working`, `(id, rank)` ascending (`beam.py` `_fts_search_working`
/// L2456-L2474): stopword-filtered OR terms, with the CJK then Cyrillic LIKE fallbacks when the
/// query yields no terms or no rows.
fn fts_search_working(conn: &Connection, query: &str, k: usize) -> Vec<(String, f64)> {
    let terms = fts_query_terms(query);
    if terms.is_empty() {
        if has_cjk(query) {
            return cjk_like_search_working(conn, query, k);
        }
        if has_cyrillic(query) {
            return cyrillic_like_search_working(conn, query, k);
        }
        return Vec::new();
    }
    let match_expr = terms.join(" OR ");
    let fetch = || -> rusqlite::Result<Vec<(String, f64)>> {
        let mut stmt = conn.prepare(
            "SELECT id, rank FROM fts_working WHERE fts_working MATCH ?1 \
             ORDER BY rank, id LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![match_expr, k as i64], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    };
    let rows = fetch().unwrap_or_default();
    if rows.is_empty() && has_cjk(query) {
        return cjk_like_search_working(conn, query, k);
    }
    if rows.is_empty() && has_cyrillic(query) {
        return cyrillic_like_search_working(conn, query, k);
    }
    rows
}

/// FTS5 search over `fts_episodes`, `(rowid, rank)` ascending (`beam.py` `_fts_search`
/// L2423-L2453), with the same CJK/Cyrillic fallbacks.
fn fts_search_episodic(conn: &Connection, query: &str, k: usize) -> Vec<(i64, f64)> {
    let terms = fts_query_terms(query);
    if terms.is_empty() {
        if has_cjk(query) {
            return cjk_like_search_episodic(conn, query, k);
        }
        if has_cyrillic(query) {
            return cyrillic_like_search_episodic(conn, query, k);
        }
        return Vec::new();
    }
    let match_expr = terms.join(" OR ");
    let fetch = || -> rusqlite::Result<Vec<(i64, f64)>> {
        let mut stmt = conn.prepare(
            "SELECT rowid, rank FROM fts_episodes WHERE fts_episodes MATCH ?1 \
             ORDER BY rank, rowid LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![match_expr, k as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    };
    let rows = fetch().unwrap_or_default();
    if rows.is_empty() && has_cjk(query) {
        return cjk_like_search_episodic(conn, query, k);
    }
    if rows.is_empty() && has_cyrillic(query) {
        return cyrillic_like_search_episodic(conn, query, k);
    }
    rows
}

/// Shared CJK LIKE fallback core: rows matching any query CJK character, scored by the fraction of
/// unique query chars present, `rank = -score` (`beam.py` `_cjk_like_search` L2205-L2259).
fn cjk_like_rows(
    conn: &Connection,
    query: &str,
    k: usize,
    table: &str,
    id_col: &str,
) -> Vec<(Value, f64)> {
    let chars = cjk_chars(query);
    if chars.is_empty() {
        return Vec::new();
    }
    let conditions = vec!["content LIKE ? ESCAPE '\\'"; chars.len()].join(" OR ");
    let mut bind: Vec<Value> = chars
        .iter()
        .map(|c| Value::Text(format!("%{c}%")))
        .collect();
    bind.push(Value::Integer((k * 5) as i64));
    let fetch = || -> rusqlite::Result<Vec<(Value, String)>> {
        let mut stmt = conn.prepare(&format!(
            "SELECT {id_col}, content FROM {table} WHERE {conditions} LIMIT ?"
        ))?;
        let rows = stmt
            .query_map(params_from_iter(bind), |r| {
                Ok((r.get::<_, Value>(0)?, r.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    };
    let rows = fetch().unwrap_or_default();
    let mut scored: Vec<(Value, f64)> = rows
        .into_iter()
        .filter_map(|(id, content)| {
            let hits = chars.iter().filter(|c| content.contains(**c)).count();
            let score = hits as f64 / chars.len().max(1) as f64;
            (score > 0.0).then_some((id, score))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(id, s)| (id, -s)).collect()
}

fn cjk_like_search_working(conn: &Connection, query: &str, k: usize) -> Vec<(String, f64)> {
    cjk_like_rows(conn, query, k, "working_memory", "id")
        .into_iter()
        .filter_map(|(id, rank)| match id {
            Value::Text(id) => Some((id, rank)),
            _ => None,
        })
        .collect()
}

fn cjk_like_search_episodic(conn: &Connection, query: &str, k: usize) -> Vec<(i64, f64)> {
    cjk_like_rows(conn, query, k, "episodic_memory", "rowid")
        .into_iter()
        .filter_map(|(id, rank)| match id {
            Value::Integer(id) => Some((id, rank)),
            _ => None,
        })
        .collect()
}

/// Shared Cyrillic LIKE fallback core: candidate rows matching any 4-char query-word prefix
/// (case-folded via a registered Unicode `lower` UDF), re-ranked by trigram Jaccard,
/// `rank = -score` (`beam.py` `_cyrillic_like_search` L2355-L2420).
fn cyrillic_like_rows(
    conn: &Connection,
    query: &str,
    k: usize,
    table: &str,
    id_col: &str,
) -> Vec<(Value, f64)> {
    if !has_cyrillic(query) {
        return Vec::new();
    }
    // SQLite's LOWER()/LIKE are ASCII-only; register a real Unicode lower() (idempotent).
    let registered = conn
        .create_scalar_function(
            "_py_lower",
            1,
            rusqlite::functions::FunctionFlags::SQLITE_UTF8
                | rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
            |ctx| {
                let s: String = ctx.get(0)?;
                Ok(s.to_lowercase())
            },
        )
        .is_ok();
    if !registered {
        return Vec::new();
    }
    let q_words = cyrillic_words(query, 4);
    if q_words.is_empty() {
        return Vec::new();
    }
    let conditions = vec!["_py_lower(content) LIKE ? ESCAPE '\\'"; q_words.len()].join(" OR ");
    let mut bind: Vec<Value> = q_words
        .iter()
        .map(|w| {
            let prefix: String = w.chars().take(4).collect();
            Value::Text(format!("%{prefix}%"))
        })
        .collect();
    bind.push(Value::Integer((k * 5) as i64));
    let fetch = || -> rusqlite::Result<Vec<(Value, String)>> {
        let mut stmt = conn.prepare(&format!(
            "SELECT {id_col}, content FROM {table} WHERE {conditions} LIMIT ?"
        ))?;
        let rows = stmt
            .query_map(params_from_iter(bind), |r| {
                Ok((r.get::<_, Value>(0)?, r.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    };
    let rows = fetch().unwrap_or_default();
    let mut scored: Vec<(Value, f64)> = rows
        .into_iter()
        .filter_map(|(id, content)| {
            let score = cyrillic_score(query, &content);
            (score > 0.0).then_some((id, score))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(id, s)| (id, -s)).collect()
}

fn cyrillic_like_search_working(conn: &Connection, query: &str, k: usize) -> Vec<(String, f64)> {
    cyrillic_like_rows(conn, query, k, "working_memory", "id")
        .into_iter()
        .filter_map(|(id, rank)| match id {
            Value::Text(id) => Some((id, rank)),
            _ => None,
        })
        .collect()
}

fn cyrillic_like_search_episodic(conn: &Connection, query: &str, k: usize) -> Vec<(i64, f64)> {
    cyrillic_like_rows(conn, query, k, "episodic_memory", "rowid")
        .into_iter()
        .filter_map(|(id, rank)| match id {
            Value::Integer(id) => Some((id, rank)),
            _ => None,
        })
        .collect()
}

/// Working-memory vector search over the `memory_embeddings` JSON store with the recall filters
/// pushed into SQL, `(id, cosine)` descending (`beam.py` `_wm_vec_search_fallback` L2564-L2600;
/// the sqlite-vec `vec_working` fast path is a Phase-6 storage decision).
fn wm_vec_search(
    conn: &Connection,
    query: &[f32],
    k: usize,
    where_sql: &str,
    where_params: &[Value],
) -> Result<Vec<(String, f64)>> {
    let mut bind: Vec<Value> = where_params.to_vec();
    bind.push(Value::Integer(50_000));
    let mut stmt = conn.prepare(&format!(
        "SELECT wm.id, me.embedding_json FROM memory_embeddings me \
         JOIN working_memory wm ON me.memory_id = wm.id WHERE {where_sql} LIMIT ?"
    ))?;
    let rows: Vec<(String, String)> = stmt
        .query_map(params_from_iter(bind), |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut out: Vec<(String, f64)> = rows
        .into_iter()
        .filter_map(|(id, json)| {
            let vec: Vec<f32> = serde_json::from_str(&json).ok()?;
            Some((id, daemon_core::cosine(query, &vec) as f64))
        })
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(k);
    Ok(out)
}

/// Episodic in-memory vector search, `(rowid, distance)` ascending where `distance = 1 - cosine`
/// (`beam.py` `_in_memory_vec_search` L1723-L1760). The caller renormalizes by the max distance.
fn em_vec_search(conn: &Connection, query: &[f32], k: usize) -> Result<Vec<(i64, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT em.rowid, me.embedding_json FROM memory_embeddings me \
         JOIN episodic_memory em ON me.memory_id = em.id LIMIT 10000",
    )?;
    let rows: Vec<(i64, String)> = stmt
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut out: Vec<(i64, f64)> = rows
        .into_iter()
        .filter_map(|(rowid, json)| {
            let vec: Vec<f32> = serde_json::from_str(&json).ok()?;
            Some((rowid, 1.0 - daemon_core::cosine(query, &vec) as f64))
        })
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(k);
    Ok(out)
}

/// Memory ids whose `mentions` annotations fuzzy-match the query (`beam.py`
/// `_find_memories_by_entity` L1420-L1450): the whole query string is matched against the known
/// entity universe at threshold 0.8 via `entities::find_similar_entities`.
fn find_memories_by_entity(conn: &Connection, query: &str) -> Vec<String> {
    let known: Vec<String> = match annotations::query_by_kind(conn, "mentions", None, false) {
        Ok(rows) => {
            let mut values: Vec<String> = rows.into_iter().map(|a| a.value).collect();
            values.sort();
            values.dedup();
            values
        }
        Err(_) => return Vec::new(),
    };
    if known.is_empty() {
        return Vec::new();
    }
    let mut memory_ids: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (matched, _score) in entities::find_similar_entities(query, &known, 0.8) {
        if let Ok(rows) = annotations::query_by_kind(conn, "mentions", Some(&matched), false) {
            for ann in rows {
                if seen.insert(ann.memory_id.clone()) {
                    memory_ids.push(ann.memory_id);
                }
            }
        }
    }
    memory_ids
}

/// Memory ids whose extracted `fact` annotations match the query (`beam.py`
/// `_find_memories_by_fact` L1685-L1720): strict matcher by default, the legacy any-word
/// substring matcher when `lenient`.
fn find_memories_by_fact(conn: &Connection, query_lower: &str, lenient: bool) -> Vec<String> {
    let all_facts = match annotations::query_by_kind(conn, "fact", None, false) {
        Ok(rows) => rows,
        Err(_) => return Vec::new(),
    };
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();
    let mut memory_ids: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for fact in all_facts {
        let fact_text = fact.value.to_lowercase();
        let matched = if !lenient {
            strict_fact_matches(query_lower, &fact_text)
        } else {
            query_words.iter().any(|w| fact_text.contains(w)) || fact_text.contains(query_lower)
        };
        if matched && seen.insert(fact.memory_id.clone()) {
            memory_ids.push(fact.memory_id);
        }
    }
    memory_ids
}

/// Incident `graph_edges` count via substring match — gist/fact node ids embed the memory id, so
/// LIKE credits them to the parent memory (`beam.py` L5749-L5754).
fn graph_edge_count_like(conn: &Connection, memory_id: &str) -> Result<usize> {
    let like = format!("%{memory_id}%");
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM graph_edges WHERE source LIKE ?1 OR target LIKE ?1",
        params![like],
        |r| r.get(0),
    )?;
    Ok(count as usize)
}

/// The number of the row's `facts` sharing a `>2`-char whitespace token with the query
/// (`beam.py` L5756-L5771; note: plain `split()`, not the recall tokenizer).
fn fact_match_count(conn: &Connection, memory_id: &str, query_lower: &str) -> Result<usize> {
    let q_word_set: HashSet<&str> = query_lower
        .split_whitespace()
        .filter(|w| w.chars().count() > 2)
        .collect();
    if q_word_set.is_empty() {
        return Ok(0);
    }
    let mut stmt =
        conn.prepare("SELECT subject, predicate, object FROM facts WHERE source_msg_id = ?1")?;
    let rows: Vec<(String, String, String)> = stmt
        .query_map(params![memory_id], |r| {
            Ok((
                r.get::<_, Option<String>>(0)?.unwrap_or_default(),
                r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut match_count = 0;
    for (s, p, o) in rows {
        let text = format!("{s} {p} {o}").to_lowercase();
        let fact_tokens: HashSet<&str> = text
            .split_whitespace()
            .filter(|t| t.chars().count() > 2)
            .collect();
        if q_word_set.intersection(&fact_tokens).next().is_some() {
            match_count += 1;
        }
    }
    Ok(match_count)
}

/// The MIB binary-vector bonus: `0.08 * (1 - tanh(3 * hamming/dim))` (`beam.py` L5773-L5793).
fn binary_bonus(query_bv: &[u8], row_bv: &[u8]) -> f64 {
    if query_bv.len() != row_bv.len() || query_bv.is_empty() {
        return 0.0;
    }
    let dist = crate::binary_vectors::hamming_distance(query_bv, row_bv);
    let dim = (query_bv.len() * 8) as f64;
    crate::binary_vectors::binary_bonus(dist as f64 / dim)
}

/// E3.a.3 cross-tier dedup (`beam.py` `_dedup_cross_tier_summary_links` L6330-L6470): for each
/// episodic summary whose `summary_of` sources also surfaced, keep the summary only if it beats
/// every covered source (and no source is an exact `keyword >= 0.95` hit); otherwise keep the
/// sources and drop the summary. Preserves input order.
fn dedup_cross_tier_summary_links(
    results: Vec<MemoryRow>,
    ep_summary_of: &HashMap<String, String>,
) -> Vec<MemoryRow> {
    if !results.iter().any(|r| r.tier == Tier::Episodic) {
        return results;
    }
    let mut summary_map: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (ep_id, raw) in ep_summary_of {
        let wm_ids: HashSet<&str> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if !wm_ids.is_empty() {
            summary_map.insert(ep_id.as_str(), wm_ids);
        }
    }
    if summary_map.is_empty() {
        return results;
    }

    let wm_scores: HashMap<&str, f64> = results
        .iter()
        .filter(|r| r.tier == Tier::Working)
        .map(|r| (r.id.as_str(), r.score))
        .collect();
    let wm_keyword_scores: HashMap<&str, f64> = results
        .iter()
        .filter(|r| r.tier == Tier::Working)
        .map(|r| (r.id.as_str(), r.keyword_score))
        .collect();
    let ep_scores: HashMap<&str, f64> = results
        .iter()
        .filter(|r| r.tier == Tier::Episodic)
        .map(|r| (r.id.as_str(), r.score))
        .collect();

    let mut drop_wm: HashSet<String> = HashSet::new();
    let mut drop_ep: HashSet<String> = HashSet::new();
    for (ep_id, covered) in &summary_map {
        let Some(ep_score) = ep_scores.get(ep_id) else {
            continue;
        };
        let present: Vec<&str> = covered
            .iter()
            .copied()
            .filter(|w| wm_scores.contains_key(w))
            .collect();
        if present.is_empty() {
            continue;
        }
        // An exact/distinctive query hit on the raw working row keeps the original recallable.
        let exact_source_hit = present
            .iter()
            .any(|w| wm_keyword_scores.get(w).copied().unwrap_or(0.0) >= 0.95);
        let ep_wins = !exact_source_hit && present.iter().all(|w| *ep_score >= wm_scores[w]);
        if ep_wins {
            drop_wm.extend(present.iter().map(|w| (*w).to_string()));
        } else {
            drop_ep.insert((*ep_id).to_string());
        }
    }
    if drop_wm.is_empty() && drop_ep.is_empty() {
        return results;
    }
    results
        .into_iter()
        .filter(|r| {
            !((r.tier == Tier::Working && drop_wm.contains(&r.id))
                || (r.tier == Tier::Episodic && drop_ep.contains(&r.id)))
        })
        .collect()
}

/// Multi-aspect greedy selection (`beam.py` L6061-L6081): prefer rows that add not-yet-covered
/// exact query terms (`+0.06` per new term) while keeping score as the base signal. Returns
/// `selected + remaining pool` (the caller truncates to `top_k`).
fn greedy_aspect_select(
    results: Vec<MemoryRow>,
    query_words: &[String],
    top_k: usize,
) -> Vec<MemoryRow> {
    let q_word_set: HashSet<String> = expanded_query_tokens(query_words).into_iter().collect();
    let mut pool = results;
    let mut selected: Vec<MemoryRow> = Vec::new();
    let mut covered: HashSet<String> = HashSet::new();
    while !pool.is_empty() && selected.len() < top_k {
        let mut best_idx = 0;
        let mut best_key = f64::NEG_INFINITY;
        for (i, row) in pool.iter().enumerate() {
            let new_terms = recall_tokens(&row.content.to_lowercase())
                .into_iter()
                .filter(|t| q_word_set.contains(t) && !covered.contains(t))
                .collect::<HashSet<_>>()
                .len();
            let key = row.score + 0.06 * new_terms as f64;
            // Python's max() keeps the first maximum on ties.
            if key > best_key {
                best_key = key;
                best_idx = i;
            }
        }
        let picked = pool.remove(best_idx);
        for t in recall_tokens(&picked.content.to_lowercase()) {
            if q_word_set.contains(&t) {
                covered.insert(t);
            }
        }
        selected.push(picked);
    }
    selected.extend(pool);
    selected
}

/// Parse the temporal-boost target instant; `None`/invalid fall back to now (`beam.py`
/// `_parse_query_time` L1217-L1240; Python raises on invalid input — the tool layer validates, so
/// the engine stays lenient and logs).
fn parse_query_time(query_time: Option<&str>) -> chrono::DateTime<chrono::Utc> {
    let Some(raw) = query_time else {
        return chrono::Utc::now();
    };
    if let Some(dt) = parse_ts(raw) {
        return dt;
    }
    if let Some(dt) = parse_ts(&format!("{raw}T00:00:00")) {
        return dt;
    }
    tracing::debug!(query_time = raw, "invalid query_time; using now");
    chrono::Utc::now()
}

/// Temporal boost factor `exp(-hours_delta / halflife)`, future timestamps clamped to
/// `query_time`, invalid timestamps 0.0 (`beam.py` `_temporal_boost` L1264-L1285).
fn temporal_boost(
    memory_timestamp: &str,
    query_time: chrono::DateTime<chrono::Utc>,
    halflife_hours: f64,
) -> f64 {
    let Some(ts) = parse_ts(memory_timestamp) else {
        return 0.0;
    };
    let ts = ts.min(query_time);
    let hours_delta = (query_time - ts).num_seconds() as f64 / 3600.0;
    (-hours_delta / halflife_hours).exp()
}

/// Parse an ISO timestamp: RFC3339 first, then Python's naive `isoformat()` treated as UTC
/// (`beam.py` `_parse_ts_fast` L1246-L1260; legacy Python DBs store naive local timestamps).
pub(crate) fn parse_ts(ts: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if ts.is_empty() {
        return None;
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(ts) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    let naive = chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M:%S%.f")
        .or_else(|_| {
            chrono::NaiveDate::parse_from_str(ts, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
        })
        .ok()?;
    Some(chrono::DateTime::from_naive_utc_and_offset(
        naive,
        chrono::Utc,
    ))
}

/// Temporal cue words that activate the polyphonic temporal voice (`polyphonic_recall.py`
/// `_temporal_voice` L628-L633).
fn has_temporal_keyword(query: &str) -> bool {
    const TEMPORAL_KEYWORDS: &[&str] = &[
        "yesterday",
        "today",
        "recent",
        "last",
        "latest",
        "this week",
        "this month",
        "ago",
        "before",
    ];
    let lower = query.to_lowercase();
    TEMPORAL_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

/// Hours since an ISO timestamp (`None` if unparsable -> decay falls back to 0.5). Future
/// timestamps clamp to zero age.
pub(crate) fn age_hours(timestamp: &str) -> Option<f64> {
    let parsed = parse_ts(timestamp)?;
    let now = chrono::Utc::now();
    let delta = now.signed_duration_since(parsed);
    Some(delta.num_seconds().max(0) as f64 / 3600.0)
}

/// String helper for the consolidated-fact subject probe (Python `str.capitalize()`: first char
/// upper, rest lower).
fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
        None => String::new(),
    }
}
