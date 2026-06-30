// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Hybrid recall for the BEAM [`Engine`]: the base/enhanced/polyphonic pipelines, candidate
//! gathering + scoring across the working and episodic tiers, the MEMORIA supplement, scope-clause
//! construction, and the lexical/tokenizer helpers. Split out of `engine.rs` (W-MNEMO).

use super::*;
use crate::config::{RecallMode, RecallScope};
use crate::dynamics::weibull;
use crate::knowledge::{entities, episodic_graph};
use crate::recall::{mmr, polyphonic, query_intent, scoring, synonyms};
use crate::{binary_vectors, memoria};
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection};
use std::collections::{HashMap, HashSet};

impl Engine {
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
        let scope = self.config_scope();
        self.recall_with_scope(&RecallReq {
            query,
            top_k,
            query_vector,
            scope: &scope,
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

    /// As [`Engine::recall_with_vector`], but with an explicit [`RecallReq`] carrying the multi-agent
    /// identity [`RecallScope`] (the `mnemosyne_recall` tool's author/channel overrides). An empty
    /// scope is today's behavior.
    pub fn recall_with_scope(&self, req: &RecallReq) -> Result<Vec<MemoryRow>> {
        match self.config.recall_mode {
            RecallMode::Base => self.recall_base(req, scoring::DEFAULT_WEIGHTS),
            RecallMode::Enhanced => self.recall_enhanced(req),
            RecallMode::Polyphonic => self.recall_polyphonic(req),
        }
    }

    /// Build the recall scope SQL fragment (a leading ` AND ...`) plus its bound params for the
    /// given [`RecallScope`], mirroring `beam.py` L5182-L5220: a broad branch (channel / author-only
    /// / session) followed by exact author/channel filters.
    pub(crate) fn scope_clause(&self, scope: &RecallScope) -> (String, Vec<Value>) {
        let mut clause = String::new();
        let mut p: Vec<Value> = Vec::new();
        if let Some(channel) = &scope.channel_id {
            clause.push_str(" AND (session_id = ? OR scope = 'global' OR channel_id = ?)");
            p.push(Value::Text(self.config.session_id.clone()));
            p.push(Value::Text(channel.clone()));
        } else if scope.author_id.is_some() || scope.author_type.is_some() {
            clause.push_str(" AND (1=1)");
        } else {
            clause.push_str(" AND (session_id = ? OR scope = 'global')");
            p.push(Value::Text(self.config.session_id.clone()));
        }
        if let Some(author) = &scope.author_id {
            clause.push_str(" AND author_id = ?");
            p.push(Value::Text(author.clone()));
        }
        if let Some(author_type) = &scope.author_type {
            clause.push_str(" AND author_type = ?");
            p.push(Value::Text(author_type.clone()));
        }
        if let Some(channel) = &scope.channel_id {
            clause.push_str(" AND channel_id = ?");
            p.push(Value::Text(channel.clone()));
        }
        (clause, p)
    }

    /// The base hybrid cross-tier recall with explicit `(vec, fts, importance)` weights. This is the
    /// faithful port of `beam.py` `recall`; the enhanced/polyphonic pipelines build on it.
    fn recall_base(&self, req: &RecallReq, weights: (f64, f64, f64)) -> Result<Vec<MemoryRow>> {
        let q_tokens = tokenize(req.query);
        let q_entities = entities::extract_entities_regex(req.query);
        let floor = scoring::lexical_floor(q_tokens.len());
        let conn = self.store.conn.lock().unwrap();

        let ctx = GatherCtx {
            q_tokens: &q_tokens,
            q_entities: &q_entities,
            top_k: req.top_k,
            floor,
            query_vector: req.query_vector,
            weights,
            scope: req.scope,
        };
        let mut scored = self.gather_working(&conn, &ctx)?;
        let episodic = self.gather_episodic(&conn, &ctx)?;
        scored.extend(episodic);

        // Graph expansion: pull in memories that mention a query entity (or sit within two graph
        // hops of one) but were missed by the lexical/FTS/vector gates (`beam.py` L5760-L5793).
        let present: HashSet<String> = scored.iter().map(|r| r.id.clone()).collect();
        let injected = self.inject_entity_candidates(
            &conn,
            &EntityInjectCtx {
                q_entities: &q_entities,
                present: &present,
                scope: req.scope,
            },
        )?;
        scored.extend(injected);

        // Cross-tier dedup by normalized content, keeping the higher-scoring row (`beam.py` L6003).
        dedup_by_content(&mut scored);
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // MEMORIA structured-fact supplement (`beam.py` L6006-L6059): a high-relevance hit from the
        // specialist tables enters as an extra candidate (capped at 0.6) plus its source rows
        // (capped at 0.59), then we re-sort. Best-effort and only for query-shaped inputs.
        self.supplement_with_memoria(&conn, req.query, &mut scored)?;

        // Diversity rerank for multi-token queries (`beam.py` L6061), else a plain top-k slice.
        let selected: Vec<MemoryRow> = if q_tokens.len() >= 4 && scored.len() > 1 {
            let items: Vec<(String, f64)> = scored
                .iter()
                .map(|r| (r.content.clone(), r.score))
                .collect();
            mmr::mmr_rerank(&items, mmr::DEFAULT_LAMBDA, req.top_k)
                .into_iter()
                .map(|i| scored[i].clone())
                .collect()
        } else {
            scored.truncate(req.top_k);
            scored
        };

        self.bump_recall(&conn, &selected)?;
        Ok(selected)
    }

    /// Fold a high-relevance MEMORIA structured-fact hit into the candidate set
    /// (`beam.py` L6006-L6059). The hit's lexical relevance must clear `0.35`; it then enters as a
    /// `tier="memoria"` row scored `min(0.6, rel*0.6)` plus its originating `working_memory` rows as
    /// `tier="memoria_source"` rows scored `min(0.59, 0.2 + rel*0.8)` (content truncated to 500).
    /// Candidates are re-sorted by score afterward. Best-effort: failures are swallowed.
    fn supplement_with_memoria(
        &self,
        conn: &Connection,
        query: &str,
        scored: &mut Vec<MemoryRow>,
    ) -> Result<()> {
        let result = match memoria::memoria_retrieve(conn, &self.config.session_id, query, 3) {
            Some(r) if r.source != "fallback" && !r.context.is_empty() => r,
            _ => return Ok(()),
        };
        let rel = lexical_relevance(&tokenize(query), &result.context);
        if rel < 0.35 {
            return Ok(());
        }
        let memoria_score = round4((rel * 0.6).min(0.6));
        scored.push(MemoryRow {
            id: format!("memoria_{}", result.source),
            content: format!("[MEMORIA {}]\n{}", result.source, result.context),
            source: format!("memoria_{}", result.source),
            timestamp: String::new(),
            importance: 0.5,
            veracity: "unknown".to_string(),
            trust_tier: String::new(),
            tier: Tier::Working,
            tier_level: 1,
            score: memoria_score,
        });

        let source_score = round4((0.2 + rel * 0.8).min(0.59));
        for sid in &result.source_memory_ids {
            if sid.is_empty() {
                continue;
            }
            let row = conn.query_row(
                "SELECT id, content, source, timestamp, importance, veracity FROM working_memory WHERE id = ?1",
                params![sid],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        r.get::<_, f64>(4)?,
                        r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    ))
                },
            );
            if let Ok((id, content, source, timestamp, importance, veracity)) = row {
                let truncated: String = content.chars().take(500).collect();
                scored.push(MemoryRow {
                    id: format!("memoria_source_{id}"),
                    content: truncated,
                    source,
                    timestamp,
                    importance,
                    veracity,
                    trust_tier: String::new(),
                    tier: Tier::Working,
                    tier_level: 1,
                    score: source_score,
                });
            }
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
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
            scoring::DEFAULT_WEIGHTS
        } else {
            query_intent::adjust_weights(scoring::DEFAULT_WEIGHTS, intent)
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
            },
            weights,
        )?;

        // 5. Weibull re-scoring by memory type.
        self.weibull_rescore(&mut results)?;

        // 6-7. Sort, then MMR-diversify down to `top_k`.
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let final_results: Vec<MemoryRow> = if results.len() > 1 {
            let items: Vec<(String, f64)> = results
                .iter()
                .map(|r| (r.content.clone(), r.score))
                .collect();
            mmr::mmr_rerank(&items, mmr::DEFAULT_LAMBDA, req.top_k)
                .into_iter()
                .map(|i| results[i].clone())
                .collect()
        } else {
            results.truncate(req.top_k);
            results
        };

        // 8. Cache the result for next time.
        self.query_cache()
            .put(req.query, &final_results, req.query_vector);
        Ok(final_results)
    }

    /// Blend the per-type Weibull temporal boost into each row's score (`beam.py` L6266-L6278):
    /// `score = score*0.7 + weibull_boost*0.3`. The memory type is read from the row's tier table;
    /// missing/`unknown` types fall back to `general`.
    fn weibull_rescore(&self, rows: &mut [MemoryRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let conn = self.store.conn.lock().unwrap();
        for row in rows.iter_mut() {
            let table = match row.tier {
                Tier::Working => "working_memory",
                Tier::Episodic => "episodic_memory",
            };
            let mt: Option<String> = conn
                .query_row(
                    &format!("SELECT memory_type FROM {table} WHERE id = ?1"),
                    params![row.id],
                    |r| r.get(0),
                )
                .ok();
            let mut memory_type = mt.unwrap_or_default();
            if memory_type.is_empty() || memory_type == "unknown" {
                memory_type = "general".to_string();
            }
            let wb = weibull::weibull_boost(age_hours(&row.timestamp), &memory_type);
            row.score = row.score * 0.7 + wb * 0.3;
        }
        Ok(())
    }

    /// Four-voice polyphonic recall (`polyphonic_recall.py`, `MNEMOSYNE_POLYPHONIC_RECALL=1`):
    /// gathers a **vector** voice (cosine normalized `(cos+1)/2`, top 20), a **graph** voice (query
    /// entities seed `facts` subjects at `0.6`, then `ctx`-edge traversal at `0.4/depth`), a **fact**
    /// voice (consolidated `facts` whose subject is a query word, `confidence >= 0.5`), and a
    /// **temporal** voice (last-7-day working rows, `exp(-age_days/7)*importance`, only on temporal
    /// keywords), then fuses them with RRF ([`polyphonic::fuse`]), diversity-reranks, and resolves
    /// the surviving ids to rows. Voice weights stay metadata-only (fusion is pure RRF).
    fn recall_polyphonic(&self, req: &RecallReq) -> Result<Vec<MemoryRow>> {
        use polyphonic::VoiceHit;
        let conn = self.store.conn.lock().unwrap();

        // Voice 1: vector (cosine over stored embeddings, normalized to [0, 1], top 20).
        let mut vector_hits: Vec<VoiceHit> = Vec::new();
        if let Some(q) = req.query_vector {
            let mut sims: Vec<(String, f64)> = cosine_sim_map(&conn, q)?
                .into_iter()
                .map(|(id, cos)| (id, (cos + 1.0) / 2.0))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.truncate(20);
            vector_hits = sims
                .into_iter()
                .map(|(id, s)| VoiceHit {
                    memory_id: id,
                    score: s,
                })
                .collect();
        }

        // Voice 2: graph (entity-seeded gists @0.6 + fact subjects @conf*0.5, then ctx-edge
        // traversal @0.4/depth from all seeds).
        let q_entities = entities::extract_entities_regex(req.query);
        let mut graph_hits: Vec<VoiceHit> = Vec::new();
        let mut seen_graph: HashSet<String> = HashSet::new();
        let mut seed_ids: HashSet<String> = HashSet::new();
        for ent in &q_entities {
            for (mid, _text) in episodic_graph::find_gists_by_participant(&conn, ent)? {
                if seen_graph.insert(mid.clone()) {
                    graph_hits.push(VoiceHit {
                        memory_id: mid.clone(),
                        score: 0.6,
                    });
                }
                seed_ids.insert(mid);
            }
            for (mid, conf) in self.facts_for_subject(&conn, ent, 0.0)? {
                if seen_graph.insert(mid.clone()) {
                    graph_hits.push(VoiceHit {
                        memory_id: mid.clone(),
                        score: conf * 0.5,
                    });
                }
                seed_ids.insert(mid);
            }
        }
        for seed in &seed_ids {
            for rel in episodic_graph::find_related_memories(&conn, seed, 2, "ctx", 0.3)? {
                if !seed_ids.contains(&rel.memory_id) && seen_graph.insert(rel.memory_id.clone()) {
                    graph_hits.push(VoiceHit {
                        memory_id: rel.memory_id,
                        score: 0.4 / rel.depth as f64,
                    });
                }
            }
        }

        // Voice 3: fact (query words matched against consolidated `facts` subjects).
        let mut fact_hits: Vec<VoiceHit> = Vec::new();
        let mut seen_fact: HashSet<String> = HashSet::new();
        for word in tokenize(req.query) {
            if word.chars().count() < 3 {
                continue;
            }
            for (mid, conf) in self.facts_for_subject(&conn, &word, 0.5)? {
                if seen_fact.insert(mid.clone()) {
                    fact_hits.push(VoiceHit {
                        memory_id: mid,
                        score: conf,
                    });
                }
            }
        }

        // Voice 4: temporal (recent working rows, only when the query has a temporal cue).
        let mut temporal_hits: Vec<VoiceHit> = Vec::new();
        if has_temporal_keyword(req.query) {
            let week_ago = (chrono::Utc::now() - chrono::Duration::days(7)).to_rfc3339();
            let (scope_sql, scope_params) = self.scope_clause(req.scope);
            let sql = format!(
                "SELECT id, timestamp, importance FROM working_memory \
                 WHERE timestamp > ? AND superseded_by IS NULL{scope_sql} \
                 ORDER BY timestamp DESC LIMIT 20",
            );
            let mut stmt = conn.prepare(&sql)?;
            let mut bind: Vec<Value> = vec![Value::Text(week_ago)];
            bind.extend(scope_params);
            let rows = stmt.query_map(params_from_iter(bind), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, f64>(2)?,
                ))
            })?;
            for row in rows.flatten() {
                let (id, ts, imp) = row;
                let age_days = age_hours(&ts).unwrap_or(0.0) / 24.0;
                let tscore = (-age_days / 7.0).exp() * imp;
                temporal_hits.push(VoiceHit {
                    memory_id: id,
                    score: tscore,
                });
            }
        }

        let fused = polyphonic::fuse(&[
            ("vector", vector_hits),
            ("graph", graph_hits),
            ("fact", fact_hits),
            ("temporal", temporal_hits),
        ]);
        let diversified = polyphonic::diversity_rerank(fused, req.top_k);

        let mut out: Vec<MemoryRow> = Vec::new();
        for f in diversified {
            let row = match self.fetch_working(&conn, &f.memory_id, req.scope)? {
                Some(r) => Some(r),
                None => self.fetch_episodic(&conn, &f.memory_id, req.scope)?,
            };
            if let Some(mut r) = row {
                r.score = f.combined_score;
                out.push(r);
            }
        }
        self.bump_recall(&conn, &out)?;
        Ok(out)
    }

    /// `(memory_id, confidence)` for `facts` whose subject matches `subject` at or above
    /// `min_confidence` (the polyphonic fact voice).
    fn facts_for_subject(
        &self,
        conn: &Connection,
        subject: &str,
        min_confidence: f64,
    ) -> Result<Vec<(String, f64)>> {
        let mut stmt = conn.prepare(
            "SELECT source_msg_id, confidence FROM facts \
             WHERE subject = ?1 COLLATE NOCASE AND confidence >= ?2 AND source_msg_id IS NOT NULL",
        )?;
        let rows = stmt.query_map(params![subject, min_confidence], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
        })?;
        Ok(rows.flatten().collect())
    }

    /// Gather + score working-memory candidates (FTS5 ∪ vector ∪ recency fallback), with the
    /// knowledge-layer graph/fact bonuses and entity/fact multipliers (`beam.py` L5760-L5793).
    fn gather_working(&self, conn: &Connection, ctx: &GatherCtx) -> Result<Vec<MemoryRow>> {
        // Base candidates: the recency/importance fallback scan (limit 2000, `beam.py` L5262), plus
        // any FTS5 hits that fall outside that window.
        let mut rows = self.scan_working(conn, 2000, ctx.scope)?;
        let mut seen: HashSet<String> = rows.iter().map(|r| r.id.clone()).collect();
        let fts = self.fts_search(
            conn,
            "SELECT id, bm25(fts_working) FROM fts_working \
             WHERE fts_working MATCH ?1 ORDER BY bm25(fts_working) LIMIT ?2",
            ctx.q_tokens,
            (ctx.top_k * 3).max(50),
        )?;
        for id in fts.keys() {
            if seen.insert(id.clone()) {
                if let Some(row) = self.fetch_working(conn, id, ctx.scope)? {
                    rows.push(row);
                }
            }
        }

        let sims = match ctx.query_vector {
            Some(q) => cosine_sim_map(conn, q)?,
            None => HashMap::new(),
        };
        let (_vw, _fw, iw) = ctx.weights;

        let mut scored = Vec::new();
        for mut row in rows {
            let lexical = lexical_relevance(ctx.q_tokens, &row.content);
            let nfts = fts.get(&row.id).copied().unwrap_or(0.0);
            let vec_sim = sims.get(&row.id).copied().unwrap_or(0.0);
            if lexical < ctx.floor && vec_sim < VEC_SIM_FLOOR && nfts <= 0.0 {
                continue;
            }
            let relevance = scoring::blend_fts(lexical, nfts, ctx.floor);
            let decay = scoring::recency_decay(age_hours(&row.timestamp));
            let bonuses = self.knowledge_bonuses(conn, &row.id, ctx.q_entities)?;
            let mut base =
                scoring::working_memory_score(relevance, row.importance, iw, vec_sim, decay);
            base += bonuses.graph_bonus + bonuses.fact_bonus;
            base = bonuses.apply_multipliers(base);
            row.score = base * scoring::veracity_multiplier(&row.veracity);
            scored.push(row);
        }
        Ok(scored)
    }

    /// Gather + score episodic candidates (FTS5 ∪ vector ∪ recency fallback), with the MIB binary
    /// bonus and the tier/veracity post-multipliers (`beam.py` L5720-L5976).
    fn gather_episodic(&self, conn: &Connection, ctx: &GatherCtx) -> Result<Vec<MemoryRow>> {
        let mut rows = self.scan_episodic(conn, 2000, ctx.scope)?;
        let mut seen: HashSet<String> = rows.iter().map(|r| r.id.clone()).collect();
        let fts = self.fts_search(
            conn,
            "SELECT e.id, bm25(fts_episodes) FROM fts_episodes f \
             JOIN episodic_memory e ON e.rowid = f.rowid \
             WHERE fts_episodes MATCH ?1 ORDER BY bm25(fts_episodes) LIMIT ?2",
            ctx.q_tokens,
            (ctx.top_k * 3).max(20),
        )?;
        for id in fts.keys() {
            if seen.insert(id.clone()) {
                if let Some(row) = self.fetch_episodic(conn, id, ctx.scope)? {
                    rows.push(row);
                }
            }
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let sims = match ctx.query_vector {
            Some(q) => cosine_sim_map(conn, q)?,
            None => HashMap::new(),
        };
        let binaries = self.load_binary_vectors(conn)?;
        let q_bin = ctx
            .query_vector
            .map(binary_vectors::maximally_informative_binarization);

        let mut scored = Vec::new();
        for mut row in rows {
            let lexical = lexical_relevance(ctx.q_tokens, &row.content);
            let nfts = fts.get(&row.id).copied().unwrap_or(0.0);
            let sim = sims.get(&row.id).copied().unwrap_or(0.0);
            // Weak-signal gate (`beam.py` L5720): drop unless lexical, FTS, or vector say keep.
            if lexical < ctx.floor && sim < VEC_SIM_FLOOR && nfts <= 0.0 {
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
            let bonuses = self.knowledge_bonuses(conn, &row.id, ctx.q_entities)?;
            let base = scoring::episodic_score(
                sim,
                nfts,
                row.importance,
                lexical,
                decay,
                ctx.weights,
                bonuses.graph_bonus,
                bonuses.fact_bonus,
                binary_bonus,
            );
            let base = bonuses.apply_multipliers(base);
            row.score = base
                * scoring::tier_weight(row.tier_level)
                * scoring::veracity_multiplier(&row.veracity);
            scored.push(row);
        }
        Ok(scored)
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

/// Lexical relevance (`beam.py` `_lexical_relevance` L1573-L1638): `(exact_token_hits + partial +
/// full_match)/len`, where a query token absent from the content earns `+0.75` for a
/// [`synonyms::recall_synonyms`] hit (beam's conservative `_RECALL_SYNONYMS` map, L1608-L1611), else
/// `+0.4` for a `>=4`-char substring overlap; a whole-query substring adds `1.0`. Clamped to `[0, 1]`.
pub(crate) fn lexical_relevance(query_tokens: &[String], content: &str) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let lc = content.to_lowercase();
    let content_tokens: HashSet<String> = tokenize(content).into_iter().collect();
    let mut num = 0.0;
    for t in query_tokens {
        if content_tokens.contains(t) {
            num += 1.0;
            continue;
        }
        let syns = synonyms::recall_synonyms(t);
        if !syns.is_empty() && syns.iter().any(|s| content_tokens.contains(*s)) {
            num += 0.75;
            continue;
        }
        if t.len() >= 4 && lc.contains(t.as_str()) {
            num += 0.4;
        }
    }
    if lc.contains(&query_tokens.join(" ")) {
        num += 1.0;
    }
    (num / query_tokens.len() as f64).min(1.0)
}

/// Round to four decimal places (the MEMORIA supplement's `round(x, 4)`, `beam.py` L6021/L6048).
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
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

/// Hours since an ISO timestamp (`None` if unparsable -> decay falls back to 0.5).
pub(crate) fn age_hours(timestamp: &str) -> Option<f64> {
    let parsed = chrono::DateTime::parse_from_rfc3339(timestamp).ok()?;
    let now = chrono::Utc::now();
    let delta = now.signed_duration_since(parsed.with_timezone(&chrono::Utc));
    Some(delta.num_seconds().max(0) as f64 / 3600.0)
}
