// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Sleep / consolidation for the BEAM [`Engine`]: WM->episodic promotion, the plan/finish sleep
//! split, heuristic conflict detection, tiered episodic degradation, and the veracity-conflict
//! resolution surface. Split out of `engine.rs` (W-MNEMO).

use super::*;
use crate::binary_vectors;
use crate::knowledge::veracity;
use crate::util;
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection};
use std::collections::{HashMap, HashSet};

impl Engine {
    /// Unresolved `(subject, predicate)` contradictions recorded during consolidation, each with the
    /// reconstructed older/newer fact text for LLM validation (`fact_a` is the newer fact,
    /// `fact_b` the existing/older one — see [`veracity::record_conflict`] call in `consolidate_fact`).
    pub fn pending_conflicts(&self) -> Result<Vec<PendingConflict>> {
        let conn = self.store.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT c.id, a.id, a.subject || ' ' || a.predicate || ' ' || a.object, \
                    b.id, b.subject || ' ' || b.predicate || ' ' || b.object \
             FROM conflicts c \
             JOIN consolidated_facts a ON a.id = c.fact_a_id \
             JOIN consolidated_facts b ON b.id = c.fact_b_id \
             WHERE c.resolution IS NULL",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(PendingConflict {
                conflict_id: r.get::<_, i64>(0)?,
                newer_fact_id: r.get::<_, String>(1)?,
                newer_text: r.get::<_, String>(2)?,
                older_fact_id: r.get::<_, String>(3)?,
                older_text: r.get::<_, String>(4)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Resolve a recorded conflict (`veracity_consolidation.py` resolution path). On a confirmed
    /// conflict the older fact is marked `superseded_by` the newer one; either way the `conflicts`
    /// row is stamped with the LLM resolution + timestamp so it is not re-validated.
    pub fn resolve_conflict(
        &self,
        conflict_id: i64,
        confirmed: bool,
        winner_fact_id: &str,
        loser_fact_id: &str,
    ) -> Result<()> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();
        if confirmed {
            conn.execute(
                "UPDATE consolidated_facts SET superseded_by = ?1, updated_at = ?2 WHERE id = ?3",
                params![winner_fact_id, now, loser_fact_id],
            )?;
        }
        let resolution = if confirmed {
            "llm_confirmed"
        } else {
            "llm_rejected"
        };
        conn.execute(
            "UPDATE conflicts SET resolution = ?1, resolved_at = ?2 WHERE id = ?3",
            params![resolution, now, conflict_id],
        )?;
        Ok(())
    }

    /// Promote unconsolidated working-memory rows into the episodic tier (a minimal slice of
    /// `beam.py` `sleep`/consolidation L7576: no LLM summarization or tier degradation yet). Each
    /// promoted row is copied into `episodic_memory` at tier 1 — computing its MIB `binary_vector`
    /// from any stored embedding — its source working row is marked `consolidated_at`, and a
    /// `consolidation_log` entry is written. Pinned rows are exempt, exactly as in sleep
    /// (`beam.py` L7614). Returns the number of rows promoted.
    pub fn consolidate(&self) -> Result<usize> {
        let conn = self.store.conn.lock().unwrap();
        let embeddings = load_embeddings(&conn)?;
        let mut pending: Vec<EpisodicSeed> = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT id, content, source, timestamp, importance, veracity, trust_tier, scope, \
                        memory_type, event_date, event_date_precision, temporal_tags \
                 FROM working_memory \
                 WHERE consolidated_at IS NULL AND session_id = ?1 AND superseded_by IS NULL \
                   AND (pinned IS NULL OR pinned = 0)",
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
                    scope: r
                        .get::<_, Option<String>>(7)?
                        .unwrap_or_else(|| "global".into()),
                    memory_type: r
                        .get::<_, Option<String>>(8)?
                        .unwrap_or_else(|| "unknown".into()),
                    event_date: r.get::<_, Option<String>>(9)?,
                    event_date_precision: r
                        .get::<_, Option<String>>(10)?
                        .unwrap_or_else(|| "unknown".into()),
                    temporal_tags: r
                        .get::<_, Option<String>>(11)?
                        .unwrap_or_else(|| "[]".into()),
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
            let ep_id = util::memory_id(&format!(
                "episodic:{}:{}",
                self.config.session_id, seed.content
            ));
            let binary = embeddings
                .get(&seed.wm_id)
                .map(|v| binary_vectors::maximally_informative_binarization(v));
            conn.execute(
                "INSERT OR IGNORE INTO episodic_memory \
                 (id, content, source, timestamp, session_id, importance, metadata_json, veracity, \
                  memory_type, tier, binary_vector, scope, trust_tier, summary_of, \
                  event_date, event_date_precision, temporal_tags) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, '{}', ?7, ?8, 1, ?9, ?10, ?11, ?15, ?12, ?13, ?14)",
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
                    seed.event_date,
                    seed.event_date_precision,
                    seed.temporal_tags,
                    // summary_of: the source wm id (comma format) so recall's cross-tier dedup can
                    // link the promoted copy back to its still-live working row.
                    seed.wm_id,
                ],
            )?;
            // Carry the dense embedding over to the episodic id (Python embeds consolidation
            // output, `beam.py` L4030): episodic vector recall joins `memory_embeddings` on the
            // episodic id, so without this copy the promoted row would be invisible to the vec
            // voice.
            conn.execute(
                "INSERT OR IGNORE INTO memory_embeddings (memory_id, embedding_json, model) \
                 SELECT ?2, embedding_json, model FROM memory_embeddings WHERE memory_id = ?1",
                params![seed.wm_id, ep_id],
            )?;
            conn.execute(
                "UPDATE working_memory SET consolidated_at = ?2 WHERE id = ?1",
                params![seed.wm_id, now],
            )?;
            // Mirror the deterministic knowledge layer onto the episodic id so the episodic recall
            // tier carries its own entity/fact/graph signals.
            self.ingest_graph_and_veracity(&conn, &ep_id, &seed.content, &seed.veracity);
            self.emit_event(
                &conn,
                "CONSOLIDATE",
                &ep_id,
                Some(&seed.content),
                "consolidation",
                seed.importance,
            );
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

    /// The summarization prompt for one source group (`beam.py` sleep LLM path L7749). Built here so
    /// the async provider and the engine agree on the contract.
    pub fn summary_prompt(contents: &[String]) -> String {
        format!(
            "Summarize the following related memories into a single concise note that preserves all \
             durable facts, names, decisions, and dates. Be terse; no preamble.\n\n{}\n\nSummary:",
            contents.join("\n- ")
        )
    }

    /// Plan a sleep pass (`beam.py` sleep L7597-L7676): select eligible working rows (older than
    /// the `TTL/2` cutoff unless `force`, skipping pinned/consolidated rows, oldest-first, capped
    /// at [`MnemosyneConfig::sleep_batch_size`]), **atomically claim** them (set
    /// `consolidated_at`/`consolidation_claimed_at` gated on `consolidated_at IS NULL` for
    /// crash-/concurrency-safety), and group the claimed rows by source. The caller (the async
    /// provider) summarizes each group and hands the summaries to [`Engine::finish_sleep`]. Returns
    /// an empty vec when nothing is eligible.
    pub fn sleep_plan(&self, force: bool) -> Result<Vec<SleepGroup>> {
        self.sleep_plan_inner(force, true)
    }

    /// A non-mutating sleep plan: the same candidate selection and grouping as
    /// [`Engine::sleep_plan`] but WITHOUT the atomic claim, so nothing changes state
    /// (`beam.py` sleep L7639-L7641: "The dry_run branch skips the claim entirely").
    pub fn sleep_plan_dry_run(&self, force: bool) -> Result<Vec<SleepGroup>> {
        self.sleep_plan_inner(force, false)
    }

    fn sleep_plan_inner(&self, force: bool, claim: bool) -> Result<Vec<SleepGroup>> {
        let conn = self.store.conn.lock().unwrap();
        let cutoff = if force {
            "9999-12-31T23:59:59+00:00".to_string()
        } else {
            // TTL/2 hours, expressed in minutes so fractional configured TTLs survive.
            let half_ttl_minutes = (self.config.working_memory_ttl_hours * 30.0) as i64;
            (chrono::Utc::now() - chrono::Duration::minutes(half_ttl_minutes)).to_rfc3339()
        };

        struct Claimable {
            id: String,
            content: String,
            source: String,
            scope: String,
            valid_until: Option<String>,
            veracity: String,
        }
        let candidates: Vec<Claimable> = {
            let mut stmt = conn.prepare(
                "SELECT id, content, COALESCE(source, 'conversation'), \
                        COALESCE(scope, 'session'), valid_until, COALESCE(veracity, 'unknown') \
                 FROM working_memory \
                 WHERE COALESCE(session_id, 'default') = ?1 \
                   AND timestamp < ?2 \
                   AND consolidated_at IS NULL \
                   AND (pinned IS NULL OR pinned = 0) \
                   AND superseded_by IS NULL \
                 ORDER BY timestamp ASC LIMIT ?3",
            )?;
            let rows = stmt.query_map(
                params![
                    self.config.session_id,
                    cutoff,
                    self.config.sleep_batch_size as i64
                ],
                |r| {
                    Ok(Claimable {
                        id: r.get(0)?,
                        content: r.get(1)?,
                        source: r.get(2)?,
                        scope: r.get(3)?,
                        valid_until: r.get(4)?,
                        veracity: r.get(5)?,
                    })
                },
            )?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Atomic claim: mark consolidated_at gated on it still being NULL, then keep only the rows we
        // actually won (a concurrent sleep may have claimed some). Dry-run keeps all candidates.
        let now = util::now_iso();
        let claimed: Vec<Claimable> = if claim {
            let mut won: Vec<Claimable> = Vec::new();
            for c in candidates {
                let n = conn.execute(
                    "UPDATE working_memory SET consolidated_at = ?2, consolidation_claimed_at = ?2 \
                     WHERE id = ?1 AND consolidated_at IS NULL",
                    params![c.id, now],
                )?;
                if n == 1 {
                    won.push(c);
                }
            }
            won
        } else {
            candidates
        };
        if claimed.is_empty() {
            return Ok(Vec::new());
        }

        // Group by source, aggregating scope/veracity/valid_until (`beam.py` L7674-L7703).
        let mut order: Vec<String> = Vec::new();
        let mut groups: HashMap<String, SleepGroup> = HashMap::new();
        let mut veracities: HashMap<String, Vec<String>> = HashMap::new();
        for c in claimed {
            let g = groups.entry(c.source.clone()).or_insert_with(|| {
                order.push(c.source.clone());
                SleepGroup {
                    source: c.source.clone(),
                    ids: Vec::new(),
                    contents: Vec::new(),
                    scope: "session".to_string(),
                    veracity: "unknown".to_string(),
                    valid_until: None,
                }
            });
            g.ids.push(c.id);
            g.contents.push(c.content);
            if c.scope == "global" {
                g.scope = "global".to_string();
            }
            if let Some(vu) = c.valid_until {
                g.valid_until = Some(match g.valid_until.take() {
                    Some(existing) if existing < vu => existing,
                    _ => vu,
                });
            }
            veracities.entry(c.source).or_default().push(c.veracity);
        }
        for (source, vs) in &veracities {
            if let Some(g) = groups.get_mut(source) {
                g.veracity = veracity::aggregate_veracity(vs);
            }
        }
        Ok(order
            .into_iter()
            .filter_map(|s| groups.remove(&s))
            .collect())
    }

    /// Write the episodic summaries for the claimed [`SleepGroup`]s (`beam.py` sleep L7784-L7824,
    /// via `consolidate_to_episodic` L3956-L4049), then run tiered degradation and the veracity
    /// auto-resolution pass. `summaries` maps a group's `source` to its [`GroupSummary`]; a group
    /// with no entry falls back to the deterministic AAAK summary `[source] <aaak>`. Each summary
    /// is `<think>`-stripped and typed-memory classified; its embedding (when the async seam
    /// supplied one) lands in `memory_embeddings` with an MIB `binary_vector` on the episodic row
    /// (`beam.py` L4005-L4032). Clears each group's `consolidation_claimed_at`, writes a
    /// `consolidation_log` row, and returns the report.
    pub fn finish_sleep(
        &self,
        groups: &[SleepGroup],
        summaries: &HashMap<String, GroupSummary>,
    ) -> Result<SleepReport> {
        let mut report = SleepReport::default();
        if !groups.is_empty() {
            let conn = self.store.conn.lock().unwrap();
            for group in groups {
                // Strip closed <think>...</think> blocks (`beam.py` L3991-L3993); an LLM summary
                // that strips to nothing falls back to AAAK.
                let (summary, llm, embedding, model) = match summaries.get(&group.source) {
                    Some(gs) if !util::strip_think(&gs.text).is_empty() => (
                        util::strip_think(&gs.text),
                        gs.llm,
                        gs.embedding.clone(),
                        gs.model.clone(),
                    ),
                    _ => (group.aaak_summary(), false, None, String::new()),
                };
                let ep_id =
                    util::memory_id(&format!("episodic:{}:{}", self.config.session_id, summary));
                // Typed-memory classification of the summary (`beam.py` L3976-L3982).
                let memory_type = crate::dynamics::typed_memory::classify(&summary).as_str();
                // Comma-joined source ids (`beam.py` `consolidate_to_episodic` L4001); recall's
                // cross-tier dedup splits on ",".
                let summary_of = group.ids.join(",");
                // Multi-agent identity stamps, as on the write path (`beam.py` L3998-L4002).
                let channel_id = self
                    .config
                    .channel_id
                    .clone()
                    .unwrap_or_else(|| self.config.session_id.clone());
                conn.execute(
                    "INSERT OR IGNORE INTO episodic_memory \
                     (id, content, source, timestamp, session_id, importance, metadata_json, \
                      veracity, memory_type, tier, scope, summary_of, valid_until, \
                      author_id, author_type, channel_id) \
                     VALUES (?1, ?2, 'sleep_consolidation', ?3, ?4, 0.6, '{}', ?5, ?9, 1, \
                             ?6, ?7, ?8, ?10, ?11, ?12)",
                    params![
                        ep_id,
                        summary,
                        util::now_iso(),
                        self.config.session_id,
                        group.veracity,
                        group.scope,
                        summary_of,
                        group.valid_until,
                        memory_type,
                        self.config.author_id,
                        self.config.author_type,
                        channel_id,
                    ],
                )?;
                // Embed + binarize the consolidation output (`beam.py` L4005-L4032): without this
                // the summary row would be invisible to vector recall and the MIB binary bonus.
                if let Some(vec) = &embedding {
                    conn.execute(
                        "INSERT OR REPLACE INTO memory_embeddings \
                         (memory_id, embedding_json, model) VALUES (?1, ?2, ?3)",
                        params![ep_id, serde_json::to_string(vec)?, model],
                    )?;
                    conn.execute(
                        "UPDATE episodic_memory SET binary_vector = ?2 WHERE id = ?1",
                        params![
                            ep_id,
                            binary_vectors::maximally_informative_binarization(vec)
                        ],
                    )?;
                }
                self.ingest_graph_and_veracity(&conn, &ep_id, &summary, &group.veracity);
                self.emit_event(
                    &conn,
                    "CONSOLIDATE",
                    &ep_id,
                    Some(&summary),
                    "sleep_consolidation",
                    0.6,
                );
                let placeholders = group.ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let sql = format!(
                    "UPDATE working_memory SET consolidation_claimed_at = NULL WHERE id IN ({placeholders})"
                );
                let id_params: Vec<&dyn rusqlite::ToSql> = group
                    .ids
                    .iter()
                    .map(|s| s as &dyn rusqlite::ToSql)
                    .collect();
                conn.execute(&sql, id_params.as_slice())?;

                self.audit(
                    &conn,
                    "consolidate",
                    Some(&ep_id),
                    Some(&format!("{} items from {}", group.ids.len(), group.source)),
                );
                if let Some(pm) = self.plugins_if_active() {
                    pm.notify_consolidate(&serde_json::json!({
                        "summary": summary,
                        "source_wm_ids": group.ids,
                    }));
                }

                report.items_consolidated += group.ids.len();
                report.summaries_created += 1;
                if llm {
                    report.llm_used += 1;
                }
            }
            let method = if report.llm_used == report.summaries_created {
                "llm"
            } else if report.llm_used > 0 {
                "llm+aaak"
            } else {
                "aaak"
            };
            conn.execute(
                "INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview) \
                 VALUES (?1, ?2, ?3)",
                params![
                    self.config.session_id,
                    report.items_consolidated as i64,
                    format!(
                        "{} summaries ({method}) from {} items",
                        report.summaries_created, report.items_consolidated
                    ),
                ],
            )?;
        }

        let (t1, t2) = self.degrade_episodic()?;
        report.tier1_to_tier2 = t1;
        report.tier2_to_tier3 = t2;
        // Auto-resolve well-attested (subject, predicate) conflicts by confidence
        // (`veracity_consolidation.py` `run_consolidation_pass` L777 — defined but never called in
        // Python; wired into sleep here so contested facts actually converge).
        {
            let conn = self.store.conn.lock().unwrap();
            report.facts_auto_resolved = veracity::run_consolidation_pass(&conn)?;
        }
        Ok(report)
    }

    /// Run a full sleep pass with the deterministic AAAK summary (no LLM). Equivalent to
    /// [`Engine::sleep_plan`] + [`Engine::finish_sleep`] with no LLM summaries. The async provider
    /// uses the split form to inject LLM summaries; this is the standalone/no-LLM entrypoint.
    pub fn sleep(&self, force: bool) -> Result<SleepReport> {
        let groups = self.sleep_plan(force)?;
        // No LLM in this path: run heuristic conflict detection and invalidate all detected pairs
        // (the `LLM_CONFLICT_DETECTION_ENABLED == False` branch, `beam.py` L7727-L7731).
        let _ = self.resolve_sleep_conflicts(&groups);
        self.finish_sleep(&groups, &HashMap::new())
    }

    /// Heuristic embedding-cosine conflict detection over the claimed sleep groups
    /// (`beam.py` `_detect_conflicts` L3634, run per group before summarization L7705). For each
    /// in-group pair (rows are timestamp-ASC, so the first is older) all four heuristics must hold:
    /// timestamps `>= 1h` apart, cosine `> 0.88` over L2-normalized stored embeddings, `>= 2`
    /// overlapping significant tokens, and an edit-distance ratio `> 0.3` (not near-duplicates).
    /// Distinct from the ingest-time `(subject, predicate)` veracity path.
    pub fn heuristic_sleep_conflicts(&self, groups: &[SleepGroup]) -> Result<Vec<SleepConflict>> {
        let conn = self.store.conn.lock().unwrap();
        let mut out = Vec::new();
        for g in groups {
            if g.ids.len() < 2 {
                continue;
            }
            out.extend(self.detect_conflicts(&conn, &g.ids, &g.contents)?);
        }
        Ok(out)
    }

    /// Detect heuristic conflicts and invalidate every detected pair (older superseded by newer),
    /// returning the count resolved. Used by the no-LLM [`Engine::sleep`] path; the LLM-gated path
    /// in `tools::run_sleep` validates each pair before invalidating.
    pub fn resolve_sleep_conflicts(&self, groups: &[SleepGroup]) -> Result<usize> {
        let conflicts = self.heuristic_sleep_conflicts(groups)?;
        let mut resolved = 0;
        for c in &conflicts {
            if self.invalidate(&c.older_id, Some(&c.newer_id))? {
                resolved += 1;
            }
        }
        Ok(resolved)
    }

    fn detect_conflicts(
        &self,
        conn: &Connection,
        ids: &[String],
        contents: &[String],
    ) -> Result<Vec<SleepConflict>> {
        let n = ids.len();
        if n < 2 {
            return Ok(Vec::new());
        }
        // Fetch timestamps + embeddings for the claimed ids in one pass each.
        let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let id_params: Vec<Value> = ids.iter().map(|s| Value::Text(s.clone())).collect();

        let mut timestamps: HashMap<String, String> = HashMap::new();
        {
            let sql =
                format!("SELECT id, timestamp FROM working_memory WHERE id IN ({placeholders})");
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params_from_iter(id_params.iter().cloned()), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                ))
            })?;
            for row in rows.flatten() {
                timestamps.insert(row.0, row.1);
            }
        }

        let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        {
            let sql = format!(
                "SELECT memory_id, embedding_json FROM memory_embeddings WHERE memory_id IN ({placeholders})"
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params_from_iter(id_params.iter().cloned()), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?;
            for row in rows.flatten() {
                if let Ok(v) = serde_json::from_str::<Vec<f32>>(&row.1) {
                    embeddings.insert(row.0, v);
                }
            }
        }

        let mut conflicts = Vec::new();
        for i in 0..n {
            let a_id = &ids[i];
            let a_vec = match embeddings.get(a_id) {
                Some(v) => v,
                None => continue,
            };
            for j in (i + 1)..n {
                let b_id = &ids[j];
                let b_vec = match embeddings.get(b_id) {
                    Some(v) => v,
                    None => continue,
                };
                // Heuristic 1: timestamps >= 1 hour apart.
                let (ta, tb) = match (timestamps.get(a_id), timestamps.get(b_id)) {
                    (Some(ta), Some(tb)) => (ta, tb),
                    _ => continue,
                };
                let hours = match hours_between(ta, tb) {
                    Some(h) => h,
                    None => continue,
                };
                if hours < 1.0 {
                    continue;
                }
                // Heuristic 2: cosine similarity > threshold.
                if a_vec.len() != b_vec.len()
                    || a_vec.iter().all(|x| *x == 0.0)
                    || b_vec.iter().all(|x| *x == 0.0)
                {
                    continue;
                }
                if (daemon_core::cosine(a_vec, b_vec) as f64) <= 0.88 {
                    continue;
                }
                // Heuristic 3: >= 2 overlapping significant tokens.
                let tokens_a = significant_tokens(&contents[i]);
                let tokens_b = significant_tokens(&contents[j]);
                if tokens_a.intersection(&tokens_b).count() < 2 {
                    continue;
                }
                // Heuristic 4: not near-duplicates (edit-distance ratio > 0.3).
                if edit_dist_ratio(&contents[i], &contents[j]) <= 0.3 {
                    continue;
                }
                conflicts.push(SleepConflict {
                    older_id: a_id.clone(),
                    newer_id: b_id.clone(),
                    older_content: contents[i].clone(),
                    newer_content: contents[j].clone(),
                });
            }
        }
        Ok(conflicts)
    }

    /// Tiered episodic degradation (`beam.py` `degrade_episodic` L7241-L7366): tier 1 rows older
    /// than [`MnemosyneConfig::tier2_days`] are AAAK-compressed and promoted to tier 2; tier 2 rows
    /// older than [`MnemosyneConfig::tier3_days`] are signal-compressed to <=[`TIER3_MAX_CHARS`]
    /// and promoted to tier 3. When a
    /// row's content changes its stale dense embedding is invalidated (dropped + binary vector
    /// cleared) so recall doesn't score against text that no longer exists. Recall already applies a
    /// tier weight, so degraded rows score down automatically. Returns `(tier1->2, tier2->3)` counts.
    pub fn degrade_episodic(&self) -> Result<(usize, usize)> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();

        // Tier 1 -> 2: AAAK-compress.
        let tier1: Vec<(String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT id, content FROM episodic_memory \
                 WHERE tier = 1 AND created_at < datetime('now', ?1) \
                 ORDER BY created_at ASC LIMIT ?2",
            )?;
            let rows = stmt.query_map(
                params![
                    format!("-{} days", self.config.tier2_days),
                    DEGRADE_BATCH_SIZE as i64
                ],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let mut t1 = 0usize;
        for (id, content) in &tier1 {
            let compressed = crate::aaak::encode(content);
            let final_content: String = compressed.chars().take(800).collect();
            conn.execute(
                "UPDATE episodic_memory SET content = ?2, tier = 2, degraded_at = ?3 WHERE id = ?1",
                params![id, final_content, now],
            )?;
            if &final_content != content {
                self.invalidate_episodic_embedding(&conn, id)?;
            }
            t1 += 1;
        }

        // Tier 2 -> 3: signal-compress to TIER3_MAX_CHARS.
        let tier2: Vec<(String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT id, content FROM episodic_memory \
                 WHERE tier = 2 AND created_at < datetime('now', ?1) \
                 ORDER BY created_at ASC LIMIT ?2",
            )?;
            let rows = stmt.query_map(
                params![
                    format!("-{} days", self.config.tier3_days),
                    (DEGRADE_BATCH_SIZE / 2) as i64
                ],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let mut t2 = 0usize;
        for (id, content) in &tier2 {
            let compressed = compress_to(content, TIER3_MAX_CHARS);
            conn.execute(
                "UPDATE episodic_memory SET content = ?2, tier = 3, degraded_at = ?3 WHERE id = ?1",
                params![id, compressed, now],
            )?;
            if &compressed != content {
                self.invalidate_episodic_embedding(&conn, id)?;
            }
            t2 += 1;
        }
        Ok((t1, t2))
    }

    /// Drop a degraded row's stale dense embedding + binary vector (`beam.py`
    /// `_refresh_episodic_embedding` invalidation path C18.b) so recall falls back to lexical/FTS
    /// until a fresh embedding is computed.
    fn invalidate_episodic_embedding(&self, conn: &Connection, id: &str) -> Result<()> {
        conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
            params![id],
        )?;
        conn.execute(
            "UPDATE episodic_memory SET binary_vector = NULL WHERE id = ?1",
            params![id],
        )?;
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
    event_date: Option<String>,
    event_date_precision: String,
    temporal_tags: String,
}

/// Compress `content` to at most `max_chars` characters for tier-3 degradation (`beam.py`
/// `_extract_key_signal`/truncation L7344-L7349): keep the head and mark elision.
fn compress_to(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let head: String = content.chars().take(max_chars).collect();
    format!("{head} [...]")
}

/// Absolute hours between two ISO-8601 timestamps, or `None` if either fails to parse
/// (`beam.py` `_detect_conflicts` heuristic 1 L3718-L3726).
fn hours_between(a: &str, b: &str) -> Option<f64> {
    let ta = chrono::DateTime::parse_from_rfc3339(a).ok()?;
    let tb = chrono::DateTime::parse_from_rfc3339(b).ok()?;
    Some((tb - ta).num_seconds().abs() as f64 / 3600.0)
}

/// Significant content tokens for conflict overlap: `[A-Za-z]{3,}` lowercased, minus the
/// `_detect_conflicts` stop-word set (`beam.py` L3677-L3694).
fn significant_tokens(text: &str) -> HashSet<String> {
    const STOP: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "need", "this", "that", "these", "those", "it", "its", "they", "them", "their", "we", "us",
        "our", "you", "your", "he", "she", "him", "her", "his", "not", "no", "nor", "so", "if",
        "then", "than", "too", "very", "just", "about", "also", "more", "some", "any", "each",
        "every", "all", "both", "what", "when", "where", "why", "how", "which", "who", "whom",
        "get", "got", "make", "made", "take", "took", "use", "used", "like", "said", "says",
        "know", "knew", "think", "thinks", "thought", "see", "saw", "seen", "come", "came", "give",
        "gave", "tell", "told",
    ];
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"[A-Za-z]{3,}").unwrap());
    re.find_iter(&text.to_lowercase())
        .map(|m| m.as_str().to_string())
        .filter(|w| !STOP.contains(&w.as_str()))
        .collect()
}

/// Normalized edit distance in `[0, 1]` (0 = identical, 1 = completely different). Mirrors the
/// `_detect_conflicts` near-duplicate gate (`beam.py` L3696-L3703); Python uses
/// `difflib.SequenceMatcher.ratio()`, this port uses a normalized Levenshtein distance.
fn edit_dist_ratio(s1: &str, s2: &str) -> f64 {
    if s1.is_empty() && s2.is_empty() {
        return 0.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 1.0;
    }
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let dist = prev[b.len()];
    dist as f64 / a.len().max(b.len()) as f64
}
