// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Knowledge-layer surface for the BEAM [`Engine`]: graph query/link, temporal triples, canonical
//! identity facts, the per-candidate knowledge bonuses, and entity-seeded candidate injection.
//! Split out of `engine.rs` (W-MNEMO).

use super::*;
use crate::knowledge::{annotations, entities, episodic_graph};
use crate::recall::scoring;
use rusqlite::{params, Connection};
use std::collections::{HashMap, HashSet};

impl Engine {
    /// Graph neighbours of a memory within `depth` hops (`beam.py` `graph_query` /
    /// `episodic_graph::find_related_memories`).
    pub fn graph_query(
        &self,
        memory_id: &str,
        depth: usize,
    ) -> Result<Vec<episodic_graph::Related>> {
        let conn = self.store.conn.lock().unwrap();
        episodic_graph::find_related_memories(&conn, memory_id, depth.max(1), "", 0.0)
    }

    /// Add a manual graph edge between two memories (`beam.py` `graph_link` /
    /// `episodic_graph::add_edge`).
    pub fn graph_link(&self, link: &GraphLink) -> Result<()> {
        let conn = self.store.conn.lock().unwrap();
        episodic_graph::add_edge(
            &conn,
            &episodic_graph::GraphEdge {
                source: link.source.to_string(),
                target: link.target.to_string(),
                edge_type: if link.edge_type.is_empty() {
                    "related_to".to_string()
                } else {
                    link.edge_type.to_string()
                },
                weight: link.weight,
            },
        )
    }

    /// Add a temporal triple (`triples::add`).
    pub fn triple_add(&self, t: &TripleAdd) -> Result<i64> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::triples::add(
            &conn,
            t.subject,
            t.predicate,
            t.object,
            t.valid_from,
            t.valid_until,
            t.source,
            t.confidence,
            t.supersede,
        )
    }

    /// Expire open triples (`triples::end`).
    pub fn triple_end(&self, t: &TripleEnd) -> Result<usize> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::triples::end(&conn, t.subject, t.predicate, t.object, t.valid_until)
    }

    /// Query temporal triples valid at `as_of` (`triples::query`).
    pub fn triple_query(&self, q: &TripleQuery) -> Result<Vec<crate::knowledge::triples::Triple>> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::triples::query(&conn, q.subject, q.predicate, q.object, q.as_of)
    }

    /// Upsert a canonical identity fact (`canonical::remember`).
    pub fn canonical_remember(
        &self,
        c: &CanonicalRemember,
    ) -> Result<(
        crate::knowledge::canonical::CanonicalRow,
        crate::knowledge::canonical::Status,
    )> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::canonical::remember(
            &conn,
            c.owner_id,
            c.category,
            c.name,
            c.body,
            c.source,
            c.confidence,
        )
    }

    /// Read live canonical facts for an owner (`canonical::current`).
    pub fn canonical_recall(
        &self,
        owner_id: &str,
        category: Option<&str>,
        name: Option<&str>,
    ) -> Result<Vec<crate::knowledge::canonical::CanonicalRow>> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::canonical::current(&conn, owner_id, category, name)
    }

    /// Retire a canonical fact slot (`canonical::forget`).
    pub fn canonical_forget(&self, owner_id: &str, category: &str, name: &str) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        crate::knowledge::canonical::forget(&conn, owner_id, category, name)
    }

    /// Compute the knowledge-layer recall signals for a candidate keyed by `row_id`: the additive
    /// `graph_bonus` (incident `graph_edges`) and `fact_bonus` (query entities appearing in the
    /// row's `facts`), plus the entity (`*1.3`, capped) and fact (`*1.2`) post-multiplier flags
    /// (`beam.py` L5779-L5793). With no query entities all signals are inert.
    pub(crate) fn knowledge_bonuses(
        &self,
        conn: &Connection,
        row_id: &str,
        q_entities: &[String],
    ) -> Result<KnowledgeBonuses> {
        let edges = episodic_graph::edge_count(conn, row_id)?;
        let graph_bonus = scoring::graph_bonus(edges);
        if q_entities.is_empty() {
            return Ok(KnowledgeBonuses {
                graph_bonus,
                fact_bonus: 0.0,
                entity_match: false,
                fact_match: false,
            });
        }

        let mentions = annotations::query_by_memory(conn, row_id, Some("mentions"))?;
        let entity_match = q_entities.iter().any(|e| {
            mentions
                .iter()
                .any(|m| m.value.eq_ignore_ascii_case(e.as_str()))
        });

        let mut stmt =
            conn.prepare("SELECT subject, object FROM facts WHERE source_msg_id = ?1")?;
        let fact_terms: Vec<(String, String)> = stmt
            .query_map(params![row_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let fact_match_count = q_entities
            .iter()
            .filter(|e| {
                fact_terms.iter().any(|(s, o)| {
                    s.eq_ignore_ascii_case(e.as_str()) || o.eq_ignore_ascii_case(e.as_str())
                })
            })
            .count();

        Ok(KnowledgeBonuses {
            graph_bonus,
            fact_bonus: scoring::fact_bonus(fact_match_count),
            entity_match,
            fact_match: fact_match_count > 0,
        })
    }

    /// Inject working candidates that mention a query entity (or sit within two graph hops of one)
    /// but were missed by the lexical/FTS/vector gates (`beam.py` L5760-L5793). New candidates are
    /// scored with the entity-recall floor `(0.6 + 0.2*imp) * (0.7 + 0.3*decay) * veracity`.
    pub(crate) fn inject_entity_candidates(
        &self,
        conn: &Connection,
        ctx: &EntityInjectCtx,
    ) -> Result<Vec<MemoryRow>> {
        if ctx.q_entities.is_empty() {
            return Ok(Vec::new());
        }
        let mut seeds: HashSet<String> = HashSet::new();
        for entity in ctx.q_entities {
            for ann in annotations::query_by_kind(conn, "mentions", Some(entity), false)? {
                seeds.insert(ann.memory_id);
            }
        }

        // Fuzzy entity matching (`entities.py` `find_similar_entities`, threshold 0.8): also seed
        // memories that mention a *similar* entity ("Acme" vs "Acme Corp", typos), not just exact
        // string equality. The known-entity universe is the deduped `mentions` annotation values.
        let all_mentions = annotations::query_by_kind(conn, "mentions", None, true)?;
        if !all_mentions.is_empty() {
            let mut value_to_memories: HashMap<String, Vec<String>> = HashMap::new();
            for ann in &all_mentions {
                value_to_memories
                    .entry(ann.value.clone())
                    .or_default()
                    .push(ann.memory_id.clone());
            }
            let known: Vec<String> = value_to_memories.keys().cloned().collect();
            for entity in ctx.q_entities {
                for (matched, _score) in entities::find_similar_entities(entity, &known, 0.8) {
                    if matched.eq_ignore_ascii_case(entity) {
                        continue; // exact matches already seeded above
                    }
                    if let Some(ids) = value_to_memories.get(&matched) {
                        seeds.extend(ids.iter().cloned());
                    }
                }
            }
        }

        // One graph expansion (depth 2) from the directly-mentioning seeds.
        let mut expanded: HashSet<String> = HashSet::new();
        for seed in &seeds {
            for rel in episodic_graph::find_related_memories(conn, seed, 2, "", 0.0)? {
                expanded.insert(rel.memory_id);
            }
        }
        seeds.extend(expanded);

        let mut out = Vec::new();
        for id in seeds {
            if ctx.present.contains(&id) {
                continue;
            }
            if let Some(mut row) = self.fetch_working(conn, &id, ctx.scope)? {
                let decay = scoring::recency_decay(age_hours(&row.timestamp));
                let base = (0.6 + 0.2 * row.importance) * (0.7 + 0.3 * decay);
                row.score = base * scoring::veracity_multiplier(&row.veracity);
                out.push(row);
            }
        }
        Ok(out)
    }
}

/// The knowledge-layer recall signals for a single candidate.
pub(crate) struct KnowledgeBonuses {
    pub(crate) graph_bonus: f64,
    pub(crate) fact_bonus: f64,
    entity_match: bool,
    fact_match: bool,
}

impl KnowledgeBonuses {
    /// Apply the entity (`*1.3`, capped at 1.0) and fact (`*1.2`) multipliers to a base score
    /// (`beam.py` L5785-L5793).
    pub(crate) fn apply_multipliers(&self, base: f64) -> f64 {
        let mut s = base;
        if self.entity_match {
            s = (s * 1.3).min(1.0);
        }
        if self.fact_match {
            s *= 1.2;
        }
        s
    }
}
