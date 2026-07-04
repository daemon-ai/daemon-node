// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Knowledge-layer surface for the BEAM [`Engine`]: graph query/link, temporal triples, and
//! canonical identity facts. Split out of `engine.rs` (W-MNEMO). The recall-time knowledge
//! signals (graph/fact bonuses, entity/fact-aware boosts) live in `engine/recall.rs`, mirroring
//! `beam.py`'s recall body.

use super::*;
use crate::knowledge::episodic_graph;

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
}
