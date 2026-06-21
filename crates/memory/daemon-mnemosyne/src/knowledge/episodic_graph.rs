//! EpisodicGraph — port of `episodic_graph.py`.
//!
//! Regex SPO fact extraction (`is`/`has`/`uses`/`works_at`, cap 5, pronoun-led subjects rejected,
//! `episodic_graph.py` L292-L372), `facts`/`graph_edges` writes, and a `find_related_memories(depth)`
//! BFS over `graph_edges` (`episodic_graph.py` L432-L484). Rule-based gists are deferred (P2).

use crate::error::Result;
use crate::util;
use regex::Regex;
use rusqlite::{params, Connection};
use std::collections::HashSet;
use std::sync::OnceLock;

/// Pronoun/demonstrative/possessive leaders that disqualify a fact subject
/// (`episodic_graph.py` `_LOW_QUALITY_SUBJECT_LEADERS`).
const LOW_QUALITY_SUBJECT_LEADERS: &[&str] = &[
    "this", "that", "these", "those", "it", "there", "here", "i", "you", "he", "she", "they", "we",
    "him", "her", "them", "us", "my", "your", "his", "their", "our", "its",
];

/// Cap input length before regex so backtracking can't stall ingest on pathological documents
/// (`episodic_graph.py` `_EXTRACT_FACTS_MAX_CONTENT_LEN` L290).
const EXTRACT_FACTS_MAX_CONTENT_LEN: usize = 4096;

/// Cap on facts emitted per memory (`episodic_graph.py` L372).
const MAX_FACTS_PER_MEMORY: usize = 5;

/// A graph edge (`graph_edges` table, `episodic_graph.py` L146-L155).
#[derive(Clone, Debug)]
pub struct GraphEdge {
    /// Source node id.
    pub source: String,
    /// Target node id.
    pub target: String,
    /// Edge type (`rel`, `ctx`, `syn`, `related_to`, `references`, ...).
    pub edge_type: String,
    /// Edge weight.
    pub weight: f64,
}

/// A structured fact triple extracted from a memory (`episodic_graph.py` `Fact` L70-L82).
#[derive(Clone, Debug)]
pub struct Fact {
    /// Deterministic per-memory id (`fact_<memory_id>_<n>`).
    pub id: String,
    /// Subject.
    pub subject: String,
    /// Predicate (`is`/`has`/`uses`/`works_at`).
    pub predicate: String,
    /// Object.
    pub object: String,
    /// ISO timestamp.
    pub timestamp: String,
    /// Extraction confidence.
    pub confidence: f64,
}

/// A neighbour discovered by [`find_related_memories`].
#[derive(Clone, Debug)]
pub struct Related {
    /// The related memory id.
    pub memory_id: String,
    /// The edge type that linked it.
    pub edge_type: String,
    /// The edge weight.
    pub weight: f64,
    /// The BFS hop distance (1-based).
    pub depth: usize,
}

/// True if `subject` is led by a pronoun/demonstrative/possessive (`_is_low_quality_subject`).
fn is_low_quality_subject(subject: &str) -> bool {
    let first = subject
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| ".,!?;:'\"".contains(c))
        .to_lowercase();
    LOW_QUALITY_SUBJECT_LEADERS.contains(&first.as_str())
}

struct FactPattern {
    re: Regex,
    predicate: &'static str,
    object_group: usize,
    confidence: f64,
}

/// The ordered SPO patterns (`episodic_graph.py` `extract_facts` L312-L370). `is`/`has`/`uses` take
/// a capitalized subject; `works at/for/with` requires a capitalized object too.
fn fact_patterns() -> &'static [FactPattern] {
    static PATTERNS: OnceLock<Vec<FactPattern>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            FactPattern {
                re: Regex::new(r"\b([A-Z][a-zA-Z\s]+?)\s+is\s+(?:a|an|the)?\s*([a-zA-Z\s]+?)\b")
                    .unwrap(),
                predicate: "is",
                object_group: 2,
                confidence: 0.7,
            },
            FactPattern {
                re: Regex::new(r"\b([A-Z][a-zA-Z\s]+?)\s+has\s+(?:a|an|the)?\s*([a-zA-Z\d\s]+?)\b")
                    .unwrap(),
                predicate: "has",
                object_group: 2,
                confidence: 0.6,
            },
            FactPattern {
                re: Regex::new(
                    r"\b([A-Z][a-zA-Z\s]+?)\s+(?:uses?|using|used)\s+(?:a|an|the)?\s*([a-zA-Z\s]+?)\b",
                )
                .unwrap(),
                predicate: "uses",
                object_group: 2,
                confidence: 0.6,
            },
            FactPattern {
                re: Regex::new(
                    r"\b([A-Z][a-zA-Z\s]+?)\s+works?\s+(?:at|for|with)\s+([A-Z][a-zA-Z\s]+?)\b",
                )
                .unwrap(),
                predicate: "works_at",
                object_group: 2,
                confidence: 0.7,
            },
        ]
    })
}

/// Extract structured SPO facts from `content` (`episodic_graph.py` `extract_facts` L292-L372).
/// Input is truncated to [`EXTRACT_FACTS_MAX_CONTENT_LEN`] and the result is capped at
/// [`MAX_FACTS_PER_MEMORY`].
pub fn extract_facts(content: &str, memory_id: &str) -> Vec<Fact> {
    let content = if content.len() > EXTRACT_FACTS_MAX_CONTENT_LEN {
        &content[..EXTRACT_FACTS_MAX_CONTENT_LEN]
    } else {
        content
    };
    let now = util::now_iso();
    let mut facts: Vec<Fact> = Vec::new();
    for pat in fact_patterns() {
        for caps in pat.re.captures_iter(content) {
            let subject = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let object = caps
                .get(pat.object_group)
                .map(|m| m.as_str().trim())
                .unwrap_or("");
            if subject.len() > 2 && object.len() > 2 && !is_low_quality_subject(subject) {
                facts.push(Fact {
                    id: format!("fact_{}_{}", memory_id, facts.len()),
                    subject: subject.to_string(),
                    predicate: pat.predicate.to_string(),
                    object: object.to_string(),
                    timestamp: now.clone(),
                    confidence: pat.confidence,
                });
                if facts.len() >= MAX_FACTS_PER_MEMORY {
                    return facts;
                }
            }
        }
    }
    facts
}

/// Persist a fact (`episodic_graph.py` `store_fact` L395-L412). `memory_id` is recorded as the
/// `source_msg_id`.
pub fn store_fact(conn: &Connection, fact: &Fact, memory_id: &str, session_id: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO facts \
         (fact_id, session_id, subject, predicate, object, timestamp, source_msg_id, confidence) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            fact.id,
            session_id,
            fact.subject,
            fact.predicate,
            fact.object,
            fact.timestamp,
            memory_id,
            fact.confidence,
        ],
    )?;
    Ok(())
}

/// Add a graph edge (`episodic_graph.py` `add_edge` L414-L428), stamping the current time.
pub fn add_edge(conn: &Connection, edge: &GraphEdge) -> Result<()> {
    conn.execute(
        "INSERT INTO graph_edges (source, target, edge_type, weight, timestamp) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            edge.source,
            edge.target,
            edge.edge_type,
            edge.weight,
            util::now_iso()
        ],
    )?;
    Ok(())
}

/// Count edges incident to `node_id` (used for the recall `graph_bonus`).
pub fn edge_count(conn: &Connection, node_id: &str) -> Result<usize> {
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM graph_edges WHERE source = ?1 OR target = ?1",
        params![node_id],
        |r| r.get(0),
    )?;
    Ok(n as usize)
}

/// BFS over `graph_edges` from `memory_id` out to `depth` hops (`episodic_graph.py`
/// `find_related_memories` L432-L484). `edge_type` empty = any type; `min_weight` filters edges.
pub fn find_related_memories(
    conn: &Connection,
    memory_id: &str,
    depth: usize,
    edge_type: &str,
    min_weight: f64,
) -> Result<Vec<Related>> {
    let mut results: Vec<Related> = Vec::new();
    let mut current: HashSet<String> = HashSet::from([memory_id.to_string()]);
    let mut seen: HashSet<String> = HashSet::from([memory_id.to_string()]);

    for hop in 1..=depth {
        let mut next: HashSet<String> = HashSet::new();
        for mem in &current {
            let map = |r: &rusqlite::Row<'_>| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, f64>(3)?,
                ))
            };
            let edges: Vec<(String, String, String, f64)> = if edge_type.is_empty() {
                let mut stmt = conn.prepare(
                    "SELECT source, target, edge_type, weight FROM graph_edges \
                     WHERE (source = ?1 OR target = ?1) AND weight >= ?2",
                )?;
                let rows = stmt
                    .query_map(params![mem, min_weight], map)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                rows
            } else {
                let mut stmt = conn.prepare(
                    "SELECT source, target, edge_type, weight FROM graph_edges \
                     WHERE (source = ?1 OR target = ?1) AND edge_type = ?2 AND weight >= ?3",
                )?;
                let rows = stmt
                    .query_map(params![mem, edge_type, min_weight], map)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                rows
            };
            for (source, target, etype, weight) in edges {
                let neighbour = if source == *mem { target } else { source };
                if seen.insert(neighbour.clone()) {
                    next.insert(neighbour.clone());
                    results.push(Related {
                        memory_id: neighbour,
                        edge_type: etype,
                        weight,
                        depth: hop,
                    });
                }
            }
        }
        current = next;
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_spo_and_rejects_pronoun_subjects() {
        let facts = extract_facts("Maya works at Acme. It is a thing.", "m1");
        assert!(
            facts
                .iter()
                .any(|f| f.subject == "Maya" && f.predicate == "works_at" && f.object == "Acme"),
            "{facts:?}"
        );
        // "It is a thing" -> pronoun-led subject rejected.
        assert!(!facts.iter().any(|f| f.subject.eq_ignore_ascii_case("it")));
    }

    #[test]
    fn caps_at_five_facts() {
        let content = "Alpha is good. Beta is good. Gamma is good. Delta is good. \
                       Epsilon is good. Zeta is good. Eta is good.";
        let facts = extract_facts(content, "m1");
        assert!(facts.len() <= MAX_FACTS_PER_MEMORY, "{}", facts.len());
    }

}
