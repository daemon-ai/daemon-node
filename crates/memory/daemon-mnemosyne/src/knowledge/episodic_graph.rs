// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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

/// A time-aware episode summary (`episodic_graph.py` `Gist` L53-L62).
#[derive(Clone, Debug)]
pub struct Gist {
    /// Deterministic id (`gist_<memory_id>`).
    pub id: String,
    /// The concise episode summary (first sentence or first 100 chars).
    pub text: String,
    /// ISO timestamp.
    pub timestamp: String,
    /// Participant names + pronouns (capped at 5).
    pub participants: Vec<String>,
    /// Location reference, if any.
    pub location: Option<String>,
    /// Emotion class (`positive`/`negative`/`neutral`), if any.
    pub emotion: Option<String>,
    /// Temporal scope (`point_in_time`/`duration`/`range`), if any.
    pub time_scope: Option<String>,
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

/// Rule-based gist extraction (`episodic_graph.py` `extract_gist` L165-L275): participants
/// (capitalized names + pronouns, cap 5), temporal scope, location, emotion, and a first-sentence
/// summary. Zero LLM.
pub fn extract_gist(content: &str, memory_id: &str) -> Gist {
    Gist {
        id: format!("gist_{memory_id}"),
        text: create_summary(content),
        timestamp: util::now_iso(),
        participants: extract_participants(content),
        location: extract_location(content),
        emotion: extract_emotion(content),
        time_scope: extract_temporal_scope(content),
    }
}

struct GistPatterns {
    name: Regex,
    pronoun: Regex,
    temporal: Vec<(Regex, &'static str)>,
    location: Vec<Regex>,
}

fn gist_patterns() -> &'static GistPatterns {
    static P: OnceLock<GistPatterns> = OnceLock::new();
    P.get_or_init(|| GistPatterns {
        name: Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b").unwrap(),
        pronoun: Regex::new(r"(?i)\b(I|you|we|they|he|she|it|me|us|them|him|her)\b").unwrap(),
        temporal: vec![
            (
                Regex::new(r"(?i)\b(yesterday|today|tomorrow|now|soon|later|earlier)\b").unwrap(),
                "point_in_time",
            ),
            (
                Regex::new(r"(?i)\b(last\s+week|last\s+month|last\s+year|next\s+week)\b").unwrap(),
                "point_in_time",
            ),
            (
                Regex::new(r"(?i)\b(since|from|starting)\b.*\b(until|to|through|end)\b").unwrap(),
                "duration",
            ),
            (Regex::new(r"(?i)\b(between|from)\b.*\b(and|to)\b").unwrap(), "range"),
            (Regex::new(r"(?i)\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)?\b").unwrap(), "point_in_time"),
            (Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap(), "point_in_time"),
        ],
        location: vec![
            Regex::new(
                r"(?i)\b(at|in|from)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:yesterday|today|tomorrow|now|last|next|on|at)\b|$)",
            )
            .unwrap(),
            Regex::new(r"(?i)\b(office|home|work|school|hospital|store|restaurant|building|room)\b")
                .unwrap(),
        ],
    })
}

/// Participant names + pronouns, deduped (order-preserving) and capped at 5 (`_extract_participants`
/// L209-L221).
fn extract_participants(content: &str) -> Vec<String> {
    let p = gist_patterns();
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let push = |s: String, seen: &mut HashSet<String>, out: &mut Vec<String>| {
        if seen.insert(s.clone()) {
            out.push(s);
        }
    };
    for cap in p.name.captures_iter(content) {
        push(cap[1].to_string(), &mut seen, &mut out);
    }
    for cap in p.pronoun.captures_iter(content) {
        push(cap[1].to_string(), &mut seen, &mut out);
    }
    out.truncate(5);
    out
}

/// Temporal scope class, or `None` (`_extract_temporal_scope` L223-L238).
fn extract_temporal_scope(content: &str) -> Option<String> {
    for (re, scope) in &gist_patterns().temporal {
        if re.is_match(content) {
            return Some((*scope).to_string());
        }
    }
    None
}

/// Location reference, or `None` (`_extract_location` L240-L252): the prepositional-phrase pattern
/// yields its capitalized place (group 2), the keyword pattern yields the matched word (group 1).
fn extract_location(content: &str) -> Option<String> {
    let pats = &gist_patterns().location;
    if let Some(c) = pats[0].captures(content) {
        if let Some(m) = c.get(2) {
            return Some(m.as_str().trim().to_string());
        }
    }
    if let Some(c) = pats[1].captures(content) {
        if let Some(m) = c.get(1) {
            return Some(m.as_str().to_string());
        }
    }
    None
}

/// Emotion class, or `None` (`_extract_emotion` L254-L267). Order: positive, negative, neutral.
fn extract_emotion(content: &str) -> Option<String> {
    const POSITIVE: &[&str] = &[
        "happy", "excited", "great", "awesome", "love", "enjoy", "glad", "pleased",
    ];
    const NEGATIVE: &[&str] = &[
        "sad",
        "angry",
        "frustrated",
        "upset",
        "hate",
        "disappointed",
        "worried",
    ];
    const NEUTRAL: &[&str] = &["fine", "okay", "alright", "normal", "standard"];
    let lower = content.to_lowercase();
    for (class, words) in [
        ("positive", POSITIVE),
        ("negative", NEGATIVE),
        ("neutral", NEUTRAL),
    ] {
        if words.iter().any(|w| lower.contains(w)) {
            return Some(class.to_string());
        }
    }
    None
}

/// First-sentence (or first-100-char) summary (`_create_summary` L269-L275).
fn create_summary(content: &str) -> String {
    let first = content.split(['.', '!', '?']).next().unwrap_or("");
    if first.chars().count() > 10 {
        return first.trim().chars().take(100).collect();
    }
    content
        .chars()
        .take(100)
        .collect::<String>()
        .trim()
        .to_string()
}

/// Store a gist (`episodic_graph.py` `store_gist` L376-L393).
pub fn store_gist(conn: &Connection, gist: &Gist, memory_id: &str) -> Result<()> {
    let participants_json =
        serde_json::to_string(&gist.participants).unwrap_or_else(|_| "[]".into());
    conn.execute(
        "INSERT OR REPLACE INTO gists \
         (id, text, timestamp, participants_json, location, emotion, time_scope, memory_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            gist.id,
            gist.text,
            gist.timestamp,
            participants_json,
            gist.location,
            gist.emotion,
            gist.time_scope,
            memory_id,
        ],
    )?;
    Ok(())
}

/// `(memory_id, gist_text)` for gists whose `participants_json` contains `participant`
/// (`episodic_graph.py` `find_gists_by_participant` L508-L529). Used by the polyphonic graph voice.
pub fn find_gists_by_participant(
    conn: &Connection,
    participant: &str,
) -> Result<Vec<(String, String)>> {
    let like = format!("%\"{participant}\"%");
    let mut stmt = conn.prepare(
        "SELECT memory_id, text FROM gists \
         WHERE participants_json LIKE ?1 ORDER BY timestamp DESC",
    )?;
    let rows = stmt.query_map(params![like], |r| {
        Ok((
            r.get::<_, Option<String>>(0)?.unwrap_or_default(),
            r.get::<_, String>(1)?,
        ))
    })?;
    Ok(rows.flatten().filter(|(mid, _)| !mid.is_empty()).collect())
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
    fn extract_gist_pulls_participants_emotion_and_summary() {
        let g = extract_gist(
            "Alice was happy at the Office yesterday. We celebrated.",
            "m1",
        );
        assert_eq!(g.id, "gist_m1");
        assert!(g.participants.iter().any(|p| p == "Alice"));
        assert_eq!(g.emotion.as_deref(), Some("positive"));
        assert_eq!(g.time_scope.as_deref(), Some("point_in_time"));
        assert!(g.text.starts_with("Alice was happy"));
        assert!(g.participants.len() <= 5);
    }

    #[test]
    fn caps_at_five_facts() {
        let content = "Alpha is good. Beta is good. Gamma is good. Delta is good. \
                       Epsilon is good. Zeta is good. Eta is good.";
        let facts = extract_facts(content, "m1");
        assert!(facts.len() <= MAX_FACTS_PER_MEMORY, "{}", facts.len());
    }

    // parity: test_e2_remember_batch_enrichment.py::TestReviewHardening::test_extract_facts_caps_long_content (tests/test_e2_remember_batch_enrichment.py:469)
    #[test]
    fn extract_facts_truncates_pathological_long_content() {
        // ~12KB of pattern-rich content: the input is truncated to the 4096-char window and the
        // result stays capped, so adversarial long inputs can't drive regex backtracking over
        // the full text (the batch-ingest hardening).
        let long_content = "Anna is a developer. ".repeat(600);
        assert!(
            long_content.len() > EXTRACT_FACTS_MAX_CONTENT_LEN,
            "test setup: content must exceed the truncation window"
        );
        let facts = extract_facts(&long_content, "m1");
        assert!(
            !facts.is_empty(),
            "the pattern still extracts within the window"
        );
        assert!(facts.len() <= MAX_FACTS_PER_MEMORY, "{}", facts.len());
    }
}
