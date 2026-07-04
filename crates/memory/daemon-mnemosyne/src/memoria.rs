// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! MEMORIA — structured-fact extraction + retrieval (`beam.py` Phase 1/2).
//!
//! Always-on, zero-LLM regex extraction at write time populates the `memoria_*` specialist tables
//! (`extract_and_store`, port of `beam.py` `extract_and_store_facts` L4256), and a keyword router
//! ([`memoria_retrieve`], port of `beam.py` `memoria_retrieve` L4566) answers questions from those
//! tables. The recall finalizer ([`crate::engine`]) folds a high-relevance MEMORIA hit in as an
//! extra candidate (`beam.py` L6006-L6059).
//!
//! This is an English-language port: it uses the `MULTILINGUAL_PATTERNS['en']` rules
//! (`beam.py` L4126-L4147). Language detection and the de/ru/it/es pattern banks are out of scope.
//! Fact versioning is ported: when a `(session, key, fact_type)` slot gets a new value, the old
//! row is closed (`valid_to_msg_idx`) and the new row carries `version_id + 1` with a
//! `previous_value` pointer (`beam.py` `_insert_fact` L4477), and the fact retriever renders the
//! evolution chain (`beam.py` L4787-L4814).

use crate::error::Result;
use regex::Regex;
use rusqlite::{params, params_from_iter, types::Value, Connection, OptionalExtension};
use std::sync::OnceLock;

/// A MEMORIA retrieval hit (`beam.py` `memoria_retrieve` return dict). `source == "fallback"` means
/// nothing matched.
#[derive(Debug, Clone)]
pub struct MemoriaResult {
    /// The rendered context block fed to the recall supplement.
    pub context: String,
    /// The specialist table the hit came from (`memoria_facts`, `memoria_timelines`, ...).
    pub source: String,
    /// The originating `working_memory` ids, for the `memoria_source` supplement rows.
    pub source_memory_ids: Vec<String>,
}

// ── Extraction regexes (`MULTILINGUAL_PATTERNS['en']`) ───────────────────────────────────────────

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("valid memoria regex")
}

macro_rules! lazy_re {
    ($name:ident, $pat:expr) => {
        fn $name() -> &'static Regex {
            static R: OnceLock<Regex> = OnceLock::new();
            R.get_or_init(|| re($pat))
        }
    };
}

lazy_re!(
    metric_re,
    r"(?i)(\d+(?:[.,]\d+)?)\s*(ms|sec|seconds?|minutes?|hours?|days?|weeks?|months?|%|KB|MB|GB|TB|rows?|columns?|roles?|features?|bugs?|commits?|cards?|users?|items?|tests?|APIs?|endpoints?|sprints?|tickets?)"
);
lazy_re!(iso_date_re, r"\b(\d{4}-\d{2}-\d{2})\b");
lazy_re!(
    named_months_re,
    r"(?i)((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s*(?:\d{4})?)"
);
lazy_re!(
    version_a_re,
    r"([A-Z][a-zA-Z]+(?:\s*[A-Z][a-zA-Z]+)*)\s+v?(\d+\.\d+(?:\.\d+)?)"
);
lazy_re!(
    version_b_re,
    r"(?i)([A-Z][a-zA-Z]+)\s+version\s+v?(\d+\.\d+(?:\.\d+)?)"
);
lazy_re!(
    negation_re,
    r"(?i)(I(?: have|'ve)?\s*(?:never|not)\s+[^.,;!?\n]{15,120})"
);
lazy_re!(
    decision_re,
    r"(?i)(?:decided to|chose to|opted for|selected|picked|switching to)\s+([^.,;!?\n]{10,120})"
);
lazy_re!(
    entity_re,
    r"(?i)(?:the|my|our|your)\s+([a-z_]+(?:\s+(?:table|model|schema|API|endpoint|function|module|route|handler|tool|plugin|script|config|setting|workflow|pipeline|process|system|server|client|service|database|query|file|repo|branch|PR|issue|task|job)))\s+(?:needs?|requires?|should|could|would|will|has|have|uses?|runs?|handles?|processes?|supports?)\s+([^.,;!?\n]{10,80})"
);
lazy_re!(
    sequence_re,
    r"(?i)((?:first|second|third|fourth|fifth|finally|next|then|after that)[^.,;!?\n]{15,120})"
);
// `should` is split out because its Python rule uses a lookahead the `regex` crate cannot express.
lazy_re!(
    instruction_main_re,
    r"(?i)(?:always|never|must not|must|need(?:s)? to(?: not)?|required to|prefer(?: not)? to|want to(?: avoid| ensure| use| keep))\s+([^.,;!?\n]{10,200})"
);
lazy_re!(
    instruction_should_re,
    r"(?i)should(?: not)?\s+([^.,;!?\n]{10,200})"
);
lazy_re!(
    instruction_should_tail_re,
    r"(?i)^(?:you|we|i|one)\s+(?:always|never|remember|use|keep|avoid|ensure|check|verify|run|test|build|deploy|push|pull|merge|commit|close|open|update|install|configure|set|enable|disable|add|remove|create|delete|start|stop|restart|reload|reset|try|implement|write|read|switch|move|copy|rename|send|reply|respond)\b"
);
lazy_re!(
    preference_fp_re,
    r"(?i)(?:I|You)(?: |')?(?:like|love|prefer|hate|dislike|enjoy|use|stick with|switched to|moved to|changed to|want|need|tend to|usually|would rather|don't like|don't want|not a fan of|am okay with|am comfortable with|am used to|am happy with|am tired of|am sick of|prefer not to|try to avoid|find it easier to|find it better to|find it useful to)\s+([^.,;!?\n]{10,200})"
);
lazy_re!(
    preference_3p_re,
    r"(?i)(?:Nathan|Bob|User|Amy|Zander|Zella)\s+(?:likes?|loves?|prefers?|hates?|dislikes?|enjoys?|uses?|wants?|needs?|tends\s+to|switches?\s+to|changes?\s+to|moves?\s+to)\s+([^.,;!?\n]{10,200})"
);
lazy_re!(
    preference_struct_re,
    r"(?im)(?:^|\n|-\s|—\s)((?:Prefers|Likes|Loves|Hates|Dislikes|Wants|Needs|Tends to|Enjoys|Uses)\s+([^.,;!?\n]{10,200}))"
);

const EVENT_KEYWORDS: &[&str] = &[
    "meeting",
    "call",
    "scheduled",
    "happened",
    "occurred",
    "plan to",
    "will be on",
    "due on",
    "release",
    "deadline",
    "launched",
    "deployed",
    "released",
    "published",
    "posted",
    "started",
    "began",
    "finished",
    "completed",
    "ended",
    "event",
    "conference",
    "workshop",
    "appointment",
];
const TRANSIENT_KEYWORDS: &[&str] = &[
    "forecast",
    "weather",
    "temperature",
    "rain",
    "snow",
    "wind",
    "humidity",
    "chance",
    "regenrisiko",
    "zum",
    "heute",
    "morgen",
    "gestern",
    "today",
    "tomorrow",
    "yesterday",
    "week",
    "month",
];
const METRIC_CTX_STOP: &[&str] = &[
    "the", "and", "for", "was", "of", "to", "a", "an", "in", "on", "at", "by", "is", "are", "has",
    "had", "not", "but", "or",
];
const INSTRUCTION_FALSE_POSITIVES: &[&str] = &[
    "i think you should leave",
    "should behave",
    "their work style",
];
const FACT_MATCH_STOPWORDS: &[&str] = &[
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "related",
    "should",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "totally",
    "unrelated",
    "use",
    "uses",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "again",
    "into",
    "not",
    "please",
    "somewhere",
    "supposed",
    "them",
    "then",
    "they",
    "whatever",
];

// ── Write path ───────────────────────────────────────────────────────────────────────────────

/// Extract structured MEMORIA facts from `content` and store them in the specialist tables, keyed
/// to `session_id` and `source_memory_id` (`beam.py` `extract_and_store_facts` L4256-L4475).
/// Best-effort: callers swallow errors so extraction never blocks a write.
pub fn extract_and_store(
    conn: &Connection,
    session_id: &str,
    content: &str,
    message_idx: i64,
    source_memory_id: &str,
) -> Result<()> {
    extract_metrics(conn, session_id, content, message_idx, source_memory_id)?;
    extract_dates(conn, session_id, content, message_idx, source_memory_id)?;
    extract_versions(conn, session_id, content, message_idx, source_memory_id)?;
    extract_negations(conn, session_id, content, message_idx, source_memory_id)?;
    extract_decisions(conn, session_id, content, message_idx, source_memory_id)?;
    extract_entities(conn, session_id, content, message_idx, source_memory_id)?;
    extract_sequences(conn, session_id, content, message_idx, source_memory_id)?;
    extract_instructions(conn, session_id, content, message_idx, source_memory_id)?;
    extract_preferences(conn, session_id, content, message_idx, source_memory_id)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn insert_fact(
    conn: &Connection,
    session: &str,
    msg_idx: i64,
    ftype: &str,
    key: &str,
    value: &str,
    ctx: &str,
    importance: f64,
    source_memory_id: &str,
) -> Result<()> {
    let plain_insert = |conn: &Connection| -> Result<()> {
        conn.execute(
            "INSERT INTO memoria_facts \
             (session_id, message_idx, fact_type, key, value, context_snippet, importance, \
              timestamp, valid_from_msg_idx, source_memory_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                session,
                msg_idx,
                ftype,
                key,
                value,
                ctx,
                importance,
                crate::util::now_iso(),
                msg_idx,
                source_memory_id
            ],
        )?;
        Ok(())
    };

    // Dates all share generic keys ("iso_date", "named_date"): versioning would create false
    // evolution chains when different events happen on different dates (`beam.py` L4480-L4489).
    if ftype == "date" {
        return plain_insert(conn);
    }

    // Version the (session, key, fact_type) slot: close the current row and chain the new value
    // to it (`beam.py` `_insert_fact` L4491-L4517).
    let existing: Option<(i64, String)> = conn
        .query_row(
            "SELECT id, value FROM memoria_facts \
             WHERE session_id = ?1 AND key = ?2 AND fact_type = ?3 AND valid_to_msg_idx IS NULL \
             ORDER BY version_id DESC LIMIT 1",
            params![session, key, ftype],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    match existing {
        Some((old_id, old_value)) if old_value != value => {
            conn.execute(
                "UPDATE memoria_facts SET valid_to_msg_idx = ?1, previous_value = value \
                 WHERE id = ?2",
                params![msg_idx, old_id],
            )?;
            let prev_version: i64 = conn
                .query_row(
                    "SELECT version_id FROM memoria_facts WHERE id = ?1",
                    params![old_id],
                    |r| r.get(0),
                )
                .optional()?
                .unwrap_or(0);
            conn.execute(
                "INSERT INTO memoria_facts \
                 (session_id, message_idx, fact_type, key, value, context_snippet, importance, \
                  timestamp, version_id, previous_value, updated_msg_idx, valid_from_msg_idx, \
                  source_memory_id) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    session,
                    msg_idx,
                    ftype,
                    key,
                    value,
                    ctx,
                    importance,
                    crate::util::now_iso(),
                    prev_version + 1,
                    old_value,
                    msg_idx,
                    msg_idx,
                    source_memory_id
                ],
            )?;
            Ok(())
        }
        _ => plain_insert(conn),
    }
}

fn insert_timeline(
    conn: &Connection,
    session: &str,
    date: &str,
    msg_idx: i64,
    desc: &str,
    source: &str,
    source_memory_id: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO memoria_timelines (session_id, date, message_idx, description, source, source_memory_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![session, date, msg_idx, desc, source, source_memory_id],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn insert_kg(
    conn: &Connection,
    session: &str,
    subject: &str,
    predicate: &str,
    obj: &str,
    msg_idx: i64,
    confidence: f64,
    source_memory_id: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO memoria_kg (session_id, subject, predicate, object, message_idx, confidence, source_memory_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![session, subject, predicate, obj, msg_idx, confidence, source_memory_id],
    )?;
    Ok(())
}

fn extract_metrics(
    conn: &Connection,
    session: &str,
    content: &str,
    msg_idx: i64,
    src: &str,
) -> Result<()> {
    let md_inline = re(r"`[^`]*`");
    let md_bold = re(r"\*\*[^*]+\*\*");
    let md_italic = re(r"[*_]{1,2}[^*_\n]+[*_]{1,2}");
    let md_ops = re(r"[=<>|&]");
    let css = re(r"^(pt-|lg:|pr-|pl-|pb-|px-|py-|mt-|mr-|mb-|ml-|mx-|my-)");
    let stray = re(r"^[`*\]]");
    let emphasis = re(r"[*_]{2,}");
    let code_ops = re(r"[`=<>|]");

    for cap in metric_re().captures_iter(content).take(10) {
        let m = cap.get(0).unwrap();
        let num = cap.get(1).unwrap().as_str();
        let unit = cap.get(2).unwrap().as_str();
        let mut unit_clean = unit.to_lowercase();
        if unit_clean.ends_with('s') && !unit_clean.ends_with("ms") {
            unit_clean.pop();
        }
        let pre_start = floor_boundary(content, m.start().saturating_sub(50));
        let pre_text = &content[pre_start..m.start()];
        if TRANSIENT_KEYWORDS
            .iter()
            .any(|kw| pre_text.to_lowercase().contains(kw))
        {
            continue;
        }
        let clean_pre = md_inline.replace_all(pre_text, " ");
        let clean_pre = md_bold.replace_all(&clean_pre, " ");
        let clean_pre = md_italic.replace_all(&clean_pre, " ");
        let clean_pre = md_ops.replace_all(&clean_pre, " ").into_owned();
        let trim_set: &[char] = &[
            '.', ',', ':', ';', '!', '?', '(', ')', '[', ']', '"', '\'', '`', '*', '_',
        ];
        let ctx_words: Vec<String> = clean_pre
            .split_whitespace()
            .map(|w| w.trim_matches(trim_set))
            .filter(|w| {
                w.len() > 2
                    && !METRIC_CTX_STOP.contains(&w.to_lowercase().as_str())
                    && !css.is_match(w)
                    && !stray.is_match(w)
            })
            .map(|w| w.to_lowercase())
            .collect();
        let ctx_words: Vec<String> = ctx_words.iter().rev().take(3).rev().cloned().collect();
        let prefix = ctx_words.join("_");
        let mut key = if prefix.is_empty() {
            unit_clean.clone()
        } else {
            format!("{prefix}_{unit_clean}")
        };
        if key.contains('`') || key.contains("**") || emphasis.is_match(&key) {
            continue;
        }
        if code_ops.find_iter(&key).count() > 2 {
            continue;
        }
        if unit_clean == "%" {
            let nonalpha = re(r"[^a-zA-Z0-9\s]").find_iter(&clean_pre).count();
            let words = clean_pre.split_whitespace().count();
            if words > 0 && nonalpha as f64 / words as f64 > 0.6 {
                continue;
            }
        }
        let val = format!("{num}{unit}");
        if unit_clean == "%" {
            key = key.replace("_%", "_pct");
            if !key.ends_with("_pct") {
                key = if prefix.is_empty() {
                    "pct".to_string()
                } else {
                    format!("{prefix}_pct")
                };
            }
        }
        insert_fact(
            conn,
            session,
            msg_idx,
            "metric",
            &key,
            &val,
            &context_snippet(content, m.start(), 60),
            0.65,
            src,
        )?;
    }
    Ok(())
}

fn extract_dates(
    conn: &Connection,
    session: &str,
    content: &str,
    msg_idx: i64,
    src: &str,
) -> Result<()> {
    for cap in iso_date_re().captures_iter(content) {
        let m = cap.get(1).unwrap();
        let dt = m.as_str();
        let ctx = context_snippet(content, m.start(), 100);
        let ctx_lower = ctx.to_lowercase();
        let has_event = EVENT_KEYWORDS.iter().any(|kw| ctx_lower.contains(kw));
        if !has_event {
            insert_fact(
                conn, session, msg_idx, "date", "iso_date", dt, &ctx, 0.5, src,
            )?;
        } else {
            insert_fact(
                conn, session, msg_idx, "date", "iso_date", dt, &ctx, 0.7, src,
            )?;
            let desc: String = ctx.chars().take(120).collect();
            insert_timeline(conn, session, dt, msg_idx, &desc, "iso_date", src)?;
        }
    }
    for cap in named_months_re().captures_iter(content) {
        let m = cap.get(1).unwrap();
        let dt = m.as_str().trim();
        let ctx = context_snippet(content, m.start(), 60);
        insert_fact(
            conn,
            session,
            msg_idx,
            "date",
            "named_date",
            dt,
            &ctx,
            0.7,
            src,
        )?;
    }
    Ok(())
}

fn extract_versions(
    conn: &Connection,
    session: &str,
    content: &str,
    msg_idx: i64,
    src: &str,
) -> Result<()> {
    for cap in version_a_re().captures_iter(content) {
        let m = cap.get(0).unwrap();
        let name = cap.get(1).unwrap().as_str().trim();
        let ver = cap.get(2).unwrap().as_str();
        let key = format!("{}_version", name.to_lowercase().replace(' ', "_"));
        insert_fact(
            conn,
            session,
            msg_idx,
            "version",
            &key,
            ver,
            &context_snippet(content, m.start(), 60),
            0.7,
            src,
        )?;
    }
    let mut seen: Vec<String> = Vec::new();
    for cap in version_b_re().captures_iter(content) {
        let m = cap.get(0).unwrap();
        let name = cap.get(1).unwrap().as_str().trim();
        let ver = cap.get(2).unwrap().as_str();
        if matches!(
            name.to_lowercase().as_str(),
            "running" | "using" | "installed" | "upgraded" | "currently"
        ) {
            continue;
        }
        let key = format!("{}_version", name.to_lowercase().replace(' ', "_"));
        if !seen.contains(&ver.to_string()) {
            seen.push(ver.to_string());
            insert_fact(
                conn,
                session,
                msg_idx,
                "version",
                &key,
                ver,
                &context_snippet(content, m.start(), 60),
                0.7,
                src,
            )?;
        }
    }
    Ok(())
}

fn extract_negations(
    conn: &Connection,
    session: &str,
    content: &str,
    msg_idx: i64,
    src: &str,
) -> Result<()> {
    for cap in negation_re().captures_iter(content) {
        let neg_text = cap.get(1).unwrap().as_str().trim();
        let neg_lower = neg_text.to_lowercase();
        let mut obj = neg_text.to_string();
        for sw in ["never", "not"] {
            if neg_lower.contains(sw) {
                if let Some(idx) = neg_lower.find(sw) {
                    obj = neg_text[idx + sw.len()..].trim().to_string();
                    break;
                }
            }
        }
        let obj: String = obj.chars().take(80).collect();
        insert_kg(conn, session, "user", "negation", &obj, msg_idx, 0.75, src)?;
    }
    Ok(())
}

fn extract_decisions(
    conn: &Connection,
    session: &str,
    content: &str,
    msg_idx: i64,
    src: &str,
) -> Result<()> {
    for cap in decision_re().captures_iter(content) {
        let decision = cap.get(1).unwrap().as_str().trim();
        insert_kg(
            conn, session, "user", "decision", decision, msg_idx, 0.65, src,
        )?;
    }
    Ok(())
}

fn extract_entities(
    conn: &Connection,
    session: &str,
    content: &str,
    msg_idx: i64,
    src: &str,
) -> Result<()> {
    for cap in entity_re().captures_iter(content) {
        let entity = cap.get(1).unwrap().as_str().trim();
        let action = cap.get(2).unwrap().as_str().trim();
        insert_kg(
            conn, session, entity, "requires", action, msg_idx, 0.65, src,
        )?;
    }
    Ok(())
}

fn extract_sequences(
    conn: &Connection,
    session: &str,
    content: &str,
    msg_idx: i64,
    src: &str,
) -> Result<()> {
    for cap in sequence_re().captures_iter(content) {
        let m = cap.get(0).unwrap();
        let seq = cap.get(1).unwrap().as_str().trim();
        let first_word = seq.split_whitespace().next().unwrap_or("").to_lowercase();
        let val: String = seq.chars().take(120).collect();
        insert_fact(
            conn,
            session,
            msg_idx,
            "sequence",
            &first_word,
            &val,
            &context_snippet(content, m.start(), 60),
            0.6,
            src,
        )?;
    }
    Ok(())
}

fn extract_instructions(
    conn: &Connection,
    session: &str,
    content: &str,
    msg_idx: i64,
    src: &str,
) -> Result<()> {
    let bare_q = re(
        r"(?i)^(?:should|sollte|dovrebbe|dovresti)\s+(?:i|we|it|they|he|she|the|ich|wir|es|man|der|die|das|io|noi|lui|lei|loro)\b",
    );
    let store = |m_start: usize, instr: &str| -> Result<()> {
        let instr = instr.trim();
        let instr_lower = instr.to_lowercase();
        if INSTRUCTION_FALSE_POSITIVES
            .iter()
            .any(|fp| instr_lower.contains(fp))
        {
            return Ok(());
        }
        if bare_q.is_match(instr) {
            return Ok(());
        }
        let instr_trunc: String = instr.chars().take(200).collect();
        conn.execute(
            "INSERT INTO memoria_instructions (session_id, message_idx, instruction, topic, context_snippet, source_memory_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session,
                msg_idx,
                instr_trunc,
                "",
                context_snippet(content, m_start, 60),
                src
            ],
        )?;
        Ok(())
    };

    for cap in instruction_main_re().captures_iter(content) {
        let m = cap.get(0).unwrap();
        store(m.start(), m.as_str())?;
    }
    // `should`/`should not` only count when the tail is `(you|we|i|one) <imperative-verb>`.
    for cap in instruction_should_re().captures_iter(content) {
        let m = cap.get(0).unwrap();
        let tail = cap.get(1).unwrap().as_str();
        if instruction_should_tail_re().is_match(tail) {
            store(m.start(), m.as_str())?;
        }
    }
    Ok(())
}

fn extract_preferences(
    conn: &Connection,
    session: &str,
    content: &str,
    msg_idx: i64,
    src: &str,
) -> Result<()> {
    let word_re = re(r"[a-zA-Z]{4,}");
    let store = |m_start: usize, pref: &str, topic: &str| -> Result<()> {
        let pref = pref.trim();
        let topic = topic.trim();
        let topic_trunc: String = topic.chars().take(60).collect();
        // Build a topic key for evolution detection.
        let words: Vec<String> = word_re
            .find_iter(topic)
            .map(|m| m.as_str().to_string())
            .filter(|w| !FACT_MATCH_STOPWORDS.contains(&w.to_lowercase().as_str()))
            .collect();
        let joined = words.join(" ");
        let topic_key: String = if joined.is_empty() {
            topic.chars().take(20).collect()
        } else {
            joined.chars().take(30).collect()
        };
        let evolution: Option<String> = if topic_key.is_empty() {
            None
        } else {
            let like = format!("%{topic_key}%");
            conn.query_row(
                "SELECT preference FROM memoria_preferences \
                 WHERE session_id = ?1 AND (topic LIKE ?2 OR preference LIKE ?2) \
                 ORDER BY message_idx DESC LIMIT 1",
                params![session, like],
                |r| r.get::<_, String>(0),
            )
            .ok()
            .map(|prev| format!("was: {}", prev.chars().take(120).collect::<String>()))
        };
        let pref_trunc: String = pref.chars().take(200).collect();
        conn.execute(
            "INSERT INTO memoria_preferences (session_id, message_idx, preference, topic, evolution, context_snippet, source_memory_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session,
                msg_idx,
                pref_trunc,
                topic_trunc,
                evolution,
                context_snippet(content, m_start, 60),
                src
            ],
        )?;
        Ok(())
    };

    for cap in preference_fp_re().captures_iter(content) {
        let m = cap.get(0).unwrap();
        store(m.start(), m.as_str(), cap.get(1).unwrap().as_str())?;
    }
    for cap in preference_3p_re().captures_iter(content) {
        let m = cap.get(0).unwrap();
        store(m.start(), m.as_str(), cap.get(1).unwrap().as_str())?;
    }
    for cap in preference_struct_re().captures_iter(content) {
        let pref = cap.get(1).unwrap();
        let tail = cap.get(2).unwrap().as_str();
        store(pref.start(), pref.as_str(), tail)?;
    }
    Ok(())
}

// ── Retrieve path ────────────────────────────────────────────────────────────────────────────

/// Route `query` to the appropriate MEMORIA specialist table (`beam.py` `memoria_retrieve` L4566).
/// Returns `None` when the query does not classify to an ability or nothing matches.
pub fn memoria_retrieve(
    conn: &Connection,
    session_id: &str,
    query: &str,
    top_k: usize,
) -> Option<MemoriaResult> {
    match classify_ability(query).as_str() {
        "IE" | "KU" => fact_retrieve(conn, session_id, query, top_k),
        "TR" => timeline_retrieve(conn, session_id, query, top_k),
        "CR" => negation_retrieve(conn, session_id, query, top_k),
        "MR" => entity_retrieve(conn, session_id, query, top_k),
        "EO" => chrono_retrieve(conn, session_id, top_k),
        "IF" => instruction_retrieve(conn, session_id, query, top_k),
        "PF" => preference_retrieve(conn, session_id, query, top_k),
        _ => None,
    }
}

/// Classify a question into a BEAM ability from keywords (`beam.py` `_classify_ability` L4594).
fn classify_ability(query: &str) -> String {
    let q = query.to_lowercase();
    let any = |kws: &[&str]| kws.iter().any(|k| q.contains(k));

    if any(&[
        "how many days",
        "how many weeks",
        "how many months",
        "how long",
        "how much time",
        "what date",
        "what day",
        "when did",
        "when does",
        "what is the deadline",
        "how many years",
        "between which dates",
        "timeline",
        "how far apart",
    ]) {
        return "TR".into();
    }
    if any(&[
        "list the order",
        "walk me through",
        "order in which",
        "chronological",
        "in what order",
        "sequence of events",
    ]) {
        return "EO".into();
    }
    if any(&[
        "have i",
        "did i",
        "am i",
        "has this",
        "contradict",
        "contradiction",
        "conflict",
    ]) {
        return "CR".into();
    }
    if any(&[
        "how many",
        "what is the",
        "what are the",
        "what was the",
        "what were the",
        "what was my",
        "when does",
        "what is",
        "what was",
        "what version",
        "which version",
        "when was",
        "when were",
        "how much",
        "how big",
        "how large",
        "how fast",
    ]) && !any(&[
        "how many days",
        "how many weeks",
        "how many months",
        "how many years",
        "how far apart",
    ]) {
        return "IE".into();
    }
    if any(&[
        "my preference",
        "my preferences",
        "what do i like",
        "what do i prefer",
        "what do i hate",
        "what do i dislike",
        "what do i love",
        "what do i want",
        "what do i need",
        "what do i tend",
        "do i like",
        "do i prefer",
        "do i hate",
        "do i dislike",
        "do i love",
        "do i want",
        "do i need",
        "my favorite",
        "my favourite",
        "my fav",
        "things i like",
        "things i love",
        "things i hate",
        "things i dislike",
        "things i prefer",
        "things i tend",
        "things i don",
        "things i avoid",
        "what i like",
        "what i love",
        "what i hate",
        "what i dislike",
        "what i prefer",
        "what i tend",
        "what i don",
        "what i avoid",
        "preferences",
    ]) {
        return "PF".into();
    }
    if any(&[
        "tell me about my background",
        "previous development",
        "work experience",
        "personal background",
    ]) {
        return "ABS".into();
    }
    if any(&[
        "across my",
        "across all",
        "in my project",
        "in my sessions",
        "across sessions",
    ]) {
        return "MR".into();
    }
    for prefix in ["what ", "when ", "where ", "which ", "who ", "how "] {
        if q.starts_with(prefix) {
            return "IE".into();
        }
    }
    String::new()
}

/// One `memoria_facts` candidate row with its version metadata (`beam.py` L4688).
#[derive(Clone)]
struct FactRow {
    ftype: String,
    key: String,
    value: String,
    previous_value: Option<String>,
    updated_msg_idx: Option<i64>,
    version_id: i64,
    source_memory_id: Option<String>,
}

fn query_facts(conn: &Connection, sql: &str, binds: &[Value]) -> Vec<FactRow> {
    let mut out = Vec::new();
    if let Ok(mut stmt) = conn.prepare(sql) {
        if let Ok(rows) = stmt.query_map(params_from_iter(binds.iter().cloned()), |r| {
            Ok(FactRow {
                ftype: r.get(0)?,
                key: r.get(1)?,
                value: r.get(2)?,
                previous_value: r.get(3)?,
                updated_msg_idx: r.get(4)?,
                version_id: r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                source_memory_id: r.get(6)?,
            })
        }) {
            out = rows.flatten().collect();
        }
    }
    out
}

/// Query `memoria_facts` for metric/version/entity matches (`beam.py` `_memoria_fact_retrieve`
/// L4670). Multi-pass: numbers, capitalized terms, synonym map, context-snippet fallback.
fn fact_retrieve(
    conn: &Connection,
    session: &str,
    query: &str,
    top_k: usize,
) -> Option<MemoriaResult> {
    let q_lower = query.to_lowercase();
    let mut facts: Vec<FactRow> = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();
    let push = |rows: Vec<FactRow>, facts: &mut Vec<FactRow>, seen: &mut Vec<(String, String)>| {
        for row in rows {
            let fk = (row.key.clone(), row.value.clone());
            if !seen.contains(&fk) {
                seen.push(fk);
                facts.push(row);
            }
        }
    };
    let sel = "SELECT fact_type, key, value, previous_value, updated_msg_idx, version_id, \
               source_memory_id FROM memoria_facts";

    // Pass 1: numbers in the query -> match fact values.
    let numbers: Vec<&str> = re(r"\b(\d+)\b")
        .captures_iter(query)
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .collect();
    for num in numbers.iter().take(3) {
        let rows = query_facts(
            conn,
            &format!("{sel} WHERE value LIKE ? AND session_id = ? LIMIT ?"),
            &[
                Value::Text(format!("%{num}%")),
                Value::Text(session.to_string()),
                Value::Integer(top_k as i64),
            ],
        );
        push(rows, &mut facts, &mut seen);
    }

    // Pass 2: capitalized terms -> match key + value.
    let term_re = re(r"\b[A-Z][a-z]+(?:[-][A-Z][a-z]+)*\b");
    let stop2: &[&str] = &[
        "Have", "Did", "Do", "Can", "Will", "Would", "Should", "What", "When", "Where", "Which",
        "Who", "How", "Why", "Is", "Are", "Was", "Were", "The", "A", "An", "This", "That", "My",
        "Me", "I", "You", "Many", "Much",
    ];
    let terms: Vec<&str> = term_re
        .find_iter(query)
        .map(|m| m.as_str())
        .filter(|t| !stop2.contains(t))
        .collect();
    for term in terms.iter().take(5) {
        let rows = query_facts(
            conn,
            &format!("{sel} WHERE (key LIKE ? OR value LIKE ?) AND session_id = ? LIMIT ?"),
            &[
                Value::Text(format!("%{term}%")),
                Value::Text(format!("%{term}%")),
                Value::Text(session.to_string()),
                Value::Integer(top_k as i64),
            ],
        );
        push(rows, &mut facts, &mut seen);
    }

    // Pass 3: synonym map.
    let synonym_map: &[(&str, &str, Option<&str>)] = &[
        ("version", "version", None),
        ("latency", "metric", Some("ms")),
        ("speed", "metric", Some("ms")),
        ("response time", "metric", Some("ms")),
        ("how many", "metric", None),
        ("how much", "metric", None),
        ("what date", "date", None),
        ("what day", "date", None),
        ("deployed", "date", None),
        ("deploy", "date", None),
        ("released", "date", None),
        ("release", "date", None),
        ("launched", "date", None),
    ];
    if facts.is_empty() {
        for (phrase, ftype, unit_hint) in synonym_map {
            if q_lower.contains(phrase) {
                let rows = match unit_hint {
                    Some(unit) => query_facts(
                        conn,
                        &format!(
                            "{sel} WHERE fact_type = ? AND key LIKE ? AND session_id = ? LIMIT ?"
                        ),
                        &[
                            Value::Text(ftype.to_string()),
                            Value::Text(format!("%{unit}%")),
                            Value::Text(session.to_string()),
                            Value::Integer(top_k as i64),
                        ],
                    ),
                    None => query_facts(
                        conn,
                        &format!("{sel} WHERE fact_type = ? AND session_id = ? LIMIT ?"),
                        &[
                            Value::Text(ftype.to_string()),
                            Value::Text(session.to_string()),
                            Value::Integer(top_k as i64),
                        ],
                    ),
                };
                push(rows, &mut facts, &mut seen);
                if !facts.is_empty() {
                    break;
                }
            }
        }
    }

    // Pass 4: context-snippet fallback.
    if facts.is_empty() {
        let q_stop: &[&str] = &[
            "what", "when", "where", "which", "who", "how", "why", "is", "are", "was", "were",
            "do", "does", "did", "can", "will", "would", "should", "could", "may", "the", "a",
            "an", "in", "on", "at", "to", "for", "of", "with", "my", "me", "i", "you", "it", "its",
            "this", "that", "these", "those", "tell", "list", "describe", "explain", "walk",
            "through",
        ];
        let words: Vec<String> = re(r"\b[a-zA-Z]{3,}\b")
            .find_iter(&q_lower)
            .map(|m| m.as_str().to_string())
            .filter(|w| !q_stop.contains(&w.as_str()))
            .collect();
        for word in words.iter().take(5) {
            let rows = query_facts(
                conn,
                &format!("{sel} WHERE context_snippet LIKE ? AND session_id = ? LIMIT ?"),
                &[
                    Value::Text(format!("%{word}%")),
                    Value::Text(session.to_string()),
                    Value::Integer(top_k as i64),
                ],
            );
            push(rows, &mut facts, &mut seen);
            if !facts.is_empty() {
                break;
            }
        }
    }

    if facts.is_empty() {
        return None;
    }

    // Group by key and keep only the newest version per key — multiple versions of one key read
    // as contradictions — rendering the older hits as an evolution chain (`beam.py` L4787-L4814).
    let mut key_order: Vec<String> = Vec::new();
    let mut by_key: std::collections::HashMap<String, Vec<FactRow>> =
        std::collections::HashMap::new();
    for f in facts {
        if !by_key.contains_key(&f.key) {
            key_order.push(f.key.clone());
        }
        by_key.entry(f.key.clone()).or_default().push(f);
    }
    let mut latest: Vec<(FactRow, Option<String>)> = Vec::new();
    for key in key_order {
        let mut versions = by_key.remove(&key).unwrap_or_default();
        versions.sort_by_key(|v| std::cmp::Reverse(v.version_id));
        let newest = versions[0].clone();
        let evolution = (versions.len() > 1).then(|| {
            let mut chain: Vec<&str> = versions[1..].iter().map(|v| v.value.as_str()).collect();
            chain.reverse();
            format!("{} -> {}", chain.join(" -> "), newest.value)
        });
        latest.push((newest, evolution));
    }
    latest.sort_by_key(|(f, _)| std::cmp::Reverse(f.version_id));

    let mut lines = Vec::new();
    let mut source_ids = Vec::new();
    for (f, evolution) in latest.iter().take(top_k) {
        let mut line = format!("[Fact {}] {}: {}", f.ftype, f.key, f.value);
        if let Some(evo) = evolution {
            line.push_str(&format!(" (evolved: {evo})"));
        } else if let (Some(prev), true) = (&f.previous_value, f.version_id > 0) {
            let at = f
                .updated_msg_idx
                .map_or_else(|| "?".to_string(), |i| i.to_string());
            line.push_str(&format!(" (was: {prev}, updated at msg_idx {at})"));
        }
        lines.push(line);
        if let Some(sid) = &f.source_memory_id {
            source_ids.push(sid.clone());
        }
    }
    Some(MemoriaResult {
        context: lines.join("\n"),
        source: "memoria_facts".into(),
        source_memory_ids: source_ids,
    })
}

fn timeline_retrieve(
    conn: &Connection,
    session: &str,
    query: &str,
    top_k: usize,
) -> Option<MemoriaResult> {
    let date_terms: Vec<&str> = iso_date_re()
        .captures_iter(query)
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .collect();
    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let q_lower = query.to_lowercase();
    let month_hit = months.iter().find(|m| q_lower.contains(**m));

    let (sql, binds): (String, Vec<Value>) = if let Some(dt) = date_terms.first() {
        (
            "SELECT date, description, message_idx FROM memoria_timelines WHERE date LIKE ? AND session_id = ? ORDER BY date LIMIT ?".into(),
            vec![Value::Text(format!("%{dt}%")), Value::Text(session.to_string()), Value::Integer(top_k as i64)],
        )
    } else if let Some(month) = month_hit {
        (
            "SELECT date, description, message_idx FROM memoria_timelines WHERE date LIKE ? AND session_id = ? ORDER BY date LIMIT ?".into(),
            vec![Value::Text(format!("{}%", &month[..3])), Value::Text(session.to_string()), Value::Integer(top_k as i64)],
        )
    } else {
        (
            "SELECT date, description, message_idx FROM memoria_timelines WHERE session_id = ? ORDER BY date DESC LIMIT ?".into(),
            vec![Value::Text(session.to_string()), Value::Integer(top_k as i64)],
        )
    };

    let mut lines = Vec::new();
    if let Ok(mut stmt) = conn.prepare(&sql) {
        if let Ok(rows) = stmt.query_map(params_from_iter(binds), |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?.unwrap_or_default(),
            ))
        }) {
            for (date, desc) in rows.flatten() {
                let d: String = desc.chars().take(120).collect();
                lines.push(format!("[{date}] {d}"));
            }
        }
    }
    if lines.is_empty() {
        return None;
    }
    Some(MemoriaResult {
        context: lines.join("\n"),
        source: "memoria_timelines".into(),
        source_memory_ids: Vec::new(),
    })
}

fn negation_retrieve(
    conn: &Connection,
    session: &str,
    query: &str,
    top_k: usize,
) -> Option<MemoriaResult> {
    let stop: &[&str] = &["Have", "Did", "Do", "Can", "Will", "Would", "Should"];
    let mut terms: Vec<String> = re(r"\b[A-Z][a-z]+\b")
        .find_iter(query)
        .map(|m| m.as_str().to_string())
        .filter(|t| t.len() > 3 && !stop.contains(&t.as_str()))
        .collect();
    if terms.is_empty() {
        terms = query
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .take(3)
            .map(|w| w.to_string())
            .collect();
    }
    for term in &terms {
        let mut lines = Vec::new();
        if let Ok(mut stmt) = conn.prepare(
            "SELECT subject, object, message_idx FROM memoria_kg \
             WHERE predicate='negation' AND (subject LIKE ? OR object LIKE ?) AND session_id = ? LIMIT ?",
        ) {
            let like = format!("%{term}%");
            if let Ok(rows) = stmt.query_map(
                params![like, like, session, top_k as i64],
                |r| r.get::<_, String>(1),
            ) {
                for obj in rows.flatten() {
                    lines.push(format!("[Negation] user said never/not: {obj}"));
                }
            }
        }
        if !lines.is_empty() {
            return Some(MemoriaResult {
                context: lines.join("\n"),
                source: "memoria_kg_negation".into(),
                source_memory_ids: Vec::new(),
            });
        }
    }
    None
}

fn entity_retrieve(
    conn: &Connection,
    session: &str,
    query: &str,
    top_k: usize,
) -> Option<MemoriaResult> {
    let stop: &[&str] = &[
        "Have", "Did", "Do", "Can", "Will", "Would", "Should", "What", "When", "Where", "Which",
        "Who", "How", "Why",
    ];
    let entities: Vec<String> = re(r"\b[A-Z][a-z]+\b")
        .find_iter(query)
        .map(|m| m.as_str())
        .filter(|t| !stop.contains(t) && t.len() > 3)
        .map(|t| t.to_lowercase())
        .collect();

    let mut rows: Vec<(String, String, String)> = Vec::new();
    for entity in entities.iter().take(3) {
        rows = fetch_kg(
            conn,
            "SELECT subject, predicate, object FROM memoria_kg WHERE (subject LIKE ? OR object LIKE ?) AND session_id = ? LIMIT ?",
            &[Value::Text(format!("%{entity}%")), Value::Text(format!("%{entity}%")), Value::Text(session.to_string()), Value::Integer(top_k as i64)],
        );
        if !rows.is_empty() {
            break;
        }
    }
    if rows.is_empty() {
        rows = fetch_kg(
            conn,
            "SELECT subject, predicate, object FROM memoria_kg WHERE session_id = ? ORDER BY message_idx LIMIT ?",
            &[Value::Text(session.to_string()), Value::Integer(top_k as i64)],
        );
    }
    if rows.is_empty() {
        return None;
    }
    let lines: Vec<String> = rows
        .iter()
        .map(|(s, p, o)| format!("[{p}] {s} -> {o}"))
        .collect();
    Some(MemoriaResult {
        context: lines.join("\n"),
        source: "memoria_kg".into(),
        source_memory_ids: Vec::new(),
    })
}

fn fetch_kg(conn: &Connection, sql: &str, binds: &[Value]) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    if let Ok(mut stmt) = conn.prepare(sql) {
        if let Ok(rows) = stmt.query_map(params_from_iter(binds.iter().cloned()), |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            ))
        }) {
            out = rows.flatten().collect();
        }
    }
    out
}

fn chrono_retrieve(conn: &Connection, session: &str, top_k: usize) -> Option<MemoriaResult> {
    let mut lines = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT value, message_idx FROM memoria_facts \
         WHERE fact_type='sequence' AND session_id = ? ORDER BY message_idx ASC LIMIT ?",
    ) {
        if let Ok(rows) = stmt.query_map(params![session, top_k as i64], |r| r.get::<_, String>(0))
        {
            for (i, value) in rows.flatten().enumerate() {
                lines.push(format!("[{}] {value}", i + 1));
            }
        }
    }
    if lines.is_empty() {
        return None;
    }
    Some(MemoriaResult {
        context: lines.join("\n"),
        source: "memoria_sequences".into(),
        source_memory_ids: Vec::new(),
    })
}

fn instruction_retrieve(
    conn: &Connection,
    session: &str,
    query: &str,
    top_k: usize,
) -> Option<MemoriaResult> {
    let words = topic_words(query);
    let mut lines = Vec::new();
    let mut matched = false;
    for word in words.iter().take(5) {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT instruction FROM memoria_instructions \
             WHERE (instruction LIKE ? OR topic LIKE ?) AND session_id = ? AND active = 1 LIMIT ?",
        ) {
            let like = format!("%{word}%");
            if let Ok(rows) = stmt.query_map(params![like, like, session, top_k as i64], |r| {
                r.get::<_, String>(0)
            }) {
                for instr in rows.flatten() {
                    matched = true;
                    let t: String = instr.chars().take(120).collect();
                    lines.push(format!("[Instruction] {t}"));
                }
            }
        }
        if matched {
            break;
        }
    }
    if !matched {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT instruction FROM memoria_instructions \
             WHERE session_id = ? AND active = 1 ORDER BY message_idx DESC LIMIT ?",
        ) {
            if let Ok(rows) =
                stmt.query_map(params![session, top_k as i64], |r| r.get::<_, String>(0))
            {
                for instr in rows.flatten() {
                    let t: String = instr.chars().take(120).collect();
                    lines.push(format!("[Instruction] {t}"));
                }
            }
        }
    }
    if lines.is_empty() {
        return None;
    }
    Some(MemoriaResult {
        context: lines.join("\n"),
        source: "memoria_instructions".into(),
        source_memory_ids: Vec::new(),
    })
}

fn preference_retrieve(
    conn: &Connection,
    session: &str,
    query: &str,
    top_k: usize,
) -> Option<MemoriaResult> {
    let words = topic_words(query);
    let mut rows: Vec<(String, Option<String>)> = Vec::new();
    for word in words.iter().take(5) {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT preference, evolution FROM memoria_preferences \
             WHERE (preference LIKE ? OR topic LIKE ?) AND session_id = ? LIMIT ?",
        ) {
            let like = format!("%{word}%");
            if let Ok(qr) = stmt.query_map(params![like, like, session, top_k as i64], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
            }) {
                rows = qr.flatten().collect();
            }
        }
        if !rows.is_empty() {
            break;
        }
    }
    if rows.is_empty() {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT preference, evolution FROM memoria_preferences \
             WHERE session_id = ? ORDER BY message_idx DESC LIMIT ?",
        ) {
            if let Ok(qr) = stmt.query_map(params![session, top_k as i64], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
            }) {
                rows = qr.flatten().collect();
            }
        }
    }
    if rows.is_empty() {
        return None;
    }
    let lines: Vec<String> = rows
        .iter()
        .map(|(pref, evo)| {
            let p: String = pref.chars().take(120).collect();
            match evo {
                Some(e) if !e.is_empty() => format!("[Preference] {p} ({e})"),
                _ => format!("[Preference] {p}"),
            }
        })
        .collect();
    Some(MemoriaResult {
        context: lines.join("\n"),
        source: "memoria_preferences".into(),
        source_memory_ids: Vec::new(),
    })
}

/// The query topic words shared by the instruction/preference retrievers (`beam.py` L4946-L4954).
fn topic_words(query: &str) -> Vec<String> {
    let stop: &[&str] = &[
        "what", "when", "where", "which", "who", "how", "why", "is", "are", "was", "were", "do",
        "does", "did", "can", "will", "would", "should", "could", "may", "the", "a", "an", "in",
        "on", "at", "to", "for", "of", "with", "my", "me", "i", "you", "it", "its", "this", "that",
        "these", "those", "tell", "list", "describe", "explain", "have", "has", "had", "am",
    ];
    let q_lower = query.to_lowercase();
    re(r"\b[a-zA-Z]{3,}\b")
        .find_iter(&q_lower)
        .map(|m| m.as_str().to_string())
        .filter(|w| !stop.contains(&w.as_str()))
        .collect()
}

// ── Helpers ──────────────────────────────────────────────────────────────────────────────────

/// Surrounding context around byte position `pos`, char-boundary-safe (`beam.py` `_context_snippet`
/// L4542). Adds `...` ellipses when truncated and caps the result at 200 chars.
fn context_snippet(content: &str, pos: usize, width: usize) -> String {
    let start = floor_boundary(content, pos.saturating_sub(width));
    let end = ceil_boundary(content, (pos + width).min(content.len()));
    let core = content[start..end].trim();
    let mut s = String::new();
    if start > 0 {
        s.push_str("...");
    }
    s.push_str(core);
    if end < content.len() {
        s.push_str("...");
    }
    s.chars().take(200).collect()
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn store() -> Store {
        Store::open_in_memory().expect("open store")
    }

    #[test]
    fn ability_classifier_routes_questions() {
        assert_eq!(classify_ability("How many days until launch?"), "TR");
        assert_eq!(
            classify_ability("What version of Postgres are we on?"),
            "IE"
        );
        assert_eq!(
            classify_ability("Walk me through the order of events"),
            "EO"
        );
        assert_eq!(classify_ability("Have I ever used Redis?"), "CR");
        assert_eq!(classify_ability("what do i prefer for testing?"), "PF");
        assert_eq!(classify_ability("just a statement, not a question"), "");
    }

    #[test]
    fn extract_and_retrieve_a_metric_fact() {
        let s = store();
        let conn = s.conn.lock().unwrap();
        extract_and_store(
            &conn,
            "sess",
            "The dashboard API response time of 250ms was measured today.",
            0,
            "mem-1",
        )
        .unwrap();
        let got =
            memoria_retrieve(&conn, "sess", "What is the response time?", 3).expect("a fact hit");
        assert_eq!(got.source, "memoria_facts");
        assert!(got.context.contains("250ms"), "ctx: {}", got.context);
        assert_eq!(got.source_memory_ids, vec!["mem-1".to_string()]);
    }

    #[test]
    fn extract_version_and_iso_timeline() {
        let s = store();
        let conn = s.conn.lock().unwrap();
        extract_and_store(
            &conn,
            "sess",
            "We deployed PostgreSQL v14.2 and the release is scheduled on 2024-03-29.",
            0,
            "mem-2",
        )
        .unwrap();

        let ver =
            memoria_retrieve(&conn, "sess", "What version of PostgreSQL?", 3).expect("version hit");
        assert!(ver.context.contains("14.2"), "ctx: {}", ver.context);

        let tl = memoria_retrieve(&conn, "sess", "What date is the release?", 3);
        // "what date" routes to IE (fact), but the iso date is also stored as a fact.
        assert!(tl.is_some());
    }

    #[test]
    fn negation_is_extracted_and_retrieved() {
        let s = store();
        let conn = s.conn.lock().unwrap();
        extract_and_store(
            &conn,
            "sess",
            "I have never used MongoDB for this project at all.",
            0,
            "mem-3",
        )
        .unwrap();
        let got = memoria_retrieve(&conn, "sess", "Have I used MongoDB?", 3).expect("negation hit");
        assert_eq!(got.source, "memoria_kg_negation");
        assert!(
            got.context.to_lowercase().contains("mongodb"),
            "ctx: {}",
            got.context
        );
    }

    #[test]
    fn fact_versioning_chains_value_changes() {
        let s = store();
        let conn = s.conn.lock().unwrap();
        // Same slot (postgresql_version), three values across messages -> a version chain.
        extract_and_store(&conn, "sess", "We run PostgreSQL v14.2 in prod.", 0, "m1").unwrap();
        extract_and_store(
            &conn,
            "sess",
            "Upgraded to PostgreSQL v15.1 today.",
            4,
            "m2",
        )
        .unwrap();
        extract_and_store(&conn, "sess", "Now on PostgreSQL v16.0 finally.", 9, "m3").unwrap();

        // The write side closes superseded rows and bumps version_id (`beam.py` L4491-L4517).
        let (open_rows, max_version): (i64, i64) = conn
            .query_row(
                "SELECT COUNT(*) FILTER (WHERE valid_to_msg_idx IS NULL), MAX(version_id) \
                 FROM memoria_facts WHERE key = 'postgresql_version'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(open_rows, 1, "exactly one current version per slot");
        assert_eq!(max_version, 2, "two supersessions -> version_id 2");
        let (prev, updated_at): (String, i64) = conn
            .query_row(
                "SELECT previous_value, updated_msg_idx FROM memoria_facts \
                 WHERE key = 'postgresql_version' AND valid_to_msg_idx IS NULL",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(prev, "15.1");
        assert_eq!(updated_at, 9);

        // The read side renders only the newest value, with the evolution chain when the older
        // versions also matched (`beam.py` L4787-L4814).
        let got = memoria_retrieve(&conn, "sess", "What version of PostgreSQL?", 5).expect("hit");
        assert!(got.context.contains("16.0"), "ctx: {}", got.context);
        assert_eq!(
            got.context.matches("postgresql_version:").count(),
            1,
            "one line per key: {}",
            got.context
        );
        assert!(
            got.context.contains("(evolved: 14.2 -> 15.1 -> 16.0)")
                || got.context.contains("(was: 15.1, updated at msg_idx 9)"),
            "version history rendered: {}",
            got.context
        );
    }

    #[test]
    fn plain_statement_returns_no_hit() {
        let s = store();
        let conn = s.conn.lock().unwrap();
        extract_and_store(&conn, "sess", "The API response time of 250ms.", 0, "m").unwrap();
        // A non-question query does not classify -> no MEMORIA hit.
        assert!(memoria_retrieve(&conn, "sess", "response time", 3).is_none());
    }
}
