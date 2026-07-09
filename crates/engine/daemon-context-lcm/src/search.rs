// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The transcript/DAG search stack (`daemon-context-lcm-port-spec.md` §11) — a faithful port of
//! hermes-lcm's search layer (`LCM:search_query.py`, `LCM:store.py:669-981`,
//! `LCM:dag.py:331-537`):
//!
//! * **FTS5 sanitize** (§11.1) — char-preserving scrub of FTS5 operator characters that keeps
//!   balanced `"quoted phrases"` intact (`sanitize_fts5_query`).
//! * **LIKE fallback** (§11.2) — CJK/emoji (dropped by the `unicode61` tokenizer) and risky ASCII
//!   shapes (`foo-bar`, `a:b`, unbalanced quotes) route to a substring scan; risky-ASCII queries
//!   also collapse repeat hits to once-per-term when scoring.
//! * **Sort modes + directness + widening** (§11.3) — `recency` / `relevance` / `hybrid` with a
//!   role bias (user < assistant < tool), a directness score that rewards distinct term/phrase
//!   hits and penalizes repetition/phrase-stuffing, a rank nudge for precise query shapes, and a
//!   fetch-widening ladder that pages candidates (doubling up to a hard cap) until the visible
//!   window is provably stable.
//!
//! Query-shaping lives here; the raw SQL (FTS MATCH / LIKE / recursive source-lineage CTE) lives
//! in [`crate::store`]. Ranks follow FTS5 convention: **lower is better**. LIKE candidates
//! synthesize `rank = -(term-hit score)` so more hits sort first under the same comparators.

use crate::store::{MessageFilter, MessageRow, Store, SummaryNode};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hybrid-sort age decay (`LCM:search_query.py:256`): a match's strength is divided by
/// `1 + age_hours * RATE`. Shared with the SQL `ORDER BY` builders in the store.
pub const AGE_DECAY_RATE: f64 = 0.001;
/// Directness nudge weight for message ranks (`LCM:store.py:710`).
const MESSAGE_DIRECTNESS_WEIGHT: f64 = 3e-7;
/// Directness nudge weight for node ranks (`LCM:dag.py:354`).
const NODE_DIRECTNESS_WEIGHT: f64 = 2e-7;
/// Snippet window width in characters (`LCM:search_query.py:265`).
const SNIPPET_WIDTH: usize = 80;

/// Characters that are special in FTS5 query syntax (`LCM:search_query.py:30`).
const FTS5_SPECIAL_CHARS: &[char] = &['"', '(', ')', '*', '^', '-', ':', '{', '}', '.'];
/// Edge punctuation stripped from bare tokens (`LCM:search_query.py:28`).
const STRIP_EDGE_PUNCT: &[char] = &['"', '\'', '(', ')', '[', ']', '{', '}', '.', ',', ';'];
/// FTS5 boolean operators excluded from LIKE terms (`LCM:search_query.py:25`).
const BOOLEAN_OPERATORS: &[&str] = &["AND", "OR", "NOT", "NEAR"];

/// How matches are ordered (§11.3). Anything unrecognized normalizes to [`SortMode::Recency`]
/// (`normalize_search_sort`, `LCM:search_query.py:259-262`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SortMode {
    /// Newest first; role bias and rank break ties.
    #[default]
    Recency,
    /// Best rank first; directness, role bias, then newest break ties.
    Relevance,
    /// Rank discounted by age (recent strong matches win).
    Hybrid,
}

impl SortMode {
    /// Parse a caller-supplied sort string; unknown values fall back to `Recency`.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "relevance" => SortMode::Relevance,
            "hybrid" => SortMode::Hybrid,
            _ => SortMode::Recency,
        }
    }

    /// The canonical string form.
    pub fn as_str(self) -> &'static str {
        match self {
            SortMode::Recency => "recency",
            SortMode::Relevance => "relevance",
            SortMode::Hybrid => "hybrid",
        }
    }
}

/// A scored transcript match: the row, its (possibly nudged) rank, and a context snippet.
#[derive(Clone, Debug)]
pub struct MessageResult {
    /// The matched message row.
    pub row: MessageRow,
    /// The final rank used for ordering (lower = better).
    pub rank: f64,
    /// The match excerpt: SQL `snippet()` with `>>>`/`<<<` markers on the FTS path, an ~80-char
    /// window around the first term hit on the LIKE path.
    pub snippet: String,
    /// The directness score (higher = the text is more squarely about the query).
    pub directness: f64,
}

/// A scored summary-node match.
#[derive(Clone, Debug)]
pub struct NodeResult {
    /// The matched summary node.
    pub node: SummaryNode,
    /// The final rank used for ordering (lower = better).
    pub rank: f64,
    /// The directness score (higher = better).
    pub directness: f64,
}

struct ScoredMessage {
    row: MessageRow,
    rank: f64,
    directness: f64,
    snippet: String,
}

struct ScoredNode {
    node: SummaryNode,
    rank: f64,
    directness: f64,
}

// ---- public entry points ------------------------------------------------------------------------

/// Search the lossless message transcript (`Store.search`, `LCM:store.py:671-790`). Routes
/// CJK/emoji/risky-ASCII queries (and FTS syntax errors) to the LIKE fallback; otherwise pages
/// FTS candidates through the widening ladder, applies the directness rank nudge for precise
/// query shapes, and truncates to `limit` after the final sort.
pub fn search_messages(
    store: &Store,
    query: &str,
    sort: SortMode,
    filter: &MessageFilter<'_>,
    limit: usize,
) -> crate::Result<Vec<MessageResult>> {
    let safe_query = sanitize_fts5_query(query);
    let terms = extract_search_terms(&safe_query);
    let phrases = extract_quoted_phrases(&safe_query);
    // A low-disk-degraded store has no FTS index at all (the rebuild dropped it) — route every
    // query straight to the substring scan until a later repair pass re-enables FTS.
    if requires_like_fallback(query) || store.is_degraded() {
        return search_messages_like(store, query, sort, filter, limit);
    }

    let mut fetch_limit = compute_search_fetch_limit(limit, &terms, &phrases);
    let candidate_cap = compute_search_candidate_cap(limit);
    let apply_adjustment = should_apply_directness_rank_adjustment(&terms, &phrases);
    let max_rank_bonus =
        compute_directness_rank_bonus_upper_bound(&terms, &phrases) * MESSAGE_DIRECTNESS_WEIGHT;
    let now = now_secs();
    let mut offset = 0usize;
    let mut scanned = 0usize;
    let mut results: Vec<ScoredMessage> = Vec::new();
    loop {
        let page = match store.search_messages_fts(
            &safe_query,
            filter,
            sort,
            fetch_limit as i64,
            offset as i64,
        ) {
            Ok(page) => page,
            // Any SQL error (e.g. a residual MATCH syntax error) falls back to the substring
            // scan, like Python's catch-all (`LCM:store.py:747-758`).
            Err(_) => return search_messages_like(store, query, sort, filter, limit),
        };
        scanned += page.len();
        let page_len = page.len();
        let mut last_primary = f64::INFINITY;
        for hit in page {
            let directness = message_directness_score(
                &hit.row.role,
                hit.row.content.as_deref(),
                &terms,
                &phrases,
            );
            let mut rank = hit.rank;
            if apply_adjustment {
                rank -= directness.max(0.0) * MESSAGE_DIRECTNESS_WEIGHT;
            }
            let scored = ScoredMessage {
                row: hit.row,
                rank,
                directness,
                snippet: hit.snippet,
            };
            last_primary = fts_message_primary(&scored, sort, now);
            results.push(scored);
        }
        results.sort_by(|a, b| {
            cmp_keys(
                &fts_message_sort_key(a, sort, now),
                &fts_message_sort_key(b, sort, now),
            )
        });

        // Ladder exit conditions (`LCM:store.py:774-790`): no nudge in play, a short page (the
        // index is exhausted), or the visible window is provably stable (the best not-yet-seen
        // candidate cannot beat the worst visible one even with the maximum nudge).
        if !apply_adjustment || page_len < fetch_limit || results.len() <= limit {
            break;
        }
        let worst_visible = fts_message_primary(
            &results[limit.min(results.len()).saturating_sub(1)],
            sort,
            now,
        );
        if last_primary - max_rank_bonus > worst_visible {
            break;
        }
        if scanned >= candidate_cap {
            break;
        }
        offset += page_len;
        let remaining = candidate_cap - scanned;
        if remaining == 0 {
            break;
        }
        fetch_limit = (fetch_limit * 2).min(remaining);
    }
    results.truncate(limit);
    Ok(results.into_iter().map(message_result).collect())
}

/// Search the summary DAG (`SummaryDAG.search`, `LCM:dag.py:331-415`). Same FTS/LIKE routing and
/// widening ladder as the transcript search, plus an optional `source` filter that matches nodes
/// by their **descendant raw-message lineage** (recursive CTE in the store); `session = None`
/// searches every session in the bank.
pub fn search_nodes(
    store: &Store,
    query: &str,
    session: Option<&str>,
    sort: SortMode,
    source: Option<&str>,
    limit: usize,
) -> crate::Result<Vec<NodeResult>> {
    // Python truthiness: an empty source string means "no filter" (`LCM:dag.py:384`).
    let source = source.filter(|s| !s.is_empty());
    let safe_query = sanitize_fts5_query(query);
    let terms = extract_search_terms(&safe_query);
    let phrases = extract_quoted_phrases(&safe_query);
    // Degraded store (no FTS index): LIKE-only, as in the transcript search above.
    if requires_like_fallback(query) || store.is_degraded() {
        return search_nodes_like(store, query, session, sort, source, limit);
    }

    let mut fetch_limit = compute_search_fetch_limit(limit, &terms, &phrases);
    let candidate_cap = compute_search_candidate_cap(limit);
    let apply_adjustment = should_apply_directness_rank_adjustment(&terms, &phrases);
    let max_rank_bonus =
        compute_directness_rank_bonus_upper_bound(&terms, &phrases) * NODE_DIRECTNESS_WEIGHT;
    let now = now_secs();
    let mut offset = 0usize;
    let mut scanned = 0usize;
    let mut results: Vec<ScoredNode> = Vec::new();
    let mut source_cache: HashMap<i64, bool> = HashMap::new();
    loop {
        let page = match store.search_nodes_fts(
            &safe_query,
            session,
            sort,
            fetch_limit as i64,
            offset as i64,
        ) {
            Ok(page) => page,
            Err(_) => return search_nodes_like(store, query, session, sort, source, limit),
        };
        scanned += page.len();
        let page_len = page.len();
        let mut last_primary = f64::INFINITY;
        for hit in page {
            let recency = node_recency(&hit.node);
            let mut rank = hit.rank;
            let included = match source {
                Some(src) => store.node_matches_source(hit.node.node_id, src, &mut source_cache)?,
                None => true,
            };
            if included {
                let directness = compute_directness_score(&hit.node.summary, &terms, &phrases);
                if apply_adjustment {
                    rank -= directness.max(0.0) * NODE_DIRECTNESS_WEIGHT;
                }
                results.push(ScoredNode {
                    node: hit.node,
                    rank,
                    directness,
                });
            }
            // The ladder's "best unseen" probe uses the raw page tail — source-filtered rows
            // keep their unadjusted rank (`LCM:dag.py:406`).
            last_primary = fts_node_primary(rank, recency, sort, now);
        }
        results.sort_by(|a, b| {
            cmp_keys(
                &fts_node_sort_key(a, sort, now),
                &fts_node_sort_key(b, sort, now),
            )
        });

        let exhausted = page_len < fetch_limit || scanned >= candidate_cap;
        // A source filter keeps widening until exhaustion: filtered-out rows don't count toward
        // `limit`, so the visible-window proof doesn't apply (`LCM:dag.py:393-400`).
        if source.is_some() && !exhausted {
            offset += page_len;
            let remaining = candidate_cap - scanned;
            if remaining == 0 {
                break;
            }
            fetch_limit = (fetch_limit * 2).min(remaining);
            continue;
        }
        if exhausted || !apply_adjustment || results.len() <= limit {
            break;
        }
        let worst_visible = {
            let s = &results[limit.min(results.len()).saturating_sub(1)];
            fts_node_primary(s.rank, node_recency(&s.node), sort, now)
        };
        if last_primary - max_rank_bonus > worst_visible {
            break;
        }
        offset += page_len;
        let remaining = candidate_cap - scanned;
        if remaining == 0 {
            break;
        }
        fetch_limit = (fetch_limit * 2).min(remaining);
    }
    results.truncate(limit);
    Ok(results.into_iter().map(node_result).collect())
}

// ---- LIKE fallback paths ------------------------------------------------------------------------

/// The message LIKE fallback (`Store._search_like`, `LCM:store.py:792-981`). Recency sort pages
/// under the SQL-side score/directness `ORDER BY` (continuing across the candidate cap only to
/// finish the boundary timestamp/role-bias tie group); other sorts fetch one unordered batch.
fn search_messages_like(
    store: &Store,
    query: &str,
    sort: SortMode,
    filter: &MessageFilter<'_>,
    limit: usize,
) -> crate::Result<Vec<MessageResult>> {
    let safe_query = sanitize_fts5_query(query);
    let terms = extract_search_terms(&safe_query);
    let phrases = extract_quoted_phrases(&safe_query);
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let fetch_limit = compute_search_fetch_limit(limit, &terms, &phrases);
    let collapse = contains_risky_fts_ascii(query);
    let now = now_secs();
    let mut results: Vec<ScoredMessage> = Vec::new();

    if matches!(sort, SortMode::Recency) {
        let candidate_cap = compute_search_candidate_cap(limit);
        let mut offset = 0usize;
        let mut scanned = 0usize;
        loop {
            let batch_limit = fetch_limit.min(candidate_cap.saturating_sub(scanned));
            if batch_limit == 0 {
                break;
            }
            let rows = store.search_messages_like_recency(
                &terms,
                &phrases,
                collapse,
                filter,
                batch_limit as i64,
                offset as i64,
            )?;
            scanned += rows.len();
            let page_len = rows.len();
            let boundary = rows
                .last()
                .map(|r| (r.timestamp, message_role_bias(&r.role)));
            score_like_rows(rows, &terms, &phrases, collapse, &mut results);
            offset += page_len;
            if page_len < batch_limit {
                break;
            }
            if scanned >= candidate_cap {
                // Keep pulling rows that tie the boundary (timestamp, role-bias) group so the cap
                // never splits a tie mid-group (`LCM:store.py:941-967`).
                let Some((b_ts, b_bias)) = boundary else {
                    break;
                };
                loop {
                    let tie_rows = store.search_messages_like_recency(
                        &terms,
                        &phrases,
                        collapse,
                        filter,
                        fetch_limit as i64,
                        offset as i64,
                    )?;
                    if tie_rows.is_empty() {
                        break;
                    }
                    let tie_len = tie_rows.len();
                    let mut matching: Vec<MessageRow> = Vec::new();
                    let mut reached_next_group = false;
                    for row in tie_rows {
                        if row.timestamp == b_ts && message_role_bias(&row.role) == b_bias {
                            matching.push(row);
                        } else {
                            reached_next_group = true;
                            break;
                        }
                    }
                    score_like_rows(matching, &terms, &phrases, collapse, &mut results);
                    if reached_next_group || tie_len < fetch_limit {
                        break;
                    }
                    offset += tie_len;
                }
                break;
            }
        }
    } else {
        let rows = store.search_messages_like_unordered(&terms, filter, fetch_limit as i64)?;
        score_like_rows(rows, &terms, &phrases, collapse, &mut results);
    }

    results.sort_by(|a, b| {
        cmp_keys(
            &fallback_message_sort_key(a, sort, now),
            &fallback_message_sort_key(b, sort, now),
        )
    });
    results.truncate(limit);
    Ok(results.into_iter().map(message_result).collect())
}

/// Score one batch of LIKE candidates (`add_rows`, `LCM:store.py:904-918`): term-hit count
/// (collapsed to once-per-term for risky-ASCII queries) becomes `rank = -score`; zero-score rows
/// are dropped.
fn score_like_rows(
    rows: Vec<MessageRow>,
    terms: &[String],
    phrases: &[String],
    collapse: bool,
    out: &mut Vec<ScoredMessage>,
) {
    for row in rows {
        let content = row.content.clone().unwrap_or_default();
        let mut score = 0usize;
        for term in terms {
            let hits = count_term_matches(&content, term);
            score += if collapse { hits.min(1) } else { hits };
        }
        if score == 0 {
            continue;
        }
        let directness = message_directness_score(&row.role, Some(&content), terms, phrases);
        let snippet = build_snippet(&content, terms);
        out.push(ScoredMessage {
            row,
            rank: -(score as f64),
            directness,
            snippet,
        });
    }
}

/// The node LIKE fallback (`SummaryDAG._search_like`, `LCM:dag.py:417-475`). Pages unordered
/// candidates (widening only while a source filter is dropping rows), scores by term hits, and
/// sorts with the node fallback comparators.
fn search_nodes_like(
    store: &Store,
    query: &str,
    session: Option<&str>,
    sort: SortMode,
    source: Option<&str>,
    limit: usize,
) -> crate::Result<Vec<NodeResult>> {
    let safe_query = sanitize_fts5_query(query);
    let terms = extract_search_terms(&safe_query);
    let phrases = extract_quoted_phrases(&safe_query);
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let mut fetch_limit = compute_search_fetch_limit(limit, &terms, &phrases);
    let collapse = contains_risky_fts_ascii(query);
    let candidate_cap = compute_search_candidate_cap(limit);
    let now = now_secs();
    let mut offset = 0usize;
    let mut scanned = 0usize;
    let mut results: Vec<ScoredNode> = Vec::new();
    let mut source_cache: HashMap<i64, bool> = HashMap::new();
    loop {
        let rows = store.search_nodes_like(&terms, session, fetch_limit as i64, offset as i64)?;
        scanned += rows.len();
        let page_len = rows.len();
        for node in rows {
            if let Some(src) = source {
                if !store.node_matches_source(node.node_id, src, &mut source_cache)? {
                    continue;
                }
            }
            let mut score = 0usize;
            for term in &terms {
                let hits = count_term_matches(&node.summary, term);
                score += if collapse { hits.min(1) } else { hits };
            }
            if score == 0 {
                continue;
            }
            let directness = compute_directness_score(&node.summary, &terms, &phrases);
            results.push(ScoredNode {
                node,
                rank: -(score as f64),
                directness,
            });
        }
        results.sort_by(|a, b| {
            cmp_keys(
                &fallback_node_sort_key(a, sort, now),
                &fallback_node_sort_key(b, sort, now),
            )
        });
        if source.is_none() || page_len < fetch_limit || scanned >= candidate_cap {
            break;
        }
        offset += page_len;
        let remaining = candidate_cap - scanned;
        if remaining == 0 {
            break;
        }
        fetch_limit = (fetch_limit * 2).min(remaining);
    }
    results.truncate(limit);
    Ok(results.into_iter().map(node_result).collect())
}

fn message_result(s: ScoredMessage) -> MessageResult {
    MessageResult {
        row: s.row,
        rank: s.rank,
        snippet: s.snippet,
        directness: s.directness,
    }
}

fn node_result(s: ScoredNode) -> NodeResult {
    NodeResult {
        node: s.node,
        rank: s.rank,
        directness: s.directness,
    }
}

// ---- sort keys ----------------------------------------------------------------------------------

/// Compare two sort-key tuples lexicographically (Python tuple `<`).
fn cmp_keys<const N: usize>(a: &[f64; N], b: &[f64; N]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let ord = x.total_cmp(y);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Role bias for ranking (`_message_role_bias`, `LCM:store.py:83-90`): user rows outrank
/// assistant rows outrank tool rows at equal relevance.
fn message_role_bias(role: &str) -> f64 {
    match role {
        "user" => 0.0,
        "assistant" => 1.0,
        "tool" => 2.0,
        _ => 1.0,
    }
}

/// The message FTS sort key (`_fts_result_sort_key`, `LCM:store.py:145-159`).
fn fts_message_sort_key(m: &ScoredMessage, sort: SortMode, now: f64) -> [f64; 4] {
    let ts = m.row.timestamp;
    let bias = message_role_bias(&m.row.role);
    match sort {
        SortMode::Relevance => [m.rank, -m.directness, bias, -ts],
        SortMode::Hybrid => {
            let age_hours = ((now - ts) / 3600.0).max(0.0);
            [
                m.rank / (1.0 + age_hours * AGE_DECAY_RATE),
                -m.directness,
                bias,
                -ts,
            ]
        }
        SortMode::Recency => [-ts, bias, m.rank, 0.0],
    }
}

/// The message FTS "primary" strength used by the widening-ladder proof
/// (`_fts_primary_value`, `LCM:store.py:162-170`).
fn fts_message_primary(m: &ScoredMessage, sort: SortMode, now: f64) -> f64 {
    match sort {
        SortMode::Hybrid => {
            let age_hours = ((now - m.row.timestamp) / 3600.0).max(0.0);
            m.rank / (1.0 + age_hours * AGE_DECAY_RATE)
        }
        _ => m.rank,
    }
}

/// The message LIKE sort key (`_fallback_result_sort_key`, `LCM:store.py:129-142`);
/// `score = -rank`.
fn fallback_message_sort_key(m: &ScoredMessage, sort: SortMode, now: f64) -> [f64; 4] {
    let score = -m.rank;
    let ts = m.row.timestamp;
    let bias = message_role_bias(&m.row.role);
    match sort {
        SortMode::Relevance => [-score, -m.directness, bias, -ts],
        SortMode::Hybrid => {
            let age_hours = ((now - ts) / 3600.0).max(0.0);
            [
                -(score / (1.0 + age_hours * AGE_DECAY_RATE)),
                -m.directness,
                bias,
                -ts,
            ]
        }
        SortMode::Recency => [-ts, bias, -score, -m.directness],
    }
}

/// A node's recency instant: `latest_at or created_at` with Python truthiness (0.0 falls
/// through), per `LCM:dag.py:66`.
fn node_recency(node: &SummaryNode) -> f64 {
    match node.latest_at {
        Some(v) if v != 0.0 => v,
        _ => node.created_at,
    }
}

/// The node FTS sort key (`_fts_result_sort_key`, `LCM:dag.py:78-92`).
fn fts_node_sort_key(n: &ScoredNode, sort: SortMode, now: f64) -> [f64; 3] {
    let recency = node_recency(&n.node);
    match sort {
        SortMode::Relevance => [n.rank, -n.directness, -recency],
        SortMode::Hybrid => {
            let age_hours = ((now - recency) / 3600.0).max(0.0);
            let blended_strength = (-n.rank) / (1.0 + age_hours * AGE_DECAY_RATE);
            [-blended_strength, -n.directness, -recency]
        }
        SortMode::Recency => [-recency, n.rank, 0.0],
    }
}

/// The node FTS primary strength (`_fts_primary_value`, `LCM:dag.py:95-105`).
fn fts_node_primary(rank: f64, recency: f64, sort: SortMode, now: f64) -> f64 {
    match sort {
        SortMode::Hybrid => {
            let age_hours = ((now - recency) / 3600.0).max(0.0);
            rank / (1.0 + age_hours * AGE_DECAY_RATE)
        }
        _ => rank,
    }
}

/// The node LIKE sort key (`_fallback_result_sort_key`, `LCM:dag.py:63-75`); `score = -rank`.
fn fallback_node_sort_key(n: &ScoredNode, sort: SortMode, now: f64) -> [f64; 3] {
    let score = -n.rank;
    let recency = node_recency(&n.node);
    match sort {
        SortMode::Relevance => [-score, -n.directness, -recency],
        SortMode::Hybrid => {
            let age_hours = ((now - recency) / 3600.0).max(0.0);
            [
                -(score / (1.0 + age_hours * AGE_DECAY_RATE)),
                -n.directness,
                -recency,
            ]
        }
        SortMode::Recency => [-recency, -score, -n.directness],
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---- query shaping (ports of LCM:search_query.py) ------------------------------------------------

/// Strip FTS5 syntax operators while preserving balanced phrase quotes
/// (`sanitize_fts5_query`, `LCM:search_query.py:37-65`). Character-preserving: specials become
/// spaces, balanced `"…"` regions are kept verbatim (re-quoted), a trailing unbalanced quote is
/// dropped and its fragment sanitized like bare text.
pub fn sanitize_fts5_query(query: &str) -> String {
    if query.is_empty() {
        return String::new();
    }
    let mut result: Vec<char> = Vec::new();
    let mut quote_buffer: Vec<char> = Vec::new();
    let mut in_quote = false;
    for ch in query.chars() {
        if ch == '"' {
            if in_quote {
                result.push('"');
                result.append(&mut quote_buffer);
                result.push('"');
                in_quote = false;
            } else {
                if result.last().is_some_and(|c| !c.is_whitespace()) {
                    result.push(' ');
                }
                in_quote = true;
                quote_buffer.clear();
            }
            continue;
        }
        if in_quote {
            quote_buffer.push(ch);
            continue;
        }
        result.push(if FTS5_SPECIAL_CHARS.contains(&ch) {
            ' '
        } else {
            ch
        });
    }
    if in_quote && !quote_buffer.is_empty() {
        for ch in quote_buffer {
            result.push(if FTS5_SPECIAL_CHARS.contains(&ch) {
                ' '
            } else {
                ch
            });
        }
    }
    result.into_iter().collect::<String>().trim().to_string()
}

/// The byte spans of every `"([^"]+)"` match (Python's `_QUOTED_PHRASE_RE`), as
/// `(start, end, inner)` with regex-faithful scanning: an immediately-following quote does not
/// close an empty phrase — it becomes the next candidate opener.
fn quoted_phrase_spans(text: &str) -> Vec<(usize, usize, &str)> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let Some(rel) = bytes[i + 1..].iter().position(|&b| b == b'"') else {
            break;
        };
        let j = i + 1 + rel;
        if j == i + 1 {
            i = j;
            continue;
        }
        spans.push((i, j + 1, &text[i + 1..j]));
        i = j + 1;
    }
    spans
}

/// Replace every balanced `"…"` region (quotes included) with a single space, like
/// `_QUOTED_PHRASE_RE.sub(" ", text)`.
fn strip_quoted_phrases(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev = 0;
    for (start, end, _) in quoted_phrase_spans(text) {
        out.push_str(&text[prev..start]);
        out.push(' ');
        prev = end;
    }
    out.push_str(&text[prev..]);
    out
}

/// The trimmed contents of every balanced `"…"` phrase, case-preserving
/// (`extract_quoted_phrases`, `LCM:search_query.py:144-145`).
pub fn extract_quoted_phrases(query: &str) -> Vec<String> {
    quoted_phrase_spans(query)
        .into_iter()
        .filter_map(|(_, _, inner)| {
            let trimmed = inner.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .collect()
}

/// Whether the text contains CJK codepoints the `unicode61` tokenizer drops
/// (`_CJK_RE`, `LCM:search_query.py:8-17`).
pub fn contains_cjk(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c as u32,
            0x3400..=0x4DBF   // CJK Unified Ideographs Extension A
            | 0x4E00..=0x9FFF // CJK Unified Ideographs
            | 0x3000..=0x303F // CJK Symbols and Punctuation
            | 0x3040..=0x30FF // Hiragana + Katakana
            | 0xAC00..=0xD7AF // Hangul Syllables
            | 0xFF00..=0xFFEF // Halfwidth and Fullwidth Forms
        )
    })
}

/// Whether the text contains emoji/symbol codepoints (`_EMOJI_RE`, `LCM:search_query.py:18-23`).
pub fn contains_emoji(text: &str) -> bool {
    text.chars()
        .any(|c| matches!(c as u32, 0x2600..=0x27BF | 0x1F300..=0x1FAFF))
}

/// Whether ASCII text has a shape FTS5 would mis-tokenize (`contains_risky_fts_ascii`,
/// `LCM:search_query.py:79-86`): unbalanced quotes, or `alnum[-:/]alnum` outside quoted phrases.
pub fn contains_risky_fts_ascii(text: &str) -> bool {
    let raw = text.trim();
    if raw.is_empty() {
        return false;
    }
    if raw.matches('"').count() % 2 == 1 {
        return true;
    }
    let without_phrases = strip_quoted_phrases(raw);
    let chars: Vec<char> = without_phrases.chars().collect();
    chars.windows(3).any(|w| {
        w[0].is_ascii_alphanumeric()
            && matches!(w[1], '-' | ':' | '/')
            && w[2].is_ascii_alphanumeric()
    })
}

/// Whether a query must use the LIKE fallback instead of FTS5
/// (`requires_like_fallback`, `LCM:search_query.py:89-90`).
pub fn requires_like_fallback(query: &str) -> bool {
    contains_cjk(query) || contains_emoji(query) || contains_risky_fts_ascii(query)
}

/// One bare token's search variants (`_token_variants`, `LCM:search_query.py:93-112`): edge
/// punctuation stripped, boolean operators dropped, and `-`/`:`/`/` compounds contributing both
/// the whole token and its parts.
fn token_variants(token: &str) -> Vec<String> {
    let cleaned = token.trim().trim_matches(STRIP_EDGE_PUNCT);
    if cleaned.is_empty() {
        return Vec::new();
    }
    if BOOLEAN_OPERATORS.contains(&cleaned.to_uppercase().as_str()) {
        return Vec::new();
    }
    let mut variants = vec![cleaned.to_string()];
    if cleaned.contains(['-', ':', '/']) {
        let parts: Vec<&str> = cleaned
            .split(['-', ':', '/'])
            .filter(|p| !p.is_empty())
            .collect();
        if parts.len() > 1 {
            variants.extend(parts.into_iter().map(String::from));
        }
    }
    dedup(variants)
}

/// Case-preserving, de-duplicated search terms (`extract_search_terms`,
/// `LCM:search_query.py:115-141`): quoted phrases first, then bare-token variants, then the whole
/// (punctuation-stripped) query as a last resort. Callers pass the *sanitized* query.
pub fn extract_search_terms(query: &str) -> Vec<String> {
    let text = query.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let mut terms = extract_quoted_phrases(text);
    let without_phrases = strip_quoted_phrases(text);
    for token in without_phrases.split_whitespace() {
        terms.extend(token_variants(token));
    }
    if terms.is_empty() {
        let fallback = text.trim_matches(STRIP_EDGE_PUNCT);
        if !fallback.is_empty() {
            terms.push(fallback.to_string());
        }
    }
    dedup(terms)
}

/// Case-insensitive non-overlapping occurrence count (`count_term_matches`,
/// `LCM:search_query.py:152-157`).
pub fn count_term_matches(text: &str, term: &str) -> usize {
    if text.is_empty() || term.is_empty() {
        return 0;
    }
    text.to_lowercase()
        .matches(term.to_lowercase().as_str())
        .count()
}

/// The directness score (`compute_directness_score`, `LCM:search_query.py:160-218`): distinct
/// term hits ×5 and phrase hits ×8, minus a capped repetition penalty (phrase repeats excluded
/// when phrases are present) and a phrase-stuffing analysis that penalizes repeated phrases
/// separated by little or no fresh prose.
pub fn compute_directness_score(text: &str, terms: &[String], phrases: &[String]) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let normalized_phrases: HashSet<String> = phrases
        .iter()
        .map(|p| p.trim().to_lowercase())
        .filter(|p| !p.is_empty())
        .collect();

    let mut unique_hits = 0i64;
    let mut total_hits = 0i64;
    let mut non_phrase_unique_hits = 0i64;
    let mut non_phrase_total_hits = 0i64;
    for term in terms {
        let matches = count_term_matches(text, term) as i64;
        if matches > 0 {
            unique_hits += 1;
            total_hits += matches;
            if !normalized_phrases.contains(&term.trim().to_lowercase()) {
                non_phrase_unique_hits += 1;
                non_phrase_total_hits += matches;
            }
        }
    }

    let lowered = text.to_lowercase();
    let mut phrase_hits = 0i64;
    for phrase in phrases {
        if !phrase.is_empty() && lowered.contains(phrase.to_lowercase().as_str()) {
            phrase_hits += 1;
        }
    }

    let repetition_penalty = (total_hits - unique_hits).max(0);
    let non_phrase_repetition_penalty = (non_phrase_total_hits - non_phrase_unique_hits).max(0);
    let mut score = (unique_hits * 5 + phrase_hits * 8) as f64;
    score -= if phrases.is_empty() {
        repetition_penalty.min(6)
    } else {
        non_phrase_repetition_penalty.min(6)
    } as f64;

    // Phrase-stuffing analysis: repeated phrase occurrences separated by thin gaps
    // (`LCM:search_query.py:193-216`).
    for phrase in phrases {
        let normalized_phrase = phrase.trim().to_lowercase();
        if normalized_phrase.is_empty() {
            continue;
        }
        let occurrences = lowered.matches(normalized_phrase.as_str()).count();
        if occurrences <= 1 {
            continue;
        }
        let gap_unique_counts: Vec<usize> = lowered
            .split(normalized_phrase.as_str())
            .map(distinct_alpha_word_tokens)
            .collect();
        let interior = &gap_unique_counts[1..gap_unique_counts.len().saturating_sub(1).max(1)];
        let tail = gap_unique_counts.last().copied().unwrap_or(0);
        let extra_occurrences = occurrences - 1;
        score -= extra_occurrences as f64 * 0.5;
        score -= interior.iter().filter(|&&c| c > 0 && c <= 4).count() as f64 * 1.5;
        if interior.iter().all(|&c| c == 0) && tail <= 2 {
            score -= (extra_occurrences as f64).min(3.0);
        }
    }

    score
}

/// Count distinct `[\w-]+` tokens containing at least one alphabetic char (the phrase-gap
/// analysis unit in `LCM:search_query.py:203-209`).
fn distinct_alpha_word_tokens(segment: &str) -> usize {
    let mut seen: HashSet<&str> = HashSet::new();
    for token in segment.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-')) {
        if !token.is_empty() && token.chars().any(char::is_alphabetic) {
            seen.insert(token);
        }
    }
    seen.len()
}

/// A message's directness (`_message_directness_score`, `LCM:store.py:93-99`): the text score,
/// with raw JSON tool payloads knocked down by 4.
fn message_directness_score(
    role: &str,
    content: Option<&str>,
    terms: &[String],
    phrases: &[String],
) -> f64 {
    let content = content.unwrap_or("");
    let mut score = compute_directness_score(content, terms, phrases);
    if role == "tool" {
        let stripped = content.trim_start();
        if stripped.starts_with('{') || stripped.starts_with('[') {
            score -= 4.0;
        }
    }
    score
}

/// Whether the query shape is precise enough for widening + the directness rank nudge
/// (`_is_precise_query_shape`, `LCM:search_query.py:221-224`): one term, or one phrase with at
/// most two terms.
fn is_precise_query_shape(terms: &[String], phrases: &[String]) -> bool {
    terms.len() == 1 || (phrases.len() == 1 && terms.len() <= 2)
}

/// `should_apply_directness_rank_adjustment` (`LCM:search_query.py:231-232`).
fn should_apply_directness_rank_adjustment(terms: &[String], phrases: &[String]) -> bool {
    is_precise_query_shape(terms, phrases)
}

/// The largest possible directness bonus for this query
/// (`compute_directness_rank_bonus_upper_bound`, `LCM:search_query.py:235-236`).
fn compute_directness_rank_bonus_upper_bound(terms: &[String], phrases: &[String]) -> f64 {
    (terms.len() * 5 + phrases.len() * 8) as f64
}

/// The initial candidate page size (`compute_search_fetch_limit`,
/// `LCM:search_query.py:239-243`): 5× the limit (min 20), widened to 10× (min 50) for precise
/// shapes.
fn compute_search_fetch_limit(limit: usize, terms: &[String], phrases: &[String]) -> usize {
    let base = limit.saturating_mul(5).max(limit).max(20);
    if is_precise_query_shape(terms, phrases) {
        base.max(limit.saturating_mul(10)).max(50)
    } else {
        base
    }
}

/// The hard cap on candidate rows inspected per search call
/// (`compute_search_candidate_cap`, `LCM:search_query.py:251-253`).
fn compute_search_candidate_cap(limit: usize) -> usize {
    limit.saturating_mul(20).max(limit).clamp(500, 5000)
}

/// An ~80-char snippet around the first matching term (`build_snippet`,
/// `LCM:search_query.py:265-283`), with `...` continuation markers; no hit => a leading window.
pub fn build_snippet(text: &str, terms: &[String]) -> String {
    if text.is_empty() {
        return String::new();
    }
    let content: Vec<char> = text.chars().collect();
    let lowered: Vec<char> = text.to_lowercase().chars().collect();
    for term in terms {
        if term.is_empty() {
            continue;
        }
        let needle: Vec<char> = term.to_lowercase().chars().collect();
        let Some(idx) = find_char_subslice(&lowered, &needle) else {
            continue;
        };
        let start = idx.saturating_sub(SNIPPET_WIDTH / 2).min(content.len());
        let end = (idx + needle.len() + SNIPPET_WIDTH / 2).clamp(start, content.len());
        let mut snippet: String = content[start..end].iter().collect();
        if start > 0 {
            snippet = format!("...{snippet}");
        }
        if end < content.len() {
            snippet.push_str("...");
        }
        return snippet;
    }
    let head: String = content.iter().take(SNIPPET_WIDTH).collect();
    if content.len() > SNIPPET_WIDTH {
        format!("{head}...")
    } else {
        head
    }
}

fn find_char_subslice(haystack: &[char], needle: &[char]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn dedup(v: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(v.len());
    for t in v {
        if seen.insert(t.clone()) {
            out.push(t);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{NewMessage, NewNode, SourceType, Store};

    // ---- sanitize (LCM:search_query.py:37-65) ----------------------------------------------

    #[test]
    fn sanitize_preserves_text_and_balanced_phrases() {
        assert_eq!(sanitize_fts5_query("hello world"), "hello world");
        assert_eq!(
            sanitize_fts5_query("\"exact phrase\" extra"),
            "\"exact phrase\" extra"
        );
        // Specials become spaces (char-preserving, not token-quoting).
        assert_eq!(sanitize_fts5_query("foo* OR bar^"), "foo  OR bar");
        assert!(sanitize_fts5_query("***").is_empty());
        // A trailing unbalanced quote is dropped; its fragment is sanitized like bare text.
        assert_eq!(
            sanitize_fts5_query("he said \"unclosed*rest"),
            "he said unclosed rest"
        );
        // Opening a phrase mid-token inserts the separating space.
        assert_eq!(
            sanitize_fts5_query("mid\"quoted\"tail"),
            "mid \"quoted\"tail"
        );
        // An immediately-closed quote pair survives as an empty phrase (Python parity), while
        // phrase *extraction* skips it: `"([^"]+)"` treats the second quote as the next opener.
        assert_eq!(sanitize_fts5_query("\"\"x\"y\""), "\"\"x \"y\"");
        assert_eq!(extract_quoted_phrases("\"\"x\"y\""), vec!["x".to_string()]);
    }

    // ---- fallback routing (LCM:search_query.py:71-90) --------------------------------------

    #[test]
    fn like_fallback_detects_cjk_emoji_risky_and_unbalanced() {
        assert!(requires_like_fallback("検索クエリ"));
        assert!(requires_like_fallback("party 🎉 time"));
        assert!(requires_like_fallback("foo-bar"));
        assert!(requires_like_fallback("a:b"));
        assert!(requires_like_fallback("path/to"));
        assert!(requires_like_fallback("\"unbalanced"));
        assert!(!requires_like_fallback("plain ascii words"));
        assert!(!requires_like_fallback("\"balanced phrase\""));
        // Risky ASCII inside a balanced phrase is protected by the quotes.
        assert!(!requires_like_fallback("\"foo-bar\""));
        // The Python CJK ranges include symbols/punctuation and full-width forms.
        assert!(contains_cjk("。"));
        assert!(contains_cjk("ＡＢＣ"));
        assert!(contains_cjk("한글"));
    }

    // ---- terms/phrases (LCM:search_query.py:93-145) ----------------------------------------

    #[test]
    fn terms_are_case_preserving_with_compound_variants() {
        // No lowercasing; dedup is exact.
        assert_eq!(
            extract_search_terms("Foo BAR foo"),
            vec!["Foo", "BAR", "foo"]
        );
        // Compound tokens contribute the whole and the parts.
        assert_eq!(
            extract_search_terms("foo-bar baz"),
            vec!["foo-bar", "foo", "bar", "baz"]
        );
        // Boolean operators are dropped from terms.
        assert_eq!(extract_search_terms("foo OR bar"), vec!["foo", "bar"]);
        // Phrases come first, then bare tokens.
        assert_eq!(
            extract_search_terms("\"Quoted Phrase\" extra"),
            vec!["Quoted Phrase", "extra"]
        );
        assert_eq!(
            extract_quoted_phrases("a \"Quoted Phrase\" b"),
            vec!["Quoted Phrase".to_string()]
        );
        // All-punctuation queries fall back to the stripped whole query, or nothing.
        assert!(extract_search_terms("...").is_empty());
        assert_eq!(extract_search_terms("(AND)"), vec!["AND"]);
    }

    // ---- directness (LCM:search_query.py:160-218) -------------------------------------------

    #[test]
    fn directness_rewards_hits_and_penalizes_repetition() {
        let terms = vec!["alpha".to_string(), "beta".to_string()];
        let phrases: Vec<String> = vec![];
        let both = compute_directness_score("alpha beta", &terms, &phrases);
        let one = compute_directness_score("alpha only", &terms, &phrases);
        assert_eq!(both, 10.0);
        assert_eq!(one, 5.0);
        let repeated = compute_directness_score("alpha alpha alpha beta", &terms, &phrases);
        assert_eq!(repeated, 8.0, "two repeats cost two points");
        // Matching is case-insensitive.
        assert_eq!(
            compute_directness_score("ALPHA Beta", &terms, &phrases),
            10.0
        );
    }

    #[test]
    fn directness_penalizes_phrase_stuffing() {
        let terms = vec!["hello world".to_string()];
        let phrases = vec!["hello world".to_string()];
        let single = compute_directness_score("hello world", &terms, &phrases);
        assert_eq!(single, 13.0, "5 for the term + 8 for the phrase");
        // Back-to-back repeats with no fresh prose: -0.5 per extra occurrence, -1.0 stuffing.
        let stuffed = compute_directness_score("hello world hello world", &terms, &phrases);
        assert_eq!(stuffed, 11.5);
        assert!(stuffed < single);
        // A thin interior gap (1-4 fresh words) costs 1.5.
        let thin_gap =
            compute_directness_score("hello world and then hello world", &terms, &phrases);
        assert_eq!(thin_gap, 11.0);
    }

    // ---- fetch limits (LCM:search_query.py:227-253) -----------------------------------------

    #[test]
    fn fetch_limits_widen_for_precise_shapes() {
        let one_term = vec!["a".to_string()];
        let many: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let none: Vec<String> = vec![];
        assert!(is_precise_query_shape(&one_term, &none));
        assert!(!is_precise_query_shape(&many, &none));
        assert_eq!(compute_search_fetch_limit(10, &one_term, &none), 100);
        assert_eq!(compute_search_fetch_limit(10, &many, &none), 50);
        assert_eq!(compute_search_fetch_limit(1, &many, &none), 20);
        assert_eq!(compute_search_candidate_cap(10), 500);
        assert_eq!(compute_search_candidate_cap(300), 5000);
    }

    // ---- snippets (LCM:search_query.py:265-283) ---------------------------------------------

    #[test]
    fn snippet_windows_around_first_hit() {
        let content = format!("{}needle{}", "x".repeat(200), "y".repeat(200));
        let snip = build_snippet(&content, &["needle".to_string()]);
        assert!(snip.contains("needle"));
        assert!(snip.starts_with("...") && snip.ends_with("..."));
        // No hit: a leading window with a continuation marker.
        let lead = build_snippet(&"z".repeat(100), &["absent".to_string()]);
        assert_eq!(lead, format!("{}...", "z".repeat(80)));
    }

    // ---- integration over a real store -------------------------------------------------------

    fn seed(store: &Store, session: &str) {
        let base = 1_000.0;
        let rows = [
            ("user", "the quick brown fox", base + 1.0),
            ("assistant", "a slow brown bear sleeps", base + 2.0),
            ("user", "quick thoughts on foxes and bears", base + 100.0),
        ];
        for (role, content, ts) in rows {
            store
                .append_batch(
                    session,
                    &[NewMessage {
                        role: role.into(),
                        content: Some(content.into()),
                        ..Default::default()
                    }],
                    ts,
                )
                .unwrap();
        }
    }

    #[test]
    fn recency_vs_relevance_ordering() {
        let store = Store::open_in_memory().unwrap();
        seed(&store, "s1");
        let filter = MessageFilter {
            session: Some("s1"),
            ..Default::default()
        };
        // "quick" hits rows 1 and 3; recency surfaces the newest (row 3) first.
        let recency = search_messages(&store, "quick", SortMode::Recency, &filter, 10).unwrap();
        assert_eq!(recency.len(), 2);
        assert!(recency[0].row.timestamp > recency[1].row.timestamp);
        // The FTS path carries the SQL snippet markers.
        assert!(recency[0].snippet.contains(">>>quick<<<"));
    }

    #[test]
    fn recency_breaks_timestamp_ties_by_role_bias() {
        let store = Store::open_in_memory().unwrap();
        for role in ["tool", "user", "assistant"] {
            store
                .append_batch(
                    "s1",
                    &[NewMessage {
                        role: role.into(),
                        content: Some("shared needle content".into()),
                        ..Default::default()
                    }],
                    500.0,
                )
                .unwrap();
        }
        let filter = MessageFilter {
            session: Some("s1"),
            ..Default::default()
        };
        let hits = search_messages(&store, "needle", SortMode::Recency, &filter, 10).unwrap();
        let roles: Vec<&str> = hits.iter().map(|h| h.row.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool"]);
    }

    #[test]
    fn like_fallback_finds_cjk_and_scores_hits() {
        let store = Store::open_in_memory().unwrap();
        store
            .append_batch(
                "s1",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("検索 テスト 検索".into()),
                    ..Default::default()
                }],
                1.0,
            )
            .unwrap();
        let filter = MessageFilter {
            session: Some("s1"),
            ..Default::default()
        };
        let hits = search_messages(&store, "検索", SortMode::Recency, &filter, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rank, -2.0, "two occurrences => rank -2");
        assert!(hits[0].snippet.contains("検索"));
    }

    #[test]
    fn risky_ascii_collapses_repeats_in_like_scoring() {
        let store = Store::open_in_memory().unwrap();
        // "foo-bar" is risky ASCII: it routes to LIKE, sanitize turns `-` into a space
        // (terms [foo, bar]), and repeat hits collapse to once per term.
        store
            .append_batch(
                "s1",
                &[
                    NewMessage {
                        role: "user".into(),
                        content: Some("foo foo foo foo".into()),
                        ..Default::default()
                    },
                    NewMessage {
                        role: "user".into(),
                        content: Some("foo-bar setup".into()),
                        ..Default::default()
                    },
                ],
                1.0,
            )
            .unwrap();
        let filter = MessageFilter {
            session: Some("s1"),
            ..Default::default()
        };
        let hits = search_messages(&store, "foo-bar", SortMode::Relevance, &filter, 10).unwrap();
        assert_eq!(hits.len(), 2);
        // The compound row hits both terms (score 2); the spam row collapses to one "foo" hit.
        assert_eq!(hits[0].row.content.as_deref(), Some("foo-bar setup"));
        assert_eq!(hits[0].rank, -2.0);
        assert_eq!(hits[1].rank, -1.0);
    }

    fn node(session: &str, depth: i64, summary: &str, sources: &[i64], st: SourceType) -> NewNode {
        NewNode {
            session_id: session.into(),
            depth,
            summary: summary.into(),
            token_count: 10,
            source_token_count: 100,
            source_ids: sources.to_vec(),
            source_type: st,
            created_at: 1_000.0,
            earliest_at: Some(900.0),
            latest_at: Some(1_100.0),
            expand_hint: "hint".into(),
        }
    }

    #[test]
    fn node_search_filters_by_source_lineage() {
        let store = Store::open_in_memory().unwrap();
        let ids = store
            .append_batch(
                "s1",
                &[
                    NewMessage {
                        role: "user".into(),
                        source: "telegram".into(),
                        content: Some("alpha message".into()),
                        ..Default::default()
                    },
                    NewMessage {
                        role: "user".into(),
                        source: "matrix".into(),
                        content: Some("beta message".into()),
                        ..Default::default()
                    },
                ],
                1.0,
            )
            .unwrap();
        let leaf_tg = store
            .add_node(&node(
                "s1",
                0,
                "compaction covers telegram traffic",
                &ids[..1],
                SourceType::Messages,
            ))
            .unwrap();
        let leaf_mx = store
            .add_node(&node(
                "s1",
                0,
                "compaction covers matrix traffic",
                &ids[1..],
                SourceType::Messages,
            ))
            .unwrap();
        let parent = store
            .add_node(&node(
                "s1",
                1,
                "compaction rollup of both platforms",
                &[leaf_tg],
                SourceType::Nodes,
            ))
            .unwrap();

        // Unfiltered: all three nodes match "compaction".
        let all = search_nodes(
            &store,
            "compaction",
            Some("s1"),
            SortMode::Recency,
            None,
            10,
        )
        .unwrap();
        assert_eq!(all.len(), 3);
        // source=telegram: the telegram leaf + the parent whose lineage reaches it.
        let tg = search_nodes(
            &store,
            "compaction",
            Some("s1"),
            SortMode::Recency,
            Some("telegram"),
            10,
        )
        .unwrap();
        let tg_ids: Vec<i64> = tg.iter().map(|n| n.node.node_id).collect();
        assert!(tg_ids.contains(&leaf_tg) && tg_ids.contains(&parent));
        assert!(!tg_ids.contains(&leaf_mx));
        // source=matrix: only the matrix leaf (the parent's lineage does not reach it).
        let mx = search_nodes(
            &store,
            "compaction",
            Some("s1"),
            SortMode::Recency,
            Some("matrix"),
            10,
        )
        .unwrap();
        let mx_ids: Vec<i64> = mx.iter().map(|n| n.node.node_id).collect();
        assert_eq!(mx_ids, vec![leaf_mx]);
    }

    #[test]
    fn node_like_fallback_finds_cjk_summaries() {
        let store = Store::open_in_memory().unwrap();
        store
            .add_node(&node(
                "s1",
                0,
                "検索 モジュールの要約",
                &[1],
                SourceType::Messages,
            ))
            .unwrap();
        let hits = search_nodes(&store, "検索", Some("s1"), SortMode::Recency, None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rank, -1.0);
    }

    // ---- Wave 2 theme 1: search ranking at scale (role bias, batch caps, limit stability) -----

    /// Append one message with an explicit timestamp; returns its store_id.
    fn append_one(store: &Store, session: &str, role: &str, content: &str, ts: f64) -> i64 {
        store
            .append_batch(
                session,
                &[NewMessage {
                    role: role.into(),
                    content: Some(content.into()),
                    ..Default::default()
                }],
                ts,
            )
            .unwrap()[0]
    }

    // PARITY: hermes-lcm tests/test_lcm_core.py::test_search_relevance_prefers_user_over_newer_assistant_on_similar_match
    #[test]
    fn relevance_prefers_user_over_newer_assistant_on_similar_match() {
        let store = Store::open_in_memory().unwrap();
        // The user hit is older; the assistant hit is newer with identical content.
        let user_id = append_one(
            &store,
            "sess1",
            "user",
            "vendoring should stay external plugin host support only",
            1.0,
        );
        let assistant_id = append_one(
            &store,
            "sess1",
            "assistant",
            "vendoring should stay external plugin host support only",
            2.0,
        );
        let filter = MessageFilter {
            session: Some("sess1"),
            ..Default::default()
        };
        let results =
            search_messages(&store, "vendoring", SortMode::Relevance, &filter, 2).unwrap();
        assert_eq!(results[0].row.store_id, user_id);
        assert_eq!(results[1].row.store_id, assistant_id);
    }

    // PARITY: hermes-lcm tests/test_lcm_core.py::test_search_relevance_does_not_let_weaker_user_hit_beat_stronger_assistant_hit
    #[test]
    fn relevance_does_not_let_weaker_user_hit_beat_stronger_assistant_hit() {
        let store = Store::open_in_memory().unwrap();
        let weaker_user_id = append_one(
            &store,
            "sess1",
            "user",
            "vendoring blah blah external blah host",
            1.0,
        );
        let stronger_assistant_id =
            append_one(&store, "sess1", "assistant", "vendoring external host", 2.0);
        let filter = MessageFilter {
            session: Some("sess1"),
            ..Default::default()
        };
        let results = search_messages(
            &store,
            "vendoring external host",
            SortMode::Relevance,
            &filter,
            2,
        )
        .unwrap();
        assert_eq!(results[0].row.store_id, stronger_assistant_id);
        assert_eq!(results[1].row.store_id, weaker_user_id);
    }

    // PARITY: hermes-lcm tests/test_lcm_core.py::test_search_relevance_still_surfaces_preferred_user_hit_from_large_same_rank_pool
    #[test]
    fn relevance_surfaces_preferred_user_hit_from_large_same_rank_pool() {
        let store = Store::open_in_memory().unwrap();
        let preferred_user_id = append_one(&store, "sess1", "user", "vendoring", 1.0);
        for _ in 0..150 {
            append_one(&store, "sess1", "assistant", "vendoring", 2.0);
        }
        let filter = MessageFilter {
            session: Some("sess1"),
            ..Default::default()
        };
        let results =
            search_messages(&store, "vendoring", SortMode::Relevance, &filter, 5).unwrap();
        assert_eq!(results[0].row.store_id, preferred_user_id);
        assert_eq!(results[0].row.role, "user");
    }

    // PARITY: hermes-lcm tests/test_lcm_core.py::test_search_relevance_top_results_do_not_change_when_limit_increases_on_large_single_term_pool
    #[test]
    fn relevance_top_results_stable_when_limit_increases_on_large_pool() {
        let store = Store::open_in_memory().unwrap();
        append_one(&store, "sess1", "user", "vendoring", 1.0);
        let mut batch: Vec<NewMessage> = Vec::new();
        for idx in 0..250 {
            let (role, content) = if idx % 5 == 0 {
                ("tool", "{\"vendoring\":\"vendoring vendoring vendoring\"}")
            } else {
                (
                    "assistant",
                    "vendoring vendoring vendoring vendoring vendoring spam",
                )
            };
            batch.push(NewMessage {
                role: role.into(),
                content: Some(content.into()),
                ..Default::default()
            });
        }
        store.append_batch("sess1", &batch, 2.0).unwrap();
        let filter = MessageFilter {
            session: Some("sess1"),
            ..Default::default()
        };
        let top_5: Vec<i64> = search_messages(&store, "vendoring", SortMode::Relevance, &filter, 5)
            .unwrap()
            .iter()
            .map(|r| r.row.store_id)
            .collect();
        let top_50: Vec<i64> =
            search_messages(&store, "vendoring", SortMode::Relevance, &filter, 50)
                .unwrap()
                .iter()
                .take(5)
                .map(|r| r.row.store_id)
                .collect();
        assert_eq!(top_5, top_50);
    }

    // PARITY: hermes-lcm tests/test_lcm_core.py::test_search_recency_same_timestamp_pool_is_limit_stable
    #[test]
    fn recency_same_timestamp_pool_is_limit_stable() {
        let store = Store::open_in_memory().unwrap();
        let mut batch: Vec<NewMessage> = Vec::new();
        for idx in 0..120 {
            batch.push(NewMessage {
                role: "assistant".into(),
                content: Some(format!(
                    "alpha alpha alpha beta beta gamma gamma gamma spam {idx}"
                )),
                ..Default::default()
            });
        }
        batch.push(NewMessage {
            role: "assistant".into(),
            content: Some("keep alpha beta gamma concise".into()),
            ..Default::default()
        });
        // A single batch timestamp reproduces the Python same-timestamp pool.
        store.append_batch("sess1", &batch, 500.0).unwrap();
        let filter = MessageFilter {
            session: Some("sess1"),
            ..Default::default()
        };
        let short =
            search_messages(&store, "alpha beta gamma", SortMode::Recency, &filter, 5).unwrap();
        let long =
            search_messages(&store, "alpha beta gamma", SortMode::Recency, &filter, 200).unwrap();
        assert!(short.iter().all(|r| r.row.timestamp == 500.0));
        let short_ids: Vec<i64> = short.iter().map(|r| r.row.store_id).collect();
        let long_ids: Vec<i64> = long.iter().take(5).map(|r| r.row.store_id).collect();
        assert_eq!(short_ids, long_ids);
    }
}
