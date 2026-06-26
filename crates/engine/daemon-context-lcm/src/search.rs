//! The transcript/DAG search stack (`daemon-context-lcm-port-spec.md` §11).
//!
//! Port of hermes-lcm's search layer over the SQLite FTS5 indexes:
//!
//! * **FTS5 sanitize** (§11.1) — turn arbitrary user text into a syntactically safe MATCH
//!   expression, preserving `"quoted phrases"`.
//! * **LIKE fallback** (§11.2) — FTS5's `unicode61` tokenizer drops CJK/emoji and chokes on a few
//!   risky ASCII shapes, so those route to a substring (`LIKE`) scan scored in Rust.
//! * **Sort modes + directness + snippets** (§11.3) — `recency` / `relevance` / `hybrid`, with a
//!   tiny directness nudge for precise query shapes and an 80-char snippet around the first hit.
//!
//! Query-shaping lives here; the raw SQL (FTS MATCH / LIKE / node MATCH) lives in [`crate::store`].
//! Ranks follow FTS5 convention: **lower is better**. LIKE candidates synthesize a rank as
//! `-(directness)` so more/closer hits sort first under the same comparators.

use crate::store::{MessageFilter, MessageRow, NodeHit, Store, SummaryNode};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hybrid-sort age decay (§11.3): older rows divide their rank by `1 + age_hours * RATE`.
const AGE_DECAY_RATE: f64 = 0.001;
/// Directness nudge weight for message ranks (precise shapes only).
const MESSAGE_DIRECTNESS_WEIGHT: f64 = 3e-7;
/// Directness nudge weight for node ranks (precise shapes only).
const NODE_DIRECTNESS_WEIGHT: f64 = 2e-7;
/// Snippet half-window is derived from this total width (§11.3).
const SNIPPET_WINDOW: usize = 80;

/// How matches are ordered (§11.3). Anything unrecognized normalizes to [`SortMode::Recency`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SortMode {
    /// Newest first, FTS rank as the tie-break.
    #[default]
    Recency,
    /// Best FTS rank first, newest as the tie-break.
    Relevance,
    /// Rank discounted by age (recent strong matches win).
    Hybrid,
}

impl SortMode {
    /// Parse a caller-supplied sort string; unknown values fall back to `Recency` (§11.3).
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
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

/// A scored transcript match: the row, its (possibly nudged) FTS rank, and a context snippet.
#[derive(Clone, Debug)]
pub struct MessageResult {
    /// The matched message row.
    pub row: MessageRow,
    /// The final rank used for ordering (lower = better).
    pub rank: f64,
    /// An ~80-char snippet around the first matching term.
    pub snippet: String,
}

/// A scored summary-node match.
#[derive(Clone, Debug)]
pub struct NodeResult {
    /// The matched summary node.
    pub node: SummaryNode,
    /// The final rank used for ordering (lower = better).
    pub rank: f64,
}

/// Search the lossless message transcript (§11). Picks the FTS or LIKE path, applies the directness
/// nudge for precise queries, orders by `sort`, truncates to `limit`, and attaches snippets.
pub fn search_messages(
    store: &Store,
    query: &str,
    sort: SortMode,
    filter: &MessageFilter<'_>,
    limit: usize,
) -> crate::Result<Vec<MessageResult>> {
    let terms = extract_search_terms(query);
    let phrases = extract_phrases(query);
    let precise = is_precise_query_shape(query);
    let cap = candidate_cap(limit) as i64;

    let sanitized = sanitize_fts5_query(query);
    let use_like = sanitized.is_empty() || requires_like_fallback(query);

    let mut hits: Vec<(MessageRow, f64)> = if use_like {
        like_message_candidates(store, &terms, &phrases, filter, cap)?
    } else {
        match store.search_messages_fts(&sanitized, filter, cap) {
            Ok(found) => found.into_iter().map(|h| (h.row, h.rank)).collect(),
            // A residual FTS syntax error still routes to the substring scan (§11.2).
            Err(_) => like_message_candidates(store, &terms, &phrases, filter, cap)?,
        }
    };

    if precise {
        for (row, rank) in hits.iter_mut() {
            let text = row.content.as_deref().unwrap_or("");
            *rank -= directness_score(text, &terms, &phrases) * MESSAGE_DIRECTNESS_WEIGHT;
        }
    }

    sort_message_hits(&mut hits, sort);
    hits.truncate(limit);
    Ok(hits
        .into_iter()
        .map(|(row, rank)| {
            let snippet = snippet(row.content.as_deref().unwrap_or(""), &terms);
            MessageResult { row, rank, snippet }
        })
        .collect())
}

/// Search the summary DAG (§11). FTS-only (summaries are dense, ASCII-ish prose), with the same
/// directness nudge for precise shapes.
pub fn search_nodes(
    store: &Store,
    query: &str,
    session_id: &str,
    limit: usize,
) -> crate::Result<Vec<NodeResult>> {
    let sanitized = sanitize_fts5_query(query);
    if sanitized.is_empty() {
        return Ok(Vec::new());
    }
    let terms = extract_search_terms(query);
    let phrases = extract_phrases(query);
    let precise = is_precise_query_shape(query);
    let cap = candidate_cap(limit) as i64;

    let mut hits: Vec<NodeHit> = store
        .search_nodes_fts(&sanitized, session_id, cap)
        .unwrap_or_default();
    if precise {
        for h in hits.iter_mut() {
            h.rank -= directness_score(&h.node.summary, &terms, &phrases) * NODE_DIRECTNESS_WEIGHT;
        }
    }
    hits.sort_by(|a, b| a.rank.total_cmp(&b.rank));
    hits.truncate(limit);
    Ok(hits
        .into_iter()
        .map(|h| NodeResult {
            node: h.node,
            rank: h.rank,
        })
        .collect())
}

/// Pull LIKE candidates and synthesize a rank from their directness (lower = better).
fn like_message_candidates(
    store: &Store,
    terms: &[String],
    phrases: &[String],
    filter: &MessageFilter<'_>,
    cap: i64,
) -> crate::Result<Vec<(MessageRow, f64)>> {
    let rows = store.search_messages_like(terms, filter, cap)?;
    Ok(rows
        .into_iter()
        .map(|row| {
            let text = row.content.as_deref().unwrap_or("");
            let rank = -directness_score(text, terms, phrases);
            (row, rank)
        })
        .collect())
}

/// Order candidates in place per the sort mode (§11.3).
fn sort_message_hits(hits: &mut [(MessageRow, f64)], sort: SortMode) {
    match sort {
        SortMode::Recency => hits.sort_by(|a, b| {
            b.0.timestamp
                .total_cmp(&a.0.timestamp)
                .then(a.1.total_cmp(&b.1))
        }),
        SortMode::Relevance => hits.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then(b.0.timestamp.total_cmp(&a.0.timestamp))
        }),
        SortMode::Hybrid => {
            let now = now_secs();
            hits.sort_by(|a, b| {
                hybrid_key(a.1, a.0.timestamp, now)
                    .total_cmp(&hybrid_key(b.1, b.0.timestamp, now))
                    .then(b.0.timestamp.total_cmp(&a.0.timestamp))
            });
        }
    }
}

/// `rank / (1 + age_hours * RATE)` (§11.3). Age is clamped non-negative.
fn hybrid_key(rank: f64, timestamp: f64, now: f64) -> f64 {
    let age_hours = ((now - timestamp) / 3600.0).max(0.0);
    rank / (1.0 + age_hours * AGE_DECAY_RATE)
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Turn arbitrary text into an FTS5-safe MATCH expression (§11.1): every bare term and every quoted
/// phrase becomes a `"quoted"` token (so FTS operators can never reach the parser). Empty result =>
/// the caller should take the LIKE path.
pub fn sanitize_fts5_query(query: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for phrase in extract_phrases(query) {
        let cleaned = phrase.replace('"', " ");
        let cleaned = cleaned.trim();
        if !cleaned.is_empty() {
            out.push(format!("\"{cleaned}\""));
        }
    }
    for term in bare_terms(query) {
        out.push(format!("\"{term}\""));
    }
    out.join(" ")
}

/// Whether a query must use the LIKE fallback instead of FTS5 (§11.2): it contains CJK/emoji (the
/// `unicode61` tokenizer drops them), has unbalanced quotes, or contains a risky ASCII shape like
/// `foo-bar`, `a:b`, `x/y` that FTS would mis-tokenize.
pub fn requires_like_fallback(query: &str) -> bool {
    if query.chars().filter(|&c| c == '"').count() % 2 != 0 {
        return true;
    }
    if query.chars().any(is_cjk_or_emoji) {
        return true;
    }
    let chars: Vec<char> = query.chars().collect();
    chars.windows(3).any(|w| {
        w[0].is_ascii_alphanumeric()
            && matches!(w[1], '-' | ':' | '/')
            && w[2].is_ascii_alphanumeric()
    })
}

/// Lowercased, de-duplicated alphanumeric tokens from the whole query (used by the LIKE scan).
pub fn extract_search_terms(query: &str) -> Vec<String> {
    dedup(tokenize(query))
}

/// Lowercased contents of each balanced `"..."` phrase (unbalanced trailing quote ignored).
pub fn extract_phrases(query: &str) -> Vec<String> {
    let mut phrases = Vec::new();
    let mut chars = query.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            let mut buf = String::new();
            let mut closed = false;
            for d in chars.by_ref() {
                if d == '"' {
                    closed = true;
                    break;
                }
                buf.push(d);
            }
            let trimmed = buf.trim().to_lowercase();
            if closed && !trimmed.is_empty() {
                phrases.push(trimmed);
            }
        }
    }
    phrases
}

/// Whether the query is "precise" enough to earn the directness nudge (§11.3): a single bare term,
/// or one phrase plus at most two bare terms.
pub fn is_precise_query_shape(query: &str) -> bool {
    let terms = bare_terms(query);
    let phrases = extract_phrases(query);
    (phrases.is_empty() && terms.len() == 1) || (phrases.len() == 1 && terms.len() <= 2)
}

/// A coarse "directness" score (§11.3): reward distinct term/phrase hits, lightly penalize
/// repetition. Higher = the text is more squarely about the query.
pub fn directness_score(text: &str, terms: &[String], phrases: &[String]) -> f64 {
    let lc = text.to_lowercase();
    let unique_hits = terms.iter().filter(|t| lc.contains(t.as_str())).count();
    let phrase_hits = phrases.iter().filter(|p| lc.contains(p.as_str())).count();
    let total_occurrences: usize = terms.iter().map(|t| count_occurrences(&lc, t)).sum();
    let repeats = total_occurrences.saturating_sub(unique_hits);
    let penalty = repeats.min(6);
    let base = unique_hits * 5 + phrase_hits * 8;
    (base as f64 - penalty as f64).max(0.0)
}

/// An ~80-char snippet around the first matching term (§11.3); empty terms => a leading window.
pub fn snippet(content: &str, terms: &[String]) -> String {
    if content.is_empty() {
        return String::new();
    }
    let lc = content.to_lowercase();
    let hit = terms
        .iter()
        .filter_map(|t| lc.find(t.as_str()))
        .min()
        .unwrap_or(0);
    let half = SNIPPET_WINDOW / 2;
    let start = floor_char_boundary(content, hit.saturating_sub(half));
    let end = ceil_char_boundary(content, (hit + half).min(content.len()));
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.push_str(content[start..end].trim());
    if end < content.len() {
        out.push('…');
    }
    out
}

/// Fetch-width / candidate cap (§11.3): pull a multiple of `limit` so post-sort truncation keeps
/// quality, bounded so a huge `limit` can't scan the world.
fn candidate_cap(limit: usize) -> usize {
    limit.saturating_mul(20).clamp(500, 5000).max(limit)
}

// ---- token helpers ------------------------------------------------------------------------------

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

fn bare_terms(query: &str) -> Vec<String> {
    dedup(tokenize(&strip_phrases(query)))
}

/// Replace every balanced `"..."` region (and a trailing unbalanced quote) with spaces.
fn strip_phrases(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    let mut in_quote = false;
    for c in query.chars() {
        if c == '"' {
            in_quote = !in_quote;
            out.push(' ');
        } else if in_quote {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

fn dedup(mut v: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    v.retain(|t| seen.insert(t.clone()));
    v
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(needle).count()
}

fn is_cjk_or_emoji(c: char) -> bool {
    let u = c as u32;
    matches!(u,
        0x3040..=0x30FF        // Hiragana + Katakana
        | 0x3400..=0x4DBF      // CJK Unified Ext A
        | 0x4E00..=0x9FFF      // CJK Unified
        | 0xAC00..=0xD7A3      // Hangul syllables
        | 0xF900..=0xFAFF      // CJK compatibility ideographs
        | 0x20000..=0x2FA1F    // CJK Unified Ext B+ / compatibility supplement
        | 0x2600..=0x27BF      // misc symbols + dingbats
        | 0x1F300..=0x1FAFF    // emoji blocks
    )
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
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
    use crate::store::{NewMessage, Store};

    #[test]
    fn sanitize_preserves_phrases_and_strips_operators() {
        assert_eq!(sanitize_fts5_query("hello world"), "\"hello\" \"world\"");
        assert_eq!(
            sanitize_fts5_query("\"exact phrase\" extra"),
            "\"exact phrase\" \"extra\""
        );
        // FTS operators are neutralized by quoting.
        assert_eq!(
            sanitize_fts5_query("foo* OR bar^"),
            "\"foo\" \"or\" \"bar\""
        );
        assert!(sanitize_fts5_query("***").is_empty());
    }

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
    }

    #[test]
    fn terms_and_phrases_extracted() {
        assert_eq!(extract_search_terms("Foo BAR foo"), vec!["foo", "bar"]);
        assert_eq!(
            extract_phrases("a \"Quoted Phrase\" b"),
            vec!["quoted phrase".to_string()]
        );
        assert!(is_precise_query_shape("singleterm"));
        assert!(is_precise_query_shape("\"a phrase\" plus"));
        assert!(!is_precise_query_shape("three different bare terms"));
    }

    #[test]
    fn directness_rewards_hits_and_penalizes_repetition() {
        let terms = vec!["alpha".to_string(), "beta".to_string()];
        let phrases: Vec<String> = vec![];
        let both = directness_score("alpha beta", &terms, &phrases);
        let one = directness_score("alpha only", &terms, &phrases);
        assert!(both > one);
        let repeated = directness_score("alpha alpha alpha beta", &terms, &phrases);
        assert!(repeated < both, "repetition is penalized");
    }

    #[test]
    fn snippet_windows_around_first_hit() {
        let content = format!("{}needle{}", "x".repeat(200), "y".repeat(200));
        let snip = snippet(&content, &["needle".to_string()]);
        assert!(snip.contains("needle"));
        assert!(snip.starts_with('…') && snip.ends_with('…'));
        assert!(snip.chars().count() <= SNIPPET_WINDOW + 8);
    }

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
    }

    #[test]
    fn like_fallback_finds_cjk() {
        let store = Store::open_in_memory().unwrap();
        store
            .append_batch(
                "s1",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("検索 テスト".into()),
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
        assert!(hits[0].row.content.as_deref().unwrap().contains("検索"));
    }
}
