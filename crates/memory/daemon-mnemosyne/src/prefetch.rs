// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Prefetch hardening — the provider's per-turn context-injection pipeline
//! (`hermes_memory_provider/__init__.py` L87-L379, L1474-L1663).
//!
//! Manual recall tools can stay broad; prefetch is *silently* injected into every model call, so
//! it over-fetches then filters hard: low-quality fragment drop, source-quality gating (distilled
//! memories beat raw transcript, assistant transcript is excluded), topic-signal thresholds,
//! adjusted-score ranking, and token-set semantic dedup. Named [`PrefetchProfile`]s bundle the
//! knobs; per-contact identity rows bypass recall entirely and are injected deterministically at
//! the front (they must surface on every turn even when the query doesn't match them).

use crate::engine::MemoryRow;
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// Over-fetch factor: recall more, filter junk, cap back to `top_k` (`_PREFETCH_OVERFETCH`).
pub const PREFETCH_OVERFETCH: usize = 16;

/// Lone tokens shorter than this are dropped (`_PREFETCH_MIN_FRAGMENT_CHARS`).
const MIN_FRAGMENT_CHARS: usize = 8;

/// Single-token stopwords that never survive the low-quality filter
/// (`_PREFETCH_FRAGMENT_STOPWORDS`).
const FRAGMENT_STOPWORDS: &[&str] = &[
    "still", "what", "most", "almost", "back", "now", "too", "right", "being", "going", "here",
    "there", "then", "just", "also", "only", "even", "very", "really", "again", "away", "off",
    "out", "up", "down", "over", "that", "this", "it", "so",
];

/// Extra stopwords for the dedup token sets (`_PREFETCH_DEDUP_STOPWORDS` = fragment ∪ these).
const DEDUP_EXTRA_STOPWORDS: &[&str] = &[
    "about", "after", "before", "because", "could", "from", "have", "into", "like", "more", "need",
    "needs", "than", "them", "they", "want", "wants", "when", "where", "which", "while", "would",
    "yourself",
];

/// Sources treated as distilled/high-quality context (`_PREFETCH_DISTILLED_SOURCES`).
const DISTILLED_SOURCES: &[&str] = &[
    "preference",
    "correction",
    "fact",
    "identity",
    "insight",
    "sleep_consolidation",
];

/// A named bundle of prefetch knobs (`PrefetchProfile` dataclass L279-L296). Operators select one
/// via `[mnemosyne].prefetch_profile`.
#[derive(Clone, Debug)]
pub struct PrefetchProfile {
    /// Profile name.
    pub name: &'static str,
    /// Final injected count after over-fetch + filter (`_PREFETCH_TOP_K`).
    pub top_k: usize,
    /// Optional per-call importance-weight override (`None` -> recall default).
    pub importance_weight: Option<f64>,
    /// Optional per-call vec-weight override.
    pub vec_weight: Option<f64>,
    /// Optional per-call FTS-weight override.
    pub fts_weight: Option<f64>,
    /// Soft temporal boost weight fed to recall.
    pub temporal_weight: f64,
    /// Temporal boost half-life (hours).
    pub temporal_halflife: f64,
    /// Rows below this score AND below `min_importance` are dropped.
    pub min_score: f64,
    /// See `min_score` (either passes the gate).
    pub min_importance: f64,
    /// Minimum topical signal for distilled rows.
    pub min_topic_signal: f64,
    /// Minimum topical signal for raw transcript rows (stricter).
    pub raw_min_topic_signal: f64,
    /// Per-memory content char cap; `0` -> the configured/env default.
    pub content_char_limit: usize,
    /// Drop bare single-token fragments.
    pub drop_low_quality: bool,
    /// Token-set semantic dedup of the final rows.
    pub semantic_dedup: bool,
    /// Exclude `[ASSISTANT]` transcript rows entirely.
    pub exclude_assistant: bool,
}

impl Default for PrefetchProfile {
    fn default() -> Self {
        Self {
            name: "general",
            top_k: 5,
            importance_weight: None,
            vec_weight: None,
            fts_weight: None,
            temporal_weight: 0.2,
            temporal_halflife: 48.0,
            min_score: 0.20,
            min_importance: 0.65,
            min_topic_signal: 0.08,
            raw_min_topic_signal: 0.18,
            content_char_limit: 0,
            drop_low_quality: true,
            semantic_dedup: true,
            exclude_assistant: true,
        }
    }
}

/// Resolve a named profile, falling back to `general` for unknown/empty (`_resolve_profile`).
pub fn resolve_profile(name: &str) -> PrefetchProfile {
    match name {
        // Favor recent, high-importance memories; same filter/dedup defaults.
        "social-chat" => PrefetchProfile {
            name: "social-chat",
            top_k: 6,
            importance_weight: Some(0.6),
            temporal_weight: 0.35,
            temporal_halflife: 24.0,
            ..Default::default()
        },
        _ => PrefetchProfile::default(),
    }
}

/// True if recalled content is a bare single-token fragment with no value as injected context
/// (`_is_low_quality_prefetch`). Multi-word phrases always pass.
pub fn is_low_quality(content: &str) -> bool {
    let c = content.trim();
    if c.is_empty() {
        return true;
    }
    c.split_whitespace().count() <= 1
        && (c.chars().count() <= MIN_FRAGMENT_CHARS
            || FRAGMENT_STOPWORDS.contains(&c.to_lowercase().as_str()))
}

fn strip_raw_prefix(content: &str) -> &str {
    let c = content.trim();
    let upper = c.to_uppercase();
    for prefix in ["[USER]", "[ASSISTANT]", "[IDENTITY]"] {
        if upper.starts_with(prefix) {
            return c[prefix.len()..].trim_start();
        }
    }
    c
}

fn token_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)[a-z0-9][a-z0-9_./:-]*").expect("valid prefetch token re"))
}

/// The dedup token set for one memory (`_prefetch_tokens`): prefix-stripped, lowercased, short
/// tokens and stopwords dropped.
fn tokens(content: &str) -> HashSet<String> {
    let c = strip_raw_prefix(content).to_lowercase();
    token_re()
        .find_iter(&c)
        .map(|m| m.as_str().to_string())
        .filter(|t| {
            t.chars().count() > 2
                && !FRAGMENT_STOPWORDS.contains(&t.as_str())
                && !DEDUP_EXTRA_STOPWORDS.contains(&t.as_str())
        })
        .collect()
}

/// Best available non-importance relevance signal for a recall row (`_prefetch_topic_signal`).
pub fn topic_signal(row: &MemoryRow) -> f64 {
    let mut signal = row.keyword_score.max(row.fts_score).max(row.dense_score);
    // Fact/entity matches are explicit relevance signals even when recall did not fill
    // keyword/FTS scores for that path.
    if row.fact_match || row.entity_match {
        signal = signal.max(0.20);
    }
    signal
}

/// Relative usefulness multiplier for injected memory (`_prefetch_source_quality`): distilled
/// memories are better prompt context than raw transcript; assistant transcript is excluded.
pub fn source_quality(row: &MemoryRow) -> f64 {
    let content = row.content.trim();
    let upper = content.to_uppercase();
    let source = row.source.to_lowercase();

    if upper.starts_with("[ASSISTANT]") {
        return 0.0;
    }
    let mut quality = 1.0;
    if DISTILLED_SOURCES.contains(&source.as_str()) {
        quality *= 1.12;
    }
    if source == "conversation" {
        quality *= 0.72;
    }
    if upper.starts_with("[USER]") {
        quality *= 0.68;
    } else if upper.starts_with("[IDENTITY]") {
        quality *= 0.80;
    } else if source.starts_with("memoria_source") {
        quality *= 0.90;
    }
    quality
}

/// Whether a row is raw transcript (stricter topic-signal threshold, `_prefetch_is_raw`).
pub fn is_raw(row: &MemoryRow) -> bool {
    let content = row.content.trim().to_uppercase();
    row.source.to_lowercase() == "conversation"
        || content.starts_with("[USER]")
        || content.starts_with("[IDENTITY]")
}

/// The prefetch ranking score (`_prefetch_adjusted_score`):
/// `(score*0.65 + signal*0.35 + importance*0.05) * quality`.
pub fn adjusted_score(row: &MemoryRow) -> f64 {
    let importance = row.importance.clamp(0.0, 1.0);
    (row.score * 0.65 + topic_signal(row) * 0.35 + importance * 0.05) * source_quality(row)
}

/// Collapse near-duplicate memory rows, keeping the best-ranked variant (`_semantic_dedup_prefetch`;
/// rows must arrive best-first). Token-less rows are dropped.
pub fn semantic_dedup(rows: Vec<MemoryRow>, threshold: f64) -> Vec<MemoryRow> {
    let mut kept: Vec<MemoryRow> = Vec::new();
    let mut kept_tokens: Vec<HashSet<String>> = Vec::new();
    for row in rows {
        let toks = tokens(&row.content);
        if toks.is_empty() {
            continue;
        }
        let duplicate = kept_tokens.iter().any(|existing| {
            let overlap = toks.intersection(existing).count();
            if overlap == 0 {
                return false;
            }
            let union = toks.union(existing).count().max(1);
            let smaller = toks.len().min(existing.len()).max(1);
            (overlap as f64 / union as f64) >= threshold
                || (overlap as f64 / smaller as f64) >= 0.86
        });
        if duplicate {
            continue;
        }
        kept.push(row);
        kept_tokens.push(toks);
    }
    kept
}

/// Format recalled memory content for prompt injection (`_format_prefetch_content`): word-boundary
/// truncation when a positive char limit is configured, untouched otherwise.
pub fn format_content(content: &str, limit: usize) -> String {
    let chars: Vec<char> = content.chars().collect();
    if limit == 0 || chars.len() <= limit {
        return content.to_string();
    }
    let mut cut: String = chars[..limit].iter().collect::<String>().trim_end().into();
    if let Some(boundary) = cut.rfind(' ') {
        // Prefer a word boundary when one exists reasonably close to the limit.
        if cut[..boundary].chars().count() >= (limit / 2).max(1) {
            cut = cut[..boundary].trim_end().to_string();
        }
    }
    format!("{cut}...")
}

/// Apply the profile's junk/quality/threshold filters, rank by [`adjusted_score`], semantic-dedup,
/// and cap back to `top_k` (`_prefetch_bank` filter block L1618-L1642).
pub fn filter_and_rank(rows: Vec<MemoryRow>, profile: &PrefetchProfile) -> Vec<MemoryRow> {
    let mut filtered: Vec<MemoryRow> = rows
        .into_iter()
        .filter(|r| {
            if profile.drop_low_quality && is_low_quality(&r.content) {
                return false;
            }
            if profile.exclude_assistant && source_quality(r) <= 0.0 {
                return false;
            }
            let required = if is_raw(r) {
                profile.raw_min_topic_signal
            } else {
                profile.min_topic_signal
            };
            if topic_signal(r) < required {
                return false;
            }
            if r.score < profile.min_score && r.importance < profile.min_importance {
                return false;
            }
            true
        })
        .collect();
    filtered.sort_by(|a, b| adjusted_score(b).total_cmp(&adjusted_score(a)));
    if profile.semantic_dedup {
        filtered = semantic_dedup(filtered, 0.72);
    }
    filtered.truncate(profile.top_k);
    filtered
}

/// Render the memory-bank block (`_prefetch_bank` L1645-L1660):
/// `  [ts] (importance X.XX[, source S])[ [TRUST]] content`, whitespace-collapsed.
pub fn render_bank_block(rows: &[MemoryRow], content_limit: usize) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let mut lines = vec!["## Mnemosyne Context".to_string()];
    for r in rows {
        let content = format_content(&r.content, content_limit);
        let content = content.split_whitespace().collect::<Vec<_>>().join(" ");
        let ts: String = r.timestamp.chars().take(16).collect();
        let source = r.source.trim();
        let source_tag = if !source.is_empty() && source != "conversation" {
            format!(", source {source}")
        } else {
            String::new()
        };
        let trust_tag = if r.trust_tier != "STATED" && !r.trust_tier.is_empty() {
            format!(" [{}]", r.trust_tier)
        } else {
            String::new()
        };
        lines.push(format!(
            "  [{}] (importance {:.2}{}){} {}",
            ts, r.importance, source_tag, trust_tag, content
        ));
    }
    lines.join("\n")
}

/// One always-inject identity row (`_identity_fichas` result shape).
#[derive(Clone, Debug)]
pub struct IdentityRow {
    /// Row content.
    pub content: String,
    /// Row importance (0.95 default when NULL).
    pub importance: f64,
    /// Row timestamp (may be empty).
    pub timestamp: String,
}

/// Render the always-inject identity block (`_prefetch_identity`): every identity row for the
/// active session, deduplicated against whatever recall already surfaced (`existing`), tagged
/// `[IDENTITY]`. Empty when there is nothing new to inject.
pub fn render_identity_block(rows: &[IdentityRow], existing: &str, content_limit: usize) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let mut lines = vec!["## Mnemosyne Context".to_string()];
    let mut seen: HashSet<&str> = HashSet::new();
    for r in rows {
        if r.content.is_empty() || !seen.insert(r.content.as_str()) {
            continue;
        }
        let disp = format_content(&r.content, content_limit);
        // Dedup against anything recall already surfaced (raw or truncated).
        if existing.contains(&r.content) || existing.contains(&disp) {
            continue;
        }
        let ts: String = r.timestamp.chars().take(16).collect();
        lines.push(format!(
            "  [{}] (importance {:.2}) [IDENTITY] {}",
            ts, r.importance, disp
        ));
    }
    if lines.len() <= 1 {
        return String::new();
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(content: &str, source: &str, score: f64, importance: f64, kw: f64) -> MemoryRow {
        MemoryRow {
            content: content.to_string(),
            source: source.to_string(),
            score,
            importance,
            keyword_score: kw,
            ..Default::default()
        }
    }

    #[test]
    fn low_quality_drops_lone_fragments_only() {
        assert!(is_low_quality(""));
        assert!(is_low_quality("still"));
        assert!(is_low_quality("almost"));
        assert!(is_low_quality("short"), "lone token <= 8 chars");
        assert!(!is_low_quality("kubernetes"), "long lone token passes");
        assert!(!is_low_quality("uses dark mode"), "phrases always pass");
    }

    #[test]
    fn source_quality_excludes_assistant_and_ranks_distilled_over_raw() {
        let distilled = row("prefers tabs over spaces", "preference", 0.5, 0.8, 0.4);
        let raw_user = row(
            "[USER] I prefer tabs over spaces",
            "conversation",
            0.5,
            0.5,
            0.4,
        );
        let assistant = row(
            "[ASSISTANT] noted, tabs it is",
            "conversation",
            0.9,
            0.5,
            0.9,
        );
        assert_eq!(source_quality(&assistant), 0.0);
        assert!(source_quality(&distilled) > 1.0);
        assert!((source_quality(&raw_user) - 0.72 * 0.68).abs() < 1e-9);
        assert!(adjusted_score(&distilled) > adjusted_score(&raw_user));
    }

    #[test]
    fn topic_signal_floors_fact_and_entity_matches() {
        let mut r = row("Maya works at Acme", "conversation", 0.4, 0.5, 0.05);
        assert!(topic_signal(&r) < 0.08);
        r.entity_match = true;
        assert!((topic_signal(&r) - 0.20).abs() < 1e-9);
    }

    #[test]
    fn filter_and_rank_applies_thresholds_and_caps() {
        let profile = PrefetchProfile::default();
        let rows = vec![
            row(
                "prefers dark mode in all editors",
                "preference",
                0.6,
                0.9,
                0.5,
            ),
            row("weak", "conversation", 0.01, 0.1, 0.01), // low-quality fragment
            row(
                "[ASSISTANT] sure, dark mode enabled",
                "conversation",
                0.9,
                0.9,
                0.9,
            ), // excluded
            row(
                "[USER] the deploy failed twice yesterday",
                "conversation",
                0.5,
                0.6,
                0.1,
            ), // raw row below raw_min_topic_signal (0.18)
            row("project Atlas ships in March", "fact", 0.05, 0.2, 0.3), // below score+importance gate
        ];
        let kept = filter_and_rank(rows, &profile);
        assert_eq!(
            kept.len(),
            1,
            "{:?}",
            kept.iter().map(|r| &r.content).collect::<Vec<_>>()
        );
        assert!(kept[0].content.contains("dark mode"));
    }

    #[test]
    fn semantic_dedup_collapses_near_duplicates() {
        let rows = vec![
            row(
                "prefers dark mode in all editors",
                "preference",
                0.6,
                0.9,
                0.5,
            ),
            row(
                "[USER] prefers dark mode in all editors",
                "conversation",
                0.4,
                0.5,
                0.4,
            ),
            row(
                "the database migration finished cleanly",
                "fact",
                0.3,
                0.7,
                0.3,
            ),
        ];
        let kept = semantic_dedup(rows, 0.72);
        assert_eq!(kept.len(), 2);
        assert!(kept[0].content.starts_with("prefers"));
        assert!(kept[1].content.contains("migration"));
    }

    #[test]
    fn format_content_truncates_on_word_boundary() {
        assert_eq!(format_content("short text", 0), "short text");
        assert_eq!(format_content("short text", 100), "short text");
        let long = "alpha beta gamma delta epsilon";
        let cut = format_content(long, 14);
        assert_eq!(cut, "alpha beta...");
    }

    #[test]
    fn identity_block_dedups_against_existing_output() {
        let rows = vec![
            IdentityRow {
                content: "[IDENTITY] speaking with Ana, prefers Spanish".into(),
                importance: 0.95,
                timestamp: "2026-01-01T10:00:00".into(),
            },
            IdentityRow {
                content: "already surfaced identity".into(),
                importance: 0.9,
                timestamp: String::new(),
            },
        ];
        let existing = "## Mnemosyne Context\n  [..] already surfaced identity";
        let block = render_identity_block(&rows, existing, 0);
        assert!(block.contains("[IDENTITY]"));
        assert!(block.contains("Ana"));
        assert!(!block.contains("already surfaced"), "{block}");

        let all_dup = render_identity_block(&rows[1..], existing, 0);
        assert_eq!(all_dup, "");
    }

    #[test]
    fn profile_resolution_falls_back_to_general() {
        assert_eq!(resolve_profile("social-chat").name, "social-chat");
        assert_eq!(resolve_profile("social-chat").top_k, 6);
        assert_eq!(resolve_profile("nope").name, "general");
        assert_eq!(resolve_profile("").name, "general");
    }
}
