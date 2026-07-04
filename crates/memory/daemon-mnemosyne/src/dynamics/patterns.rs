// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Memory compression + pattern detection — port of `patterns.py`.
//!
//! [`MemoryCompressor`] compresses memory content with ordered strategies (dictionary phrase
//! replacement, run-length encoding, semantic truncation, or `auto`), and [`PatternDetector`]
//! finds recurring temporal / content / sequence patterns over a batch of memories.
//!
//! Like the Python module, this is a self-contained utility surface: nothing else in Mnemosyne
//! imports `patterns.py` (it ships as public API only), and the port keeps that shape — pure
//! functions over caller-supplied rows, no engine wiring.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Statistics from compression operations (`patterns.py` `CompressionStats` L27-L41). Sizes are
/// UTF-8 byte lengths, mirroring Python's `len(content.encode("utf-8"))`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct CompressionStats {
    /// Input size in bytes.
    pub original_size: usize,
    /// Output size in bytes.
    pub compressed_size: usize,
    /// `compressed / original` (1.0 when the input is empty).
    pub ratio: f64,
    /// The strategy that produced the output (`dict`/`rle`/`semantic`/`none`/`auto`).
    pub method: String,
    /// Reserved by the Python dataclass (never set by any strategy there either).
    pub patterns_found: usize,
    /// Batch size for [`MemoryCompressor::compress_batch`] results.
    pub memories_compressed: usize,
}

impl CompressionStats {
    /// Percent saved vs the original (`patterns.py` L37-L41).
    pub fn savings_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (1.0 - self.compressed_size as f64 / self.original_size as f64) * 100.0
    }

    fn sized(original: usize, compressed: usize, method: &str) -> Self {
        Self {
            original_size: original,
            compressed_size: compressed,
            ratio: if original > 0 {
                compressed as f64 / original as f64
            } else {
                1.0
            },
            method: method.to_string(),
            ..Self::default()
        }
    }
}

/// Compress memory content using multiple strategies (`patterns.py` `MemoryCompressor` L44-L221).
///
/// The dictionary is an **ordered** phrase -> token list (Python dict insertion order matters:
/// replacements are applied sequentially).
pub struct MemoryCompressor {
    dictionary: Vec<(String, String)>,
}

impl Default for MemoryCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryCompressor {
    /// A compressor with the default phrase dictionary (`_build_default_dict` L58-L75).
    pub fn new() -> Self {
        let dict = [
            ("remember that ", ""),
            ("the user said ", ""),
            ("the user asked ", ""),
            ("the user wants ", ""),
            ("conversation about ", ""),
            ("please note that ", ""),
            ("important: ", ""),
            ("user preference: ", ""),
            ("project context: ", "\t"),
            ("api key ", "\u{0A}"),
            ("token ", "\u{0B}"),
            ("session ", "\u{0C}"),
            ("mnemosyne ", "\u{0D}"),
        ];
        Self {
            dictionary: dict
                .iter()
                .map(|(p, t)| (p.to_string(), t.to_string()))
                .collect(),
        }
    }

    /// A compressor with a caller-supplied ordered dictionary.
    pub fn with_dictionary(dictionary: Vec<(String, String)>) -> Self {
        Self { dictionary }
    }

    /// Compress one content string with `method` — `dict`, `rle`, `semantic`, or `auto` (dict
    /// first, RLE fallback when savings < 5%); anything else passes through as `none`
    /// (`patterns.py` `compress` L77-L109).
    pub fn compress(&self, content: &str, method: &str) -> (String, CompressionStats) {
        match method {
            "auto" => {
                let (compressed, stats) = self.dict_compress(content);
                if stats.savings_percent() < 5.0 {
                    return self.rle_compress(content);
                }
                (compressed, stats)
            }
            "dict" => self.dict_compress(content),
            "rle" => self.rle_compress(content),
            "semantic" => self.semantic_compress_single(content),
            _ => {
                let size = content.len();
                (
                    content.to_string(),
                    CompressionStats::sized(size, size, "none"),
                )
            }
        }
    }

    /// Sequential ordered phrase replacement (`_dict_compress` L111-L123).
    fn dict_compress(&self, content: &str) -> (String, CompressionStats) {
        let original_size = content.len();
        let mut compressed = content.to_string();
        for (phrase, token) in &self.dictionary {
            compressed = compressed.replace(phrase.as_str(), token);
        }
        let stats = CompressionStats::sized(original_size, compressed.len(), "dict");
        (compressed, stats)
    }

    /// Run-length encoding for repeated characters: runs longer than 3 (capped at 255) become
    /// `[c*count]` (`_rle_compress` L125-L155).
    fn rle_compress(&self, content: &str) -> (String, CompressionStats) {
        let original_size = content.len();
        if content.is_empty() {
            return (content.to_string(), CompressionStats::sized(0, 0, "rle"));
        }
        let chars: Vec<char> = content.chars().collect();
        let mut out = String::new();
        let mut count = 1usize;
        for i in 1..chars.len() {
            if chars[i] == chars[i - 1] && count < 255 {
                count += 1;
            } else {
                if count > 3 {
                    out.push_str(&format!("[{}*{}]", chars[i - 1], count));
                } else {
                    out.extend(&chars[i - count..i]);
                }
                count = 1;
            }
        }
        if count > 3 {
            out.push_str(&format!("[{}*{}]", chars[chars.len() - 1], count));
        } else {
            out.extend(&chars[chars.len() - count..]);
        }
        let stats = CompressionStats::sized(original_size, out.len(), "rle");
        (out, stats)
    }

    /// Semantic compression placeholder: contents over 500 bytes keep the first 250 and last 100
    /// characters around an elision marker (`_semantic_compress_single` L157-L171).
    fn semantic_compress_single(&self, content: &str) -> (String, CompressionStats) {
        let original_size = content.len();
        let compressed = if original_size > 500 {
            let chars: Vec<char> = content.chars().collect();
            let head: String = chars.iter().take(250).collect();
            let tail: String = chars[chars.len().saturating_sub(100)..].iter().collect();
            format!("{head} [...] {tail}")
        } else {
            content.to_string()
        };
        let stats = CompressionStats::sized(original_size, compressed.len(), "semantic");
        (compressed, stats)
    }

    /// Compress a batch of contents, returning the compressed contents plus aggregate stats
    /// (`compress_batch` L173-L204; the Python `_compressed`/`_compression_method` dict markers
    /// have no Rust counterpart — callers own their row shape).
    pub fn compress_batch(
        &self,
        contents: &[String],
        method: &str,
    ) -> (Vec<String>, CompressionStats) {
        let mut total_original = 0usize;
        let mut total_compressed = 0usize;
        let mut out = Vec::with_capacity(contents.len());
        for content in contents {
            let (c, s) = self.compress(content, method);
            total_original += s.original_size;
            total_compressed += s.compressed_size;
            out.push(c);
        }
        let stats = CompressionStats {
            original_size: total_original,
            compressed_size: total_compressed,
            ratio: if total_original > 0 {
                total_compressed as f64 / total_original as f64
            } else {
                1.0
            },
            method: method.to_string(),
            patterns_found: 0,
            memories_compressed: contents.len(),
        };
        (out, stats)
    }

    /// Decompress content compressed with `method` (`decompress` L206-L221).
    ///
    /// For `dict`, Python builds the token -> phrase reverse map with last-duplicate-wins — which
    /// includes the empty-string token, and `str.replace("", phrase)` inserts the phrase between
    /// every character (a latent Python bug for any dict-compressed content). The port restores
    /// only non-empty tokens; phrases compressed to `""` are unrecoverable by construction.
    pub fn decompress(&self, content: &str, method: &str) -> String {
        match method {
            "dict" => {
                // Last-wins reverse map over insertion order (`{v: k for k, v in ...}`).
                let mut reverse: Vec<(String, String)> = Vec::new();
                for (phrase, token) in &self.dictionary {
                    if token.is_empty() {
                        continue;
                    }
                    if let Some(entry) = reverse.iter_mut().find(|(t, _)| t == token) {
                        entry.1 = phrase.clone();
                    } else {
                        reverse.push((token.clone(), phrase.clone()));
                    }
                }
                let mut out = content.to_string();
                for (token, phrase) in &reverse {
                    out = out.replace(token.as_str(), phrase);
                }
                out
            }
            "rle" => {
                static RE: OnceLock<regex::Regex> = OnceLock::new();
                let re = RE.get_or_init(|| regex::Regex::new(r"\[(.)\*(\d+)\]").unwrap());
                re.replace_all(content, |caps: &regex::Captures<'_>| {
                    let ch = &caps[1];
                    let count: usize = caps[2].parse().unwrap_or(0);
                    ch.repeat(count)
                })
                .into_owned()
            }
            _ => content.to_string(),
        }
    }
}

/// A detected pattern in memory data (`patterns.py` `DetectedPattern` L224-L240).
#[derive(Clone, Debug, serde::Serialize)]
pub struct DetectedPattern {
    /// `temporal`, `content`, or `sequence`.
    pub pattern_type: String,
    /// Human-readable description.
    pub description: String,
    /// Confidence `[0, 1]`.
    pub confidence: f64,
    /// Up to 3 example rows/timestamps.
    pub samples: Vec<String>,
    /// Pattern-specific fields (hour/day/word counts, ...).
    pub metadata: serde_json::Value,
}

/// One memory row as pattern-detection input (the Python detectors take loose dicts; the port
/// names the fields they read: `content`, `timestamp`/`created_at`, `source`).
#[derive(Clone, Debug, Default)]
pub struct PatternMemory {
    /// The memory content.
    pub content: String,
    /// Primary ISO timestamp (`timestamp` key).
    pub timestamp: Option<String>,
    /// Fallback ISO timestamp (`created_at` key), used when `timestamp` is missing/empty.
    pub created_at: Option<String>,
    /// Ingestion source.
    pub source: Option<String>,
}

impl PatternMemory {
    /// `mem.get("timestamp") or mem.get("created_at")` — empty strings are falsy in Python.
    fn effective_timestamp(&self) -> Option<&str> {
        self.timestamp
            .as_deref()
            .filter(|s| !s.is_empty())
            .or(self.created_at.as_deref().filter(|s| !s.is_empty()))
    }
}

/// A parsed timestamp: the offset-local clock fields Python reads off `fromisoformat` results
/// (no timezone normalization), plus the round-tripped ISO string for samples.
struct ParsedTs {
    hour: u32,
    weekday: usize,
    iso: String,
}

/// Parse an ISO-8601 timestamp the way `datetime.fromisoformat(ts.replace("Z", "+00:00"))` does,
/// keeping the offset-local hour/weekday (`detect_temporal` L263-L267).
fn parse_iso(ts: &str) -> Option<ParsedTs> {
    use chrono::{Datelike, Timelike};
    let ts = ts.replace('Z', "+00:00");
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&ts) {
        return Some(ParsedTs {
            hour: dt.hour(),
            weekday: dt.weekday().num_days_from_monday() as usize,
            iso: dt.to_rfc3339(),
        });
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&ts, fmt) {
            return Some(ParsedTs {
                hour: dt.hour(),
                weekday: dt.weekday().num_days_from_monday() as usize,
                iso: dt.format("%Y-%m-%dT%H:%M:%S%.f").to_string(),
            });
        }
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(&ts, "%Y-%m-%d") {
        return Some(ParsedTs {
            hour: 0,
            weekday: d.weekday().num_days_from_monday() as usize,
            iso: format!("{}T00:00:00", d.format("%Y-%m-%d")),
        });
    }
    None
}

/// `Counter(items).most_common(n)`: count desc, first-seen order on ties (Python 3.7+ Counter
/// insertion-order tie-break).
fn most_common<T: Clone + Eq + std::hash::Hash>(items: &[T], n: usize) -> Vec<(T, usize)> {
    let mut index: HashMap<T, usize> = HashMap::new();
    let mut counts: Vec<(T, usize)> = Vec::new();
    for item in items {
        match index.get(item) {
            Some(&i) => counts[i].1 += 1,
            None => {
                index.insert(item.clone(), counts.len());
                counts.push((item.clone(), 1));
            }
        }
    }
    counts.sort_by_key(|c| std::cmp::Reverse(c.1)); // stable: insertion order preserved on ties
    counts.truncate(n);
    counts
}

/// The content-keyword regex `\b[a-zA-Z]{5,}\b` (`detect_content` L311).
fn word_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\b[a-zA-Z]{5,}\b").unwrap())
}

/// The `detect_content` stop-word set (`patterns.py` L312-L314).
const CONTENT_STOPWORDS: &[&str] = &[
    "about",
    "after",
    "before",
    "being",
    "could",
    "doing",
    "every",
    "having",
    "might",
    "other",
    "should",
    "their",
    "there",
    "these",
    "those",
    "through",
    "under",
    "where",
    "which",
    "while",
    "would",
    "mnemosyne",
    "memory",
    "memories",
];

/// Detect recurring patterns in memory data (`patterns.py` `PatternDetector` L243-L412).
pub struct PatternDetector {
    /// Patterns below this confidence are dropped (default 0.6).
    pub min_confidence: f64,
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
        }
    }
}

impl PatternDetector {
    /// A detector with a custom confidence floor.
    pub fn new(min_confidence: f64) -> Self {
        Self { min_confidence }
    }

    /// Temporal patterns: dominant hour-of-day (top 3) and day-of-week (top 2) concentrations
    /// (`detect_temporal` L256-L303). Needs at least 3 parseable timestamps.
    pub fn detect_temporal(&self, memories: &[PatternMemory]) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();
        let timestamps: Vec<ParsedTs> = memories
            .iter()
            .filter_map(|m| m.effective_timestamp())
            .filter_map(parse_iso)
            .collect();
        if timestamps.len() < 3 {
            return patterns;
        }
        let total = timestamps.len();

        let hours: Vec<u32> = timestamps.iter().map(|t| t.hour).collect();
        for (hour, count) in most_common(&hours, 3) {
            let confidence = count as f64 / total as f64;
            if confidence >= self.min_confidence {
                patterns.push(DetectedPattern {
                    pattern_type: "temporal".to_string(),
                    description: format!(
                        "Memories frequently created at {hour:02}:00 ({count}/{total} times)"
                    ),
                    confidence,
                    samples: timestamps
                        .iter()
                        .filter(|t| t.hour == hour)
                        .take(3)
                        .map(|t| t.iso.clone())
                        .collect(),
                    metadata: serde_json::json!({"hour": hour, "count": count, "total": total}),
                });
            }
        }

        const DAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        let weekdays: Vec<usize> = timestamps.iter().map(|t| t.weekday).collect();
        for (day, count) in most_common(&weekdays, 2) {
            let confidence = count as f64 / total as f64;
            if confidence >= self.min_confidence {
                patterns.push(DetectedPattern {
                    pattern_type: "temporal".to_string(),
                    description: format!(
                        "Memories frequently created on {} ({count}/{total} times)",
                        DAY_NAMES[day]
                    ),
                    confidence,
                    samples: timestamps
                        .iter()
                        .filter(|t| t.weekday == day)
                        .take(3)
                        .map(|t| t.iso.clone())
                        .collect(),
                    metadata: serde_json::json!({
                        "day": DAY_NAMES[day], "count": count, "total": total
                    }),
                });
            }
        }
        patterns
    }

    /// Content patterns: frequent keywords (top 5) and co-occurring keyword pairs (top 3)
    /// (`detect_content` L305-L354).
    pub fn detect_content(&self, memories: &[PatternMemory]) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();
        let all_text = memories
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        let words: Vec<String> = word_re()
            .find_iter(&all_text)
            .map(|m| m.as_str().to_string())
            .filter(|w| !CONTENT_STOPWORDS.contains(&w.as_str()))
            .collect();
        let total_words = words.len();

        for (word, count) in most_common(&words, 5) {
            let confidence = (count as f64 / (total_words as f64 * 0.05).max(3.0)).min(1.0);
            if count >= 2 && confidence >= self.min_confidence {
                let samples: Vec<String> = memories
                    .iter()
                    .filter(|m| m.content.to_lowercase().contains(&word))
                    .take(3)
                    .map(|m| m.content.clone())
                    .collect();
                patterns.push(DetectedPattern {
                    pattern_type: "content".to_string(),
                    description: format!("Frequent topic: '{word}' appears {count} times"),
                    confidence,
                    samples,
                    metadata: serde_json::json!({"word": word, "count": count}),
                });
            }
        }

        // Co-occurrence: keyword pairs appearing together in >= 2 memories (`detect_content`
        // L331-L352). Per-memory word sets iterate sorted (BTreeSet), so pair first-seen order —
        // and therefore tie-breaking — is deterministic where Python's set iteration is not.
        if memories.len() >= 3 {
            let mut pairs: Vec<(String, String)> = Vec::new();
            for m in memories {
                let content = m.content.to_lowercase();
                let mem_words: std::collections::BTreeSet<String> = word_re()
                    .find_iter(&content)
                    .map(|w| w.as_str().to_string())
                    .filter(|w| !CONTENT_STOPWORDS.contains(&w.as_str()))
                    .collect();
                for w1 in &mem_words {
                    for w2 in &mem_words {
                        if w1 < w2 {
                            pairs.push((w1.clone(), w2.clone()));
                        }
                    }
                }
            }
            for ((w1, w2), count) in most_common(&pairs, 3) {
                let confidence = (count as f64 / memories.len() as f64).min(1.0);
                if count >= 2 && confidence >= self.min_confidence {
                    let samples: Vec<String> = memories
                        .iter()
                        .filter(|m| {
                            let c = m.content.to_lowercase();
                            c.contains(&w1) && c.contains(&w2)
                        })
                        .take(3)
                        .map(|m| m.content.clone())
                        .collect();
                    patterns.push(DetectedPattern {
                        pattern_type: "content".to_string(),
                        description: format!(
                            "Co-occurring topics: '{w1}' + '{w2}' appear together {count} times"
                        ),
                        confidence,
                        samples,
                        metadata: serde_json::json!({
                            "word1": w1, "word2": w2, "count": count
                        }),
                    });
                }
            }
        }
        patterns
    }

    /// Sequence patterns: source A frequently followed by source B in timestamp order
    /// (`detect_sequence` L356-L390). Only the `timestamp` field participates (not `created_at`),
    /// exactly as in Python.
    pub fn detect_sequence(&self, memories: &[PatternMemory]) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();
        if memories.len() < 3 {
            return patterns;
        }
        let mut sorted_mems: Vec<&PatternMemory> = memories
            .iter()
            .filter(|m| m.timestamp.as_deref().is_some_and(|t| !t.is_empty()))
            .collect();
        sorted_mems.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        let sources: Vec<String> = sorted_mems
            .iter()
            .map(|m| m.source.clone().unwrap_or_else(|| "unknown".to_string()))
            .collect();
        if sources.len() < 2 {
            return patterns;
        }
        let source_pairs: Vec<(String, String)> = (0..sources.len() - 1)
            .map(|i| (sources[i].clone(), sources[i + 1].clone()))
            .collect();

        for ((s1, s2), count) in most_common(&source_pairs, 3) {
            let confidence = (count as f64 / ((sources.len() - 1) as f64).max(2.0)).min(1.0);
            if count >= 2 && confidence >= self.min_confidence {
                let mut samples = Vec::new();
                for i in 0..sources.len() - 1 {
                    if sources[i] == s1 && sources[i + 1] == s2 {
                        let head: String = sorted_mems[i].content.chars().take(50).collect();
                        let next: String = sorted_mems[i + 1].content.chars().take(50).collect();
                        samples.push(format!("{head}... -> {next}..."));
                        if samples.len() >= 2 {
                            break;
                        }
                    }
                }
                patterns.push(DetectedPattern {
                    pattern_type: "sequence".to_string(),
                    description: format!(
                        "Sequence pattern: '{s1}' often followed by '{s2}' ({count} times)"
                    ),
                    confidence,
                    samples,
                    metadata: serde_json::json!({
                        "source1": s1, "source2": s2, "count": count
                    }),
                });
            }
        }
        patterns
    }

    /// All detectors combined, sorted by confidence descending (stable — `detect_all` L392-L400).
    pub fn detect_all(&self, memories: &[PatternMemory]) -> Vec<DetectedPattern> {
        let mut patterns = self.detect_temporal(memories);
        patterns.extend(self.detect_content(memories));
        patterns.extend(self.detect_sequence(memories));
        patterns.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        patterns
    }

    /// A human-readable summary of detected patterns (`summarize_patterns` L402-L412).
    pub fn summarize_patterns(&self, memories: &[PatternMemory]) -> serde_json::Value {
        let patterns = self.detect_all(memories);
        let by_type = |t: &str| -> Vec<&DetectedPattern> {
            patterns.iter().filter(|p| p.pattern_type == t).collect()
        };
        serde_json::json!({
            "total_memories": memories.len(),
            "patterns_found": patterns.len(),
            "temporal_patterns": by_type("temporal"),
            "content_patterns": by_type("content"),
            "sequence_patterns": by_type("sequence"),
            "top_pattern": patterns.first(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(content: &str, timestamp: Option<&str>, source: Option<&str>) -> PatternMemory {
        PatternMemory {
            content: content.to_string(),
            timestamp: timestamp.map(String::from),
            created_at: None,
            source: source.map(String::from),
        }
    }

    #[test]
    fn dict_compression_strips_phrases_and_reports_savings() {
        let c = MemoryCompressor::new();
        let (out, stats) = c.compress("remember that the deadline is Friday", "dict");
        assert_eq!(out, "the deadline is Friday");
        assert!(stats.savings_percent() > 5.0);
        assert_eq!(stats.method, "dict");
    }

    #[test]
    fn dict_round_trip_restores_nonempty_tokens() {
        let c = MemoryCompressor::new();
        let (out, _) = c.compress("project context: daemon uses mnemosyne memory", "dict");
        assert!(out.starts_with('\t'));
        let restored = c.decompress(&out, "dict");
        // "project context: " and "mnemosyne " come back; ""-token phrases are unrecoverable.
        assert_eq!(restored, "project context: daemon uses mnemosyne memory");
    }

    #[test]
    fn rle_collapses_runs_and_round_trips() {
        let c = MemoryCompressor::new();
        let (out, stats) = c.compress("aaaaaabbbcccccc!", "rle");
        assert_eq!(out, "[a*6]bbb[c*6]!");
        assert_eq!(stats.method, "rle");
        assert_eq!(c.decompress(&out, "rle"), "aaaaaabbbcccccc!");
    }

    #[test]
    fn rle_run_at_end_is_encoded() {
        let c = MemoryCompressor::new();
        let (out, _) = c.compress("xyzzzzz", "rle");
        assert_eq!(out, "xy[z*5]");
    }

    #[test]
    fn auto_falls_back_to_rle_when_dict_saves_little() {
        let c = MemoryCompressor::new();
        let (out, stats) = c.compress("nothing to replace heeeeeeere", "auto");
        assert_eq!(stats.method, "rle");
        assert_eq!(out, "nothing to replace h[e*7]re");
    }

    #[test]
    fn semantic_truncates_over_500_bytes() {
        let c = MemoryCompressor::new();
        let long = "x".repeat(600);
        let (out, stats) = c.compress(&long, "semantic");
        assert_eq!(
            out,
            format!("{} [...] {}", "x".repeat(250), "x".repeat(100))
        );
        assert!(stats.compressed_size < stats.original_size);
        // Short content passes through.
        let (short, _) = c.compress("short", "semantic");
        assert_eq!(short, "short");
    }

    #[test]
    fn unknown_method_is_identity() {
        let c = MemoryCompressor::new();
        let (out, stats) = c.compress("abc", "gzip");
        assert_eq!(out, "abc");
        assert_eq!(stats.method, "none");
        assert_eq!(stats.ratio, 1.0);
    }

    #[test]
    fn batch_compression_aggregates_stats() {
        let c = MemoryCompressor::new();
        let contents = vec!["remember that a".to_string(), "remember that b".to_string()];
        let (out, stats) = c.compress_batch(&contents, "dict");
        assert_eq!(out, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(stats.memories_compressed, 2);
        assert_eq!(stats.original_size, 30);
        assert_eq!(stats.compressed_size, 2);
    }

    #[test]
    fn temporal_detects_dominant_hour_and_day() {
        let d = PatternDetector::default();
        // 4 of 5 at 09:00, all on a Monday (2026-06-01) -> hour + weekday patterns.
        let mems: Vec<PatternMemory> = [
            "2026-06-01T09:00:00",
            "2026-06-01T09:10:00",
            "2026-06-01T09:20:00",
            "2026-06-01T09:30:00",
            "2026-06-01T15:00:00",
        ]
        .iter()
        .map(|t| mem("m", Some(t), None))
        .collect();
        let patterns = d.detect_temporal(&mems);
        let hour = patterns
            .iter()
            .find(|p| p.metadata.get("hour").is_some())
            .expect("hour pattern");
        assert!((hour.confidence - 0.8).abs() < 1e-9);
        assert!(hour.description.contains("09:00"));
        assert_eq!(hour.samples.len(), 3);
        let day = patterns
            .iter()
            .find(|p| p.metadata.get("day").is_some())
            .expect("day pattern");
        assert_eq!(day.metadata["day"], "Mon");
        assert!((day.confidence - 1.0).abs() < 1e-9);
        // Fewer than 3 parseable timestamps -> nothing.
        assert!(d.detect_temporal(&mems[..2]).is_empty());
    }

    #[test]
    fn temporal_reads_created_at_fallback_and_offsets() {
        let d = PatternDetector::default();
        let mut mems = vec![
            PatternMemory {
                content: "m".into(),
                timestamp: Some(String::new()), // falsy -> falls through to created_at
                created_at: Some("2026-06-01T09:00:00+05:00".into()),
                source: None,
            };
            3
        ];
        // Offset-local hour is read without tz normalization (fromisoformat semantics).
        mems[0].created_at = Some("2026-06-01T09:00:00Z".into());
        let patterns = d.detect_temporal(&mems);
        assert!(patterns.iter().any(|p| p.metadata["hour"] == 9));
    }

    #[test]
    fn content_detects_frequent_topic_and_cooccurrence() {
        let d = PatternDetector::default();
        let mems = vec![
            mem("deploy kubernetes cluster today", None, None),
            mem("kubernetes cluster upgrade plan", None, None),
            mem("cluster kubernetes rollback notes", None, None),
        ];
        let patterns = d.detect_content(&mems);
        assert!(patterns
            .iter()
            .any(|p| p.description.contains("'kubernetes' appears 3 times")));
        let pair = patterns
            .iter()
            .find(|p| p.description.starts_with("Co-occurring"))
            .expect("co-occurrence pattern");
        assert_eq!(pair.metadata["word1"], "cluster");
        assert_eq!(pair.metadata["word2"], "kubernetes");
        assert_eq!(pair.samples.len(), 3);
    }

    #[test]
    fn sequence_detects_source_chains() {
        let d = PatternDetector::default();
        let mems = vec![
            mem(
                "plan the work",
                Some("2026-01-01T10:00:00"),
                Some("planner"),
            ),
            mem("do the work", Some("2026-01-01T11:00:00"), Some("executor")),
            mem(
                "plan more work",
                Some("2026-01-01T12:00:00"),
                Some("planner"),
            ),
            mem(
                "do more work",
                Some("2026-01-01T13:00:00"),
                Some("executor"),
            ),
        ];
        let patterns = d.detect_sequence(&mems);
        let p = patterns
            .iter()
            .find(|p| {
                p.description
                    .contains("'planner' often followed by 'executor'")
            })
            .expect("sequence pattern");
        assert_eq!(p.metadata["count"], 2);
        assert_eq!(p.samples.len(), 2);
        assert!(p.samples[0].contains(" -> "));
    }

    #[test]
    fn detect_all_sorts_by_confidence_and_summarizes() {
        let d = PatternDetector::default();
        let mems = vec![
            mem(
                "kubernetes cluster deploy",
                Some("2026-06-01T09:00:00"),
                Some("ops"),
            ),
            mem(
                "kubernetes cluster upgrade",
                Some("2026-06-01T09:10:00"),
                Some("ops"),
            ),
            mem(
                "kubernetes cluster notes",
                Some("2026-06-01T09:20:00"),
                Some("ops"),
            ),
        ];
        let all = d.detect_all(&mems);
        assert!(!all.is_empty());
        assert!(all.windows(2).all(|w| w[0].confidence >= w[1].confidence));
        let summary = d.summarize_patterns(&mems);
        assert_eq!(summary["total_memories"], 3);
        assert_eq!(summary["patterns_found"], all.len());
        assert!(summary["top_pattern"].is_object());
        assert!(summary["temporal_patterns"].as_array().is_some());
    }

    #[test]
    fn empty_input_summarize_has_null_top_pattern() {
        let d = PatternDetector::default();
        let summary = d.summarize_patterns(&[]);
        assert_eq!(summary["patterns_found"], 0);
        assert!(summary["top_pattern"].is_null());
    }
}
