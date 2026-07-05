// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The multi-strategy fuzzy find-and-replace behind the `fs` tool's `edit` op — a faithful port of
//! hermes `tools/fuzzy_match.py` (itself derived from OpenCode's matching chain).
//!
//! Nine strategies are tried in order, each returning the byte spans it matched; the first strategy
//! with any match wins. LLM-emitted `old_string`s routinely drift from the file in whitespace,
//! indentation, escaping, or unicode punctuation — the chain absorbs each drift class in turn:
//!
//! 1. `exact` — direct substring search (overlapping occurrences counted, python parity).
//! 2. `line_trimmed` — per-line `trim` on both sides before block comparison.
//! 3. `whitespace_normalized` — `[ \t]+` runs collapsed to one space (newlines preserved).
//! 4. `indentation_flexible` — leading whitespace stripped per line.
//! 5. `escape_normalized` — literal `\n`/`\t`/`\r` sequences in the pattern unescaped.
//! 6. `trimmed_boundary` — only the first and last pattern lines trimmed.
//! 7. `unicode_normalized` — smart quotes / em–en dashes / ellipsis / nbsp mapped to ASCII.
//! 8. `block_anchor` — first+last lines anchor; middle similarity ≥ 0.50 (unique) / 0.70 (multi).
//! 9. `context_aware` — ≥ 50% of lines with per-line similarity ≥ 0.80.
//!
//! Post-match guards (also ported): the multi-match uniqueness rule (`replace_all`), the
//! escape-drift blocker (`\'`/`\"` serialization artifacts), the conditional `\t`/`\r` unescape of
//! `new_string`, and re-indentation of `new_string` onto the file's actual base indent after a
//! non-exact match. Similarity ratios use the [`similar`] crate (difflib-`ratio()`-shaped
//! `2*matches/total`; the underlying diff algorithm differs slightly, which is acceptable for
//! these anchoring heuristics).
//!
//! Parity note: strategies 4 and 6 are *subsumed* by strategy 2 — lstrip-equal lines are
//! strip-equal, and a byte-exact middle is trim-equal — so with the chain in this (hermes') order
//! strategy 2 always claims their matches first. This is true of the python original as well;
//! both are ported in their slots for order fidelity rather than dropped, so any future upstream
//! reordering maps 1:1.
//!
//! All spans are **byte** offsets into the original content (python uses char indices; every
//! position here is computed on the same `&str`, so byte offsets are the natural equivalent).

use std::borrow::Cow;
use std::collections::HashMap;

/// A successful fuzzy replacement.
#[derive(Debug)]
pub struct FuzzyReplace {
    /// The content with all replacements applied.
    pub content: String,
    /// How many occurrences were replaced.
    pub count: usize,
    /// The name of the strategy that matched (`"exact"`, `"line_trimmed"`, ...).
    pub strategy: &'static str,
}

/// Why a fuzzy replacement could not be performed (messages mirror hermes' error strings).
#[derive(Debug, PartialEq, Eq)]
pub enum FuzzyError {
    /// `old_string` was empty.
    EmptyOld,
    /// `old_string` and `new_string` are identical.
    Identical,
    /// Multiple matches without `replace_all`.
    Ambiguous(usize),
    /// A `\'`/`\"` tool-call serialization artifact was detected (see [`detect_escape_drift`]).
    EscapeDrift(String),
    /// No strategy matched.
    NotFound,
}

impl std::fmt::Display for FuzzyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyOld => write!(f, "old_string cannot be empty"),
            Self::Identical => write!(f, "old_string and new_string are identical"),
            Self::Ambiguous(n) => write!(
                f,
                "Found {n} matches for old_string. Provide more context to make it unique, \
                 or use replace_all: true."
            ),
            Self::EscapeDrift(msg) => write!(f, "{msg}"),
            Self::NotFound => write!(f, "Could not find a match for old_string in the file"),
        }
    }
}

/// Find `old_string` in `content` through the 9-strategy chain and replace it with `new_string`.
///
/// With `replace_all` every occurrence found by the winning strategy is replaced; without it a
/// multi-occurrence match is an [`FuzzyError::Ambiguous`] error (the caller should add context or
/// pass `replace_all`). Port of `fuzzy_find_and_replace` (fuzzy_match.py:50).
pub fn fuzzy_find_and_replace(
    content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<FuzzyReplace, FuzzyError> {
    if old_string.is_empty() {
        return Err(FuzzyError::EmptyOld);
    }
    if old_string == new_string {
        return Err(FuzzyError::Identical);
    }

    type Strategy = fn(&str, &str) -> Vec<(usize, usize)>;
    let strategies: [(&'static str, Strategy); 9] = [
        ("exact", strategy_exact),
        ("line_trimmed", strategy_line_trimmed),
        ("whitespace_normalized", strategy_whitespace_normalized),
        ("indentation_flexible", strategy_indentation_flexible),
        ("escape_normalized", strategy_escape_normalized),
        ("trimmed_boundary", strategy_trimmed_boundary),
        ("unicode_normalized", strategy_unicode_normalized),
        ("block_anchor", strategy_block_anchor),
        ("context_aware", strategy_context_aware),
    ];

    for (name, strategy) in strategies {
        let matches = strategy(content, old_string);
        if matches.is_empty() {
            continue;
        }
        if matches.len() > 1 && !replace_all {
            return Err(FuzzyError::Ambiguous(matches.len()));
        }
        // Escape-drift guard: a non-exact match means we matched via normalization; if new_string
        // carries `\'`/`\"` sequences the matched file region does not, the transport almost
        // certainly injected spurious backslashes — writing it verbatim would corrupt the file.
        if name != "exact" {
            if let Some(msg) = detect_escape_drift(content, &matches, old_string, new_string) {
                return Err(FuzzyError::EscapeDrift(msg));
            }
        }
        // Conditionally unescape `\t`/`\r` in new_string when the matched region carries the real
        // control character (`\n` deliberately excluded — it serializes correctly through JSON).
        let effective_new = maybe_unescape_new_string(new_string, content, &matches);
        let reindent_against = (name != "exact").then_some(old_string);
        let new_content = apply_replacements(content, &matches, &effective_new, reindent_against);
        return Ok(FuzzyReplace {
            content: new_content,
            count: matches.len(),
            strategy: name,
        });
    }

    Err(FuzzyError::NotFound)
}

// ---------------------------------------------------------------------------------------------
// Post-match guards + replacement (fuzzy_match.py:147-336)
// ---------------------------------------------------------------------------------------------

/// Detect `\'`/`\"` tool-call escape-drift artifacts in `new_string` (fuzzy_match.py:147).
///
/// Fires when a suspect sequence is present in **both** `old_string` and `new_string` (copied
/// "context" the model meant to preserve) but absent from the matched file region — the signature
/// of a transport layer that backslash-escaped quotes in flight.
fn detect_escape_drift(
    content: &str,
    matches: &[(usize, usize)],
    old_string: &str,
    new_string: &str,
) -> Option<String> {
    if !new_string.contains("\\'") && !new_string.contains("\\\"") {
        return None;
    }
    let matched_regions: String = matches.iter().map(|&(s, e)| &content[s..e]).collect();
    for suspect in ["\\'", "\\\""] {
        if new_string.contains(suspect)
            && old_string.contains(suspect)
            && !matched_regions.contains(suspect)
        {
            let plain = &suspect[1..];
            return Some(format!(
                "Escape-drift detected: old_string and new_string contain the literal sequence \
                 {suspect:?} but the matched region of the file does not. This is almost always a \
                 tool-call serialization artifact where an apostrophe or quote got prefixed with a \
                 spurious backslash. Re-read the file and pass old_string/new_string without \
                 backslash-escaping {plain:?} characters."
            ));
        }
    }
    None
}

/// Conditionally unescape literal `\t`/`\r` sequences in `new_string` (fuzzy_match.py:271):
/// only when the matched file region actually contains the corresponding control character, so a
/// file that legitimately holds the two-character string `"\t"` is left untouched.
fn maybe_unescape_new_string(
    new_string: &str,
    content: &str,
    matches: &[(usize, usize)],
) -> String {
    if !new_string.contains("\\t") && !new_string.contains("\\r") {
        return new_string.to_string();
    }
    let matched_regions: String = matches.iter().map(|&(s, e)| &content[s..e]).collect();
    let mut out = new_string.to_string();
    if out.contains("\\t") && matched_regions.contains('\t') {
        out = out.replace("\\t", "\t");
    }
    if out.contains("\\r") && matched_regions.contains('\r') {
        out = out.replace("\\r", "\r");
    }
    out
}

/// The leading space/tab prefix of a line (fuzzy_match.py:187).
fn leading_whitespace(line: &str) -> &str {
    let end = line
        .char_indices()
        .find(|&(_, c)| c != ' ' && c != '\t')
        .map_or(line.len(), |(i, _)| i);
    &line[..end]
}

/// The first line with non-whitespace content, if any (fuzzy_match.py:195).
fn first_meaningful_line(text: &str) -> Option<&str> {
    text.split('\n').find(|line| !line.trim().is_empty())
}

/// Re-anchor `new_string`'s indentation onto the file's actual base indent after a non-exact
/// match (fuzzy_match.py:206): swap the LLM's base indent prefix for the file's, preserving the
/// relative nesting the LLM intended (the Roo Code approach).
fn reindent_replacement(file_region: &str, old_string: &str, new_string: &str) -> String {
    if new_string.is_empty() {
        return new_string.to_string();
    }
    let (Some(old_first), Some(file_first)) = (
        first_meaningful_line(old_string),
        first_meaningful_line(file_region),
    ) else {
        return new_string.to_string();
    };
    let old_indent = leading_whitespace(old_first);
    let file_indent = leading_whitespace(file_first);
    if old_indent == file_indent {
        return new_string.to_string();
    }
    let mut out_lines: Vec<String> = Vec::new();
    for line in new_string.split('\n') {
        if line.trim().is_empty() {
            out_lines.push(line.to_string());
        } else if let Some(remainder) = line.strip_prefix(old_indent) {
            out_lines.push(format!("{file_indent}{remainder}"));
        } else {
            // Less-indented than the LLM's base (a dedent) — anchor to the file's base.
            out_lines.push(format!(
                "{file_indent}{}",
                line.trim_start_matches([' ', '\t'])
            ));
        }
    }
    out_lines.join("\n")
}

/// Apply `new_string` at every matched span, back-to-front so earlier offsets stay valid
/// (fuzzy_match.py:307). `reindent_against = Some(old_string)` signals a non-exact match whose
/// replacement must be re-anchored to each region's actual indentation.
fn apply_replacements(
    content: &str,
    matches: &[(usize, usize)],
    new_string: &str,
    reindent_against: Option<&str>,
) -> String {
    let mut sorted: Vec<(usize, usize)> = matches.to_vec();
    sorted.sort_by_key(|&(start, _)| std::cmp::Reverse(start));
    let mut result = content.to_string();
    for (start, end) in sorted {
        let adjusted = match reindent_against {
            Some(old) => reindent_replacement(&content[start..end], old, new_string),
            None => new_string.to_string(),
        };
        result.replace_range(start..end, &adjusted);
    }
    result
}

// ---------------------------------------------------------------------------------------------
// Shared helpers (fuzzy_match.py:650-777)
// ---------------------------------------------------------------------------------------------

/// Every occurrence of `pattern` in `content`, **overlapping** occurrences included — python's
/// `_strategy_exact` advances one character past each match start, and the uniqueness guard
/// counts what it counts.
fn find_all_overlapping(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if pattern.is_empty() {
        return out;
    }
    let mut from = 0;
    while let Some(rel) = content[from..].find(pattern) {
        let pos = from + rel;
        out.push((pos, pos + pattern.len()));
        from = pos + content[pos..].chars().next().map_or(1, char::len_utf8);
    }
    out
}

/// Byte span of the half-open line window `[start_line, end_line)` in the original content
/// (fuzzy_match.py:650 `_calculate_line_positions`): the end excludes the window's trailing
/// newline, clamped to the content length for a final line without one.
fn line_block_span(
    lines: &[&str],
    start_line: usize,
    end_line: usize,
    content_len: usize,
) -> (usize, usize) {
    let start_pos: usize = lines[..start_line].iter().map(|l| l.len() + 1).sum();
    let end_pos: usize = lines[..end_line]
        .iter()
        .map(|l| l.len() + 1)
        .sum::<usize>()
        .saturating_sub(1);
    (start_pos, end_pos.min(content_len))
}

/// Slide a normalized pattern-line window over normalized content lines; spans come from the
/// **original** lines (fuzzy_match.py:669 `_find_normalized_matches`).
fn line_window_matches(
    content: &str,
    content_lines: &[&str],
    content_norm: &[String],
    pattern_norm: &[String],
) -> Vec<(usize, usize)> {
    let n = pattern_norm.len();
    let mut out = Vec::new();
    if n == 0 || content_norm.len() < n {
        return out;
    }
    for i in 0..=(content_norm.len() - n) {
        if content_norm[i..i + n] == pattern_norm[..] {
            out.push(line_block_span(content_lines, i, i + n, content.len()));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Strategies 1-9 (fuzzy_match.py:343-643)
// ---------------------------------------------------------------------------------------------

/// Strategy 1: exact substring match (fuzzy_match.py:343).
fn strategy_exact(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    find_all_overlapping(content, pattern)
}

/// Strategy 2: per-line `trim` on both sides before block comparison (fuzzy_match.py:356).
fn strategy_line_trimmed(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let content_lines: Vec<&str> = content.split('\n').collect();
    let content_norm: Vec<String> = content_lines.iter().map(|l| l.trim().to_string()).collect();
    let pattern_norm: Vec<String> = pattern.split('\n').map(|l| l.trim().to_string()).collect();
    line_window_matches(content, &content_lines, &content_norm, &pattern_norm)
}

/// Collapse `[ \t]+` runs to a single space, preserving newlines (fuzzy_match.py:380).
fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_run = false;
    for c in s.chars() {
        if c == ' ' || c == '\t' {
            if !in_run {
                out.push(' ');
                in_run = true;
            }
        } else {
            in_run = false;
            out.push(c);
        }
    }
    out
}

/// Strategy 3: whitespace-normalized match — find in the collapsed strings, then map the spans
/// back to the original bytes (fuzzy_match.py:376 + `_map_normalized_positions`:704).
fn strategy_whitespace_normalized(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let pattern_norm = collapse_spaces(pattern);
    let content_norm = collapse_spaces(content);
    let norm_matches = find_all_overlapping(&content_norm, &pattern_norm);
    if norm_matches.is_empty() {
        return Vec::new();
    }
    map_normalized_positions(content, &content_norm, &norm_matches)
}

/// Map spans in the whitespace-collapsed string back to the original (fuzzy_match.py:704):
/// a char-by-char walk that pins every original whitespace-run character to the single collapsed
/// space, then inverts the map and expands trailing whitespace. Best-effort, exactly like python.
fn map_normalized_positions(
    original: &str,
    normalized: &str,
    norm_matches: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let orig_chars: Vec<(usize, char)> = original.char_indices().collect();
    let norm_chars: Vec<(usize, char)> = normalized.char_indices().collect();

    // orig_to_norm[k] = the normalized byte position original char k maps to.
    let mut orig_to_norm: Vec<usize> = Vec::with_capacity(orig_chars.len());
    let mut oi = 0usize;
    let mut ni = 0usize;
    while oi < orig_chars.len() && ni < norm_chars.len() {
        let (norm_byte, nc) = norm_chars[ni];
        let (_, oc) = orig_chars[oi];
        if oc == nc {
            orig_to_norm.push(norm_byte);
            oi += 1;
            ni += 1;
        } else if (oc == ' ' || oc == '\t') && nc == ' ' {
            // A whitespace run collapsed to this one space: consume original whitespace without
            // advancing the normalized cursor until the run ends.
            orig_to_norm.push(norm_byte);
            oi += 1;
            let next_ws = orig_chars
                .get(oi)
                .is_some_and(|&(_, c)| c == ' ' || c == '\t');
            if !next_ws {
                ni += 1;
            }
        } else {
            // Extra whitespace in the original, or a mismatch that "shouldn't happen" — pin to
            // the current normalized position and move on (python's fallthrough arms).
            orig_to_norm.push(norm_byte);
            oi += 1;
        }
    }
    while oi < orig_chars.len() {
        orig_to_norm.push(normalized.len());
        oi += 1;
    }

    // Invert: for each normalized position, the first/last original char index mapping to it.
    let mut norm_to_orig_start: HashMap<usize, usize> = HashMap::new();
    let mut norm_to_orig_end: HashMap<usize, usize> = HashMap::new();
    for (k, &npos) in orig_to_norm.iter().enumerate() {
        norm_to_orig_start.entry(npos).or_insert(k);
        norm_to_orig_end.insert(npos, k);
    }
    let char_end_byte = |k: usize| -> usize {
        orig_chars
            .get(k + 1)
            .map_or(original.len(), |&(byte, _)| byte)
    };

    let mut out = Vec::new();
    for &(norm_start, norm_end) in norm_matches {
        let orig_start_char = match norm_to_orig_start.get(&norm_start) {
            Some(&k) => k,
            None => match orig_to_norm.iter().position(|&n| n >= norm_start) {
                Some(k) => k,
                None => continue,
            },
        };
        let orig_start = orig_chars[orig_start_char].0;
        // The normalized byte position of the match's final character.
        let last_char_norm = normalized[norm_start..norm_end]
            .char_indices()
            .next_back()
            .map(|(i, _)| norm_start + i);
        let mut orig_end = match last_char_norm.and_then(|p| norm_to_orig_end.get(&p)) {
            Some(&k) => char_end_byte(k),
            None => (orig_start + (norm_end - norm_start)).min(original.len()),
        };
        // Expand over trailing whitespace that the normalization collapsed away.
        let bytes = original.as_bytes();
        while orig_end < original.len() && (bytes[orig_end] == b' ' || bytes[orig_end] == b'\t') {
            orig_end += 1;
        }
        out.push((orig_start, orig_end.min(original.len())));
    }
    out
}

/// Strategy 4: indentation ignored entirely — per-line `trim_start` (fuzzy_match.py:397).
fn strategy_indentation_flexible(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let content_lines: Vec<&str> = content.split('\n').collect();
    let content_norm: Vec<String> = content_lines
        .iter()
        .map(|l| l.trim_start().to_string())
        .collect();
    let pattern_norm: Vec<String> = pattern
        .split('\n')
        .map(|l| l.trim_start().to_string())
        .collect();
    line_window_matches(content, &content_lines, &content_norm, &pattern_norm)
}

/// Strategy 5: unescape literal `\n`/`\t`/`\r` in the pattern, then exact match
/// (fuzzy_match.py:413). Skipped when the pattern carries no escapes.
fn strategy_escape_normalized(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let unescaped = pattern
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r");
    if unescaped == pattern {
        return Vec::new();
    }
    find_all_overlapping(content, &unescaped)
}

/// Strategy 6: trim only the first and last pattern lines, then slide a line window
/// (fuzzy_match.py:432).
fn strategy_trimmed_boundary(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let mut pattern_lines: Vec<String> = pattern.split('\n').map(str::to_string).collect();
    if pattern_lines.is_empty() {
        return Vec::new();
    }
    pattern_lines[0] = pattern_lines[0].trim().to_string();
    if pattern_lines.len() > 1 {
        let last = pattern_lines.len() - 1;
        pattern_lines[last] = pattern_lines[last].trim().to_string();
    }
    let content_lines: Vec<&str> = content.split('\n').collect();
    let n = pattern_lines.len();
    let mut out = Vec::new();
    if content_lines.len() < n {
        return out;
    }
    for i in 0..=(content_lines.len() - n) {
        let block = &content_lines[i..i + n];
        let mut check: Vec<String> = block.iter().map(|s| s.to_string()).collect();
        check[0] = check[0].trim().to_string();
        if check.len() > 1 {
            let last = check.len() - 1;
            check[last] = check[last].trim().to_string();
        }
        if check[..] == pattern_lines[..] {
            out.push(line_block_span(&content_lines, i, i + n, content.len()));
        }
    }
    out
}

/// The unicode → ASCII normalization table (fuzzy_match.py:36 `UNICODE_MAP`).
fn unicode_repl(c: char) -> Option<&'static str> {
    match c {
        '\u{201c}' | '\u{201d}' => Some("\""), // smart double quotes
        '\u{2018}' | '\u{2019}' => Some("'"),  // smart single quotes
        '\u{2014}' => Some("--"),              // em dash
        '\u{2013}' => Some("-"),               // en dash
        '\u{2026}' => Some("..."),             // ellipsis
        '\u{00a0}' => Some(" "),               // non-breaking space
        _ => None,
    }
}

/// Normalize smart punctuation to ASCII (fuzzy_match.py:43).
fn unicode_normalize(text: &str) -> Cow<'_, str> {
    if !text.chars().any(|c| unicode_repl(c).is_some()) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match unicode_repl(c) {
            Some(repl) => out.push_str(repl),
            None => out.push(c),
        }
    }
    Cow::Owned(out)
}

/// Strategy 7: unicode-normalize both sides, match exactly (then line-trimmed) on the normalized
/// copies, and map the spans back through an expansion-aware byte map (fuzzy_match.py:524; some
/// replacements expand one char into several ASCII chars, so a naive position copy is wrong).
fn strategy_unicode_normalized(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let norm_pattern = unicode_normalize(pattern);
    let norm_content = unicode_normalize(content);
    if matches!(norm_content, Cow::Borrowed(_)) && matches!(norm_pattern, Cow::Borrowed(_)) {
        return Vec::new();
    }

    let mut norm_matches = find_all_overlapping(&norm_content, &norm_pattern);
    if norm_matches.is_empty() {
        norm_matches = strategy_line_trimmed(&norm_content, &norm_pattern);
    }
    if norm_matches.is_empty() {
        return Vec::new();
    }

    // (orig byte, norm byte) per original char, plus a sentinel (fuzzy_match.py:474).
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut norm_pos = 0usize;
    for (orig_byte, c) in content.char_indices() {
        pairs.push((orig_byte, norm_pos));
        norm_pos += unicode_repl(c).map_or(c.len_utf8(), str::len);
    }
    pairs.push((content.len(), norm_pos));

    let mut out = Vec::new();
    for (norm_start, norm_end) in norm_matches {
        // The first original char whose normalized position is exactly the match start; python
        // skips the match when no char lands there (fuzzy_match.py:509-511).
        let Some(start_idx) = pairs[..pairs.len() - 1]
            .iter()
            .position(|&(_, n)| n == norm_start)
        else {
            continue;
        };
        let orig_start = pairs[start_idx].0;
        // Walk forward while the char still begins before the normalized end.
        let mut end_idx = start_idx;
        while end_idx < pairs.len() - 1 && pairs[end_idx].1 < norm_end {
            end_idx += 1;
        }
        out.push((orig_start, pairs[end_idx].0));
    }
    out
}

/// difflib-`ratio()`-shaped similarity over characters (`2*matches/total`; 1.0 for two empties).
fn ratio(a: &str, b: &str) -> f64 {
    f64::from(similar::TextDiff::from_chars(a, b).ratio())
}

/// Strategy 8: anchor on the first + last pattern lines (unicode-normalized, trimmed), score the
/// middle with a similarity ratio (fuzzy_match.py:555). Threshold 0.50 for a unique candidate,
/// 0.70 when several candidates compete; ≤2-line patterns have no middle and score 1.0. Spans are
/// computed from the ORIGINAL lines (normalization can change byte lengths, never line counts).
fn strategy_block_anchor(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let norm_pattern = unicode_normalize(pattern);
    let norm_content = unicode_normalize(content);
    let pattern_lines: Vec<&str> = norm_pattern.split('\n').collect();
    if pattern_lines.len() < 2 {
        return Vec::new();
    }
    let first_line = pattern_lines[0].trim();
    let last_line = pattern_lines[pattern_lines.len() - 1].trim();
    let norm_content_lines: Vec<&str> = norm_content.split('\n').collect();
    let orig_content_lines: Vec<&str> = content.split('\n').collect();
    let n = pattern_lines.len();
    if norm_content_lines.len() < n {
        return Vec::new();
    }

    let potential: Vec<usize> = (0..=(norm_content_lines.len() - n))
        .filter(|&i| {
            norm_content_lines[i].trim() == first_line
                && norm_content_lines[i + n - 1].trim() == last_line
        })
        .collect();

    let threshold = if potential.len() == 1 { 0.50 } else { 0.70 };
    let pattern_middle = pattern_lines[1..n - 1].join("\n");

    let mut out = Vec::new();
    for i in potential {
        let similarity = if n <= 2 {
            1.0
        } else {
            let content_middle = norm_content_lines[i + 1..i + n - 1].join("\n");
            ratio(&content_middle, &pattern_middle)
        };
        if similarity >= threshold {
            out.push(line_block_span(
                &orig_content_lines,
                i,
                i + n,
                content.len(),
            ));
        }
    }
    out
}

/// Strategy 9: per-line similarity — a window matches when at least 50% of its lines score ≥ 0.80
/// against the corresponding (trimmed) pattern line (fuzzy_match.py:611).
fn strategy_context_aware(content: &str, pattern: &str) -> Vec<(usize, usize)> {
    let pattern_lines: Vec<&str> = pattern.split('\n').collect();
    let content_lines: Vec<&str> = content.split('\n').collect();
    let n = pattern_lines.len();
    let mut out = Vec::new();
    if n == 0 || content_lines.len() < n {
        return out;
    }
    for i in 0..=(content_lines.len() - n) {
        let high = pattern_lines
            .iter()
            .zip(&content_lines[i..i + n])
            .filter(|(p, c)| ratio(p.trim(), c.trim()) >= 0.80)
            .count();
        #[allow(clippy::cast_precision_loss)] // line counts are far below 2^52
        if high as f64 >= n as f64 * 0.5 {
            out.push(line_block_span(&content_lines, i, i + n, content.len()));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// "Did you mean?" no-match feedback (fuzzy_match.py:780-860)
// ---------------------------------------------------------------------------------------------

/// The lines of `content` most similar to `old_string`'s first meaningful line, rendered with
/// two context lines each — the "did you mean?" snippet appended to a plain no-match error
/// (fuzzy_match.py:780 `find_closest_lines`). `None` when nothing scores above 0.3.
pub fn find_closest_lines(old_string: &str, content: &str) -> Option<String> {
    const CONTEXT_LINES: usize = 2;
    const MAX_RESULTS: usize = 3;
    if old_string.is_empty() || content.is_empty() {
        return None;
    }
    let old_lines: Vec<&str> = old_string.lines().collect();
    let content_lines: Vec<&str> = content.lines().collect();
    let anchor = old_lines.iter().map(|l| l.trim()).find(|l| !l.is_empty())?;

    let mut scored: Vec<(f64, usize)> = content_lines
        .iter()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| (ratio(anchor, line.trim()), i))
        .filter(|&(r, _)| r > 0.3)
        .collect();
    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut parts: Vec<String> = Vec::new();
    let mut seen: Vec<(usize, usize)> = Vec::new();
    for &(_, line_idx) in scored.iter().take(MAX_RESULTS) {
        let start = line_idx.saturating_sub(CONTEXT_LINES);
        let end = (line_idx + old_lines.len() + CONTEXT_LINES).min(content_lines.len());
        if seen.contains(&(start, end)) {
            continue;
        }
        seen.push((start, end));
        let snippet: Vec<String> = (start..end)
            .map(|j| format!("{:4}| {}", j + 1, content_lines[j]))
            .collect();
        parts.push(snippet.join("\n"));
    }
    (!parts.is_empty()).then(|| parts.join("\n---\n"))
}

/// A `\n\nDid you mean...` suffix for plain not-found failures only (fuzzy_match.py:842):
/// ambiguous / escape-drift / identical errors failed for unrelated reasons and would be misled
/// by a similarity hint.
pub fn format_no_match_hint(error: &FuzzyError, old_string: &str, content: &str) -> String {
    if *error != FuzzyError::NotFound {
        return String::new();
    }
    match find_closest_lines(old_string, content) {
        Some(hint) => format!("\n\nDid you mean one of these sections?\n{hint}"),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------------------------

    #[test]
    fn overlapping_occurrences_are_counted() {
        // python parity: "aa" occurs twice in "aaa" (overlap), driving the uniqueness guard.
        assert_eq!(find_all_overlapping("aaa", "aa"), vec![(0, 2), (1, 3)]);
        // multibyte safety: advancing one char past a match start must not split a codepoint.
        assert_eq!(find_all_overlapping("é é", "é"), vec![(0, 2), (3, 5)]);
    }

    #[test]
    fn line_block_span_excludes_trailing_newline() {
        let content = "aa\nbb\ncc";
        let lines: Vec<&str> = content.split('\n').collect();
        assert_eq!(line_block_span(&lines, 0, 1, content.len()), (0, 2));
        assert_eq!(line_block_span(&lines, 1, 3, content.len()), (3, 8));
    }

    #[test]
    fn collapse_spaces_preserves_newlines() {
        assert_eq!(collapse_spaces("a  \t b\n\tc"), "a b\n c");
    }

    // -- the whitespace position map ------------------------------------------------------------

    #[test]
    fn whitespace_map_expands_trailing_run() {
        let content = "let  x   =  1;\nnext";
        let m = strategy_whitespace_normalized(content, "let x = 1;");
        assert_eq!(m, vec![(0, 14)], "the full original run is covered");
    }

    // -- unicode map ----------------------------------------------------------------------------

    #[test]
    fn unicode_positions_map_through_expansion() {
        // The em dash (3 bytes) normalizes to "--" (2 bytes): spans after it must still land on
        // the original bytes.
        let content = "a \u{2014} b\nkeep";
        let m = strategy_unicode_normalized(content, "a -- b");
        assert_eq!(m, vec![(0, 7)]);
        assert_eq!(&content[0..7], "a \u{2014} b");
    }

    // -- reindent -------------------------------------------------------------------------------

    #[test]
    fn reindent_swaps_base_prefix_and_anchors_dedents() {
        let file_region = "    if x:\n        y()";
        let old = "  if x:\n    y()";
        let new = "  if x:\n    z()\nelse: pass";
        let out = reindent_replacement(file_region, old, new);
        assert_eq!(out, "    if x:\n      z()\n    else: pass");
    }

    #[test]
    fn reindent_noop_when_bases_match_or_blank() {
        assert_eq!(reindent_replacement("  a", "  a", "  b"), "  b");
        assert_eq!(reindent_replacement("", "x", "y"), "y");
    }
}
