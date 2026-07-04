// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! AAAK lossless shorthand — port of `aaak.py` (the dependency-free sleep-summary fallback).
//!
//! A deterministic compression scheme that shortens natural-language memory into a pipe-delimited
//! shorthand LLMs still parse, used as the summary when no extraction [`Provider`](daemon_core::Provider)
//! is available. Port of `aaak.py` `encode` (L125-L152): category prefixes -> phrase map
//! (longest-first) -> ordered structural replacements -> paren compaction -> trailing-phrase fixups.

use std::sync::OnceLock;

/// Category prefixes -> AAAK codes (`aaak.py` `CATEGORY_MAP` L11-L25).
const CATEGORY_MAP: &[(&str, &str)] = &[
    ("PREFERENCE", "PREF"),
    ("TRAIT", "TRAIT"),
    ("STATUS", "STAT"),
    ("INSTRUCTION", "INST"),
    ("PROJECT", "PROJ"),
    ("LOCATION", "LOC"),
    ("FAMILY", "FAM"),
    ("OCCUPATION", "OCC"),
    ("DECISION", "DEC"),
    ("EVENT", "EVT"),
    ("TOOL", "TOOL"),
    ("FACT", "FACT"),
    ("OPINION", "OPN"),
];

/// Common structural phrases -> compressed forms (`aaak.py` `PHRASE_MAP` L28-L58).
const PHRASE_MAP: &[(&str, &str)] = &[
    ("User asked ", "ASK "),
    ("User wants ", "WANT "),
    ("User prefers ", "PREF "),
    ("User likes ", "LIKE "),
    ("User dislikes ", "DISLIKE "),
    ("User is ", "IS "),
    ("User has ", "HAS "),
    ("User built ", "BUILT "),
    ("User asked for ", "ASK "),
    ("User requested ", "REQ "),
    ("Married to ", "MARRIED\u{2192}"),
    ("Email: ", "@"),
    ("GitHub: ", "GH:"),
    ("Location: ", "LOC:"),
    ("Phone: ", "PH:"),
    ("User email is ", "@"),
    ("User voice message ", "VM "),
    ("User stack: ", "STACK|"),
    ("Full-stack developer", "FSDEV"),
    ("Software Developer", "SDEV"),
    ("AI Systems Engineer", "AIENG"),
    ("real-time", "RT"),
    ("Real-time", "RT"),
    ("bilingual", "bi"),
    ("Bilingual", "bi"),
    ("self-hosted", "selfhost"),
    ("automation", "auto"),
    ("transcription", "transc"),
    ("translation", "transl"),
];

/// Structural replacements, applied in order (`aaak.py` `STRUCTURAL_REPLACEMENTS` L61-L90).
const STRUCTURAL_REPLACEMENTS: &[(&str, &str)] = &[
    (" - ", " | "),
    (" -- ", " | "),
    (" | ", " | "),
    (", ", " | "),
    (" and ", "+"),
    (" or ", "/"),
    (" for ", "\u{2192}"),
    (" to ", "\u{2192}"),
    (" with ", " w/ "),
    (" over ", ">"),
    (" instead of ", "!>"),
    (" because of ", "\u{2235}"),
    (" due to ", "\u{2235}"),
    (" using ", "\u{2192}"),
    (" built ", "\u{2192}"),
    (" in ", ":"),
    (" at ", "@"),
    (" on ", "@"),
    (" from ", "<-"),
];

/// Phrase map sorted by key length descending (`aaak.py` `_apply_phrases` L108 longest-first).
fn phrases_longest_first() -> &'static [(&'static str, &'static str)] {
    static P: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    P.get_or_init(|| {
        let mut v = PHRASE_MAP.to_vec();
        v.sort_by_key(|entry| std::cmp::Reverse(entry.0.len()));
        v
    })
}

/// Compress `CATEGORY: ` prefix to `CODE|` (`aaak.py` `_apply_category_prefixes` L97-L102).
fn apply_category_prefixes(text: &str) -> String {
    for (full, code) in CATEGORY_MAP {
        let prefix = format!("{full}: ");
        if let Some(rest) = text.strip_prefix(&prefix) {
            return format!("{code}|{rest}");
        }
    }
    text.to_string()
}

/// Remove spaces just inside parentheses (`aaak.py` `_compact_parens` L120-L122).
fn compact_parens(text: &str) -> String {
    // `re.sub(r"\(\s*", "(", text)` then `.replace(" )", ")")`.
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '(' {
            while matches!(chars.peek(), Some(w) if w.is_whitespace()) {
                chars.next();
            }
        }
    }
    out.replace(" )", ")")
}

/// Encode text into AAAK shorthand (`aaak.py` `encode` L125-L152).
pub fn encode(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    // Skip if it already looks like AAAK (pipe-delimited, few tokens).
    if text.contains('|') && text.split_whitespace().count() <= 3 {
        return text.to_string();
    }

    let mut result = apply_category_prefixes(text.trim());
    for (phrase, shorthand) in phrases_longest_first() {
        result = result.replace(phrase, shorthand);
    }
    for (pattern, replacement) in STRUCTURAL_REPLACEMENTS {
        result = result.replace(pattern, replacement);
    }
    result = compact_parens(&result);

    // Trailing-phrase fixups (`aaak.py` L147-L151; "complete" before "completed" is intentional).
    result = result.replace("working correctly", "OK");
    result = result.replace("working", "OK");
    result = result.replace("complete", "DONE");
    result = result.replace("completed", "DONE");

    result.trim().to_string()
}

/// Join a group of memory contents into a single deterministic AAAK summary line. Used by sleep when
/// no LLM is present (`beam.py` sleep summary fallback).
pub fn summarize_group(contents: &[String]) -> String {
    let encoded: Vec<String> = contents
        .iter()
        .map(|c| encode(c))
        .filter(|c| !c.is_empty())
        .collect();
    encoded.join(" || ")
}

/// Reverse category map with Python's dict semantics (`aaak.py` `REV_CATEGORY` L93): code -> full
/// category. Codes are unique, so insertion order is irrelevant here.
fn rev_category(code: &str) -> Option<&'static str> {
    CATEGORY_MAP
        .iter()
        .find(|(_, c)| *c == code)
        .map(|(full, _)| *full)
}

/// Reverse phrase map with Python's dict-comprehension semantics (`aaak.py` `REV_PHRASE` L94):
/// duplicate shorthand keys resolve to the LAST phrase that produced them (`{v: k}` last-wins —
/// e.g. `"ASK "` -> `"User asked for "`, `"@"` -> `"User email is "`).
fn rev_phrases() -> &'static [(&'static str, &'static str)] {
    static R: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    R.get_or_init(|| {
        let mut rev: Vec<(&'static str, &'static str)> = Vec::new();
        for (phrase, shorthand) in PHRASE_MAP {
            if let Some(slot) = rev.iter_mut().find(|(s, _)| s == shorthand) {
                slot.1 = phrase; // last-wins, like the Python dict comprehension
            } else {
                rev.push((shorthand, phrase));
            }
        }
        // Longest shorthand first so e.g. "STACK|" is restored before "|" handling by callers.
        rev.sort_by_key(|entry| std::cmp::Reverse(entry.0.len()));
        rev
    })
}

/// Best-effort AAAK decode using exactly the reverse maps Python builds (`REV_CATEGORY` /
/// `REV_PHRASE`, `aaak.py` L92-L94 — defined there but never shipped with a `decode`; the Rust
/// port completes the round-trip). Restores the `CATEGORY: ` prefix and the phrase shorthands.
/// Structural replacements (`+`, `→`, `|`, ...) are many-to-one and Python builds no reverse map
/// for them, so they are left as-is — AAAK is a "lossless *shorthand* LLMs parse without a
/// decoder", not a bijective codec.
pub fn decode(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let mut result = text.trim().to_string();
    // Category code prefix: `CODE|rest` -> `CATEGORY: rest`.
    if let Some(bar) = result.find('|') {
        let (code, rest) = result.split_at(bar);
        if let Some(full) = rev_category(code) {
            result = format!("{full}: {}", &rest[1..]);
        }
    }
    for (shorthand, phrase) in rev_phrases() {
        result = result.replace(shorthand, phrase);
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_prefix_compresses() {
        let got = encode("PREFERENCE: dark mode");
        assert!(got.starts_with("PREF|"), "got: {got}");
    }

    #[test]
    fn phrases_and_structure_compress() {
        let got = encode("User prefers Python and Rust");
        // "User prefers " -> "PREF ", " and " -> "+".
        assert!(got.contains("PREF"), "got: {got}");
        assert!(got.contains('+'), "got: {got}");
        assert!(!got.contains(" and "), "got: {got}");
    }

    #[test]
    fn list_becomes_pipe_delimited() {
        let got = encode("apples, oranges, pears");
        assert!(got.contains('|'), "got: {got}");
    }

    #[test]
    fn already_aaak_is_passthrough() {
        let aaak = "PREF|dark";
        assert_eq!(encode(aaak), aaak);
    }

    #[test]
    fn empty_is_empty() {
        assert_eq!(encode(""), "");
    }

    #[test]
    fn compact_parens_strips_inner_spaces() {
        let got = compact_parens("foo ( bar )");
        assert_eq!(got, "foo (bar)");
    }

    #[test]
    fn summarize_group_joins() {
        let got = summarize_group(&["User likes tea".into(), "User likes coffee".into()]);
        assert!(got.contains("||"), "got: {got}");
    }

    #[test]
    fn decode_restores_category_prefix() {
        assert_eq!(decode("PREF|dark mode"), "PREFERENCE: dark mode");
        assert_eq!(decode("EVT|standup moved"), "EVENT: standup moved");
        // Unknown code: left alone.
        assert_eq!(decode("NOPE|x"), "NOPE|x");
    }

    #[test]
    fn decode_reverses_phrases_with_last_wins_semantics() {
        // "ASK " maps back to "User asked for " (the LAST phrase producing it, dict semantics).
        assert_eq!(decode("ASK help"), "User asked for help");
        // "LIKE " is unambiguous.
        assert_eq!(decode("LIKE tea"), "User likes tea");
    }

    #[test]
    fn encode_decode_round_trips_phrase_content() {
        let original = "User likes tea";
        assert_eq!(decode(&encode(original)), original);
    }
}
