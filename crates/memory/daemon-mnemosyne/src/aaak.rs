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
}
