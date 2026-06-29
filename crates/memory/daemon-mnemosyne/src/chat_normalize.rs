// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Chat text normalization for Mnemosyne ingestion (NAI-1) — port of `chat_normalize.py`.
//!
//! Aggressive, all-algorithmic (zero-LLM) regex normalization that makes casual chat messages
//! parseable by structured extraction tools. This is a faithful, behavior-neutral port: like the
//! Python module it is **not wired into the ingest path** (upstream only `tools/bench_nai1.py` calls
//! it), it lives here so the capability is available and tested.

use regex::Regex;
use std::sync::OnceLock;

/// Contraction expansion table, matched against word boundaries (`chat_normalize.py` L15-L37).
/// Applied in order; earlier rules win (e.g. `\bu\b` rewrites the `u` in `u're` before `\bu're\b`
/// can fire — verbatim with the Python list ordering).
const CONTRACTIONS: &[(&str, &str)] = &[
    (r"\bu\b", "you"),
    (r"\bur\b", "your"),
    (r"\bu're\b", "you are"),
    (r"\br\b", "are"),
    (r"\by\b", "why"),
    (r"\bb4\b", "before"),
    (r"\bbc\b", "because"),
    (r"\bcuz\b", "because"),
    (r"\bgonna\b", "going to"),
    (r"\bwanna\b", "want to"),
    (r"\bgotta\b", "got to"),
    (r"\bkinda\b", "kind of"),
    (r"\bsorta\b", "sort of"),
    (r"\bdunno\b", "don't know"),
    (r"\blemme\b", "let me"),
    (r"\bgimme\b", "give me"),
    (r"\boutta\b", "out of"),
    (r"\bhafta\b", "have to"),
    (r"\bshoulda\b", "should have"),
    (r"\bwoulda\b", "would have"),
    (r"\bcoulda\b", "could have"),
];

/// Filler / reaction words stripped before fragment detection (`chat_normalize.py` L40-L45).
const FILLER_WORDS: &[&str] = &[
    "lol", "lmao", "lmaoo", "lmfao", "rofl", "omg", "omgg", "omggg", "brb", "idk", "idc", "tbh",
    "imo", "imho", "fwiw", "irl", "afaik", "iirc", "tldr", "nvm", "ikr", "wtf", "smh", "fr", "ngl",
    "istg", "w", "wdym",
];

/// Fragment-starting verbs that take an implicit subject (`chat_normalize.py` L48-L53).
const FRAGMENT_STARTERS: &[&str] = &[
    "going",
    "coming",
    "thinking",
    "wondering",
    "feeling",
    "trying",
    "hoping",
    "planning",
    "working",
    "looking",
    "checking",
    "running",
    "testing",
    "building",
    "fixing",
    "deploying",
];

fn contraction_rules() -> &'static [(Regex, &'static str)] {
    static RULES: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    RULES.get_or_init(|| {
        CONTRACTIONS
            .iter()
            .map(|(pat, rep)| (Regex::new(pat).expect("valid contraction regex"), *rep))
            .collect()
    })
}

fn non_ascii_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\x00-\x7F]+").expect("valid non-ascii regex"))
}

/// Aggressive regex normalization for casual chat messages (`chat_normalize.py` L56-L120).
///
/// Returns `None` if the message has no extractable meaning (empty, only filler/reactions, or too
/// short after normalization). Processing order: lowercase -> expand contractions -> strip filler
/// -> collapse 3+ repeated chars -> drop non-ASCII -> normalize whitespace -> fragment gate ->
/// optional implicit-subject injection.
pub fn normalize_chat(text: &str, add_implicit_subjects: bool) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }

    // Step 1: lowercase + trim.
    let mut text = text.to_lowercase();
    text = text.trim().to_string();

    // Step 2: expand contractions (word-boundary regex), in list order.
    for (re, rep) in contraction_rules() {
        text = re.replace_all(&text, *rep).into_owned();
    }

    // Step 3: strip filler / reaction words (after trimming surrounding punctuation).
    let meaningful: Vec<&str> = text
        .split_whitespace()
        .filter(|w| {
            let trimmed = w.trim_matches(|c| ".,!?;:'\"".contains(c));
            !FILLER_WORDS.contains(&trimmed)
        })
        .collect();
    if meaningful.is_empty() {
        return None;
    }
    text = meaningful.join(" ");

    // Step 4: collapse runs of 3+ identical chars to 1 (Python `(.)\1{2,}` -> `\1`).
    text = collapse_repeats(&text);

    // Step 5: remove emojis / non-ASCII (each maximal run -> single space).
    text = non_ascii_re().replace_all(&text, " ").into_owned();

    // Step 6: normalize whitespace.
    text = text.split_whitespace().collect::<Vec<_>>().join(" ");

    // Step 7: fragment detection — need at least 2 meaningful words.
    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len();
    if word_count < 2 {
        // A single long token might be a name / tool / endpoint.
        if word_count == 1 && words[0].chars().count() > 5 {
            return Some(text);
        }
        return None;
    }

    // Step 8: implicit-subject injection (only true 2-word fragments).
    if add_implicit_subjects && word_count == 2 && FRAGMENT_STARTERS.contains(&words[0]) {
        text = format!("i am {text}");
    }

    Some(text)
}

/// Collapse maximal runs of 3+ identical characters down to a single character; runs of length 1
/// or 2 are preserved (Python regex `(.)\1{2,}` replaced with `\1`).
fn collapse_repeats(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        let mut run = 1usize;
        while chars.peek() == Some(&ch) {
            chars.next();
            run += 1;
        }
        if run >= 3 {
            out.push(ch);
        } else {
            for _ in 0..run {
                out.push(ch);
            }
        }
    }
    out
}

/// Normalize a batch of messages, returning `None` for unparseable entries
/// (`chat_normalize.py` L123-L128).
pub fn normalize_batch(messages: &[&str]) -> Vec<Option<String>> {
    messages.iter().map(|m| normalize_chat(m, true)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_whitespace_return_none() {
        assert_eq!(normalize_chat("", true), None);
        assert_eq!(normalize_chat("   \t", true), None);
    }

    #[test]
    fn only_filler_returns_none() {
        assert_eq!(normalize_chat("lol omg lmao", true), None);
    }

    #[test]
    fn contractions_are_expanded() {
        assert_eq!(
            normalize_chat("gonna deploy soon", true).as_deref(),
            Some("going to deploy soon")
        );
    }

    #[test]
    fn single_letter_contractions() {
        // `u` -> `you`, `r` -> `are`
        assert_eq!(
            normalize_chat("u r late", true).as_deref(),
            Some("you are late")
        );
    }

    #[test]
    fn repeated_chars_collapse() {
        // "omgggg" (4 g) is not in the filler set; collapses to "omg", then single short word -> None.
        assert_eq!(normalize_chat("omgggg", true), None);
        // "soooo cool" -> "so cool"
        assert_eq!(
            normalize_chat("soooo cool", true).as_deref(),
            Some("so cool")
        );
    }

    #[test]
    fn non_ascii_stripped() {
        assert_eq!(
            normalize_chat("deploy done \u{1F389} now", true).as_deref(),
            Some("deploy done now")
        );
    }

    #[test]
    fn single_long_word_survives() {
        assert_eq!(
            normalize_chat("kubernetes", true).as_deref(),
            Some("kubernetes")
        );
        // short single word -> None
        assert_eq!(normalize_chat("hey", true), None);
    }

    #[test]
    fn implicit_subject_injected_for_two_word_fragment() {
        assert_eq!(
            normalize_chat("deploying now", true).as_deref(),
            Some("i am deploying now")
        );
        // disabled -> no injection
        assert_eq!(
            normalize_chat("deploying now", false).as_deref(),
            Some("deploying now")
        );
        // 3 words -> not a fragment, no injection
        assert_eq!(
            normalize_chat("deploying right now", true).as_deref(),
            Some("deploying right now")
        );
    }

    #[test]
    fn filler_with_punctuation_is_stripped() {
        assert_eq!(
            normalize_chat("lol, the build passed", true).as_deref(),
            Some("the build passed")
        );
    }

    #[test]
    fn batch_maps_each_message() {
        let out = normalize_batch(&["lol", "gonna ship it"]);
        assert_eq!(out[0], None);
        assert_eq!(out[1].as_deref(), Some("going to ship it"));
    }
}
