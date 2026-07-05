// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The fuzzy-match corpus: one table-driven case per strategy of the 9-strategy chain (hermes
//! `fuzzy_match.py` parity), each proving both that the strategy fires on its drift class and
//! that it reports the expected strategy name — i.e. no *earlier* strategy claimed the match.
//! Guard behaviors (multi-match, escape drift, conditional unescape, re-indent, did-you-mean)
//! get their own cases.

use daemon_tool_fs::fuzzy::{
    find_closest_lines, format_no_match_hint, fuzzy_find_and_replace, FuzzyError,
};

struct Case {
    name: &'static str,
    content: &'static str,
    old: &'static str,
    new: &'static str,
    replace_all: bool,
    want_strategy: &'static str,
    want_content: &'static str,
    want_count: usize,
}

#[test]
fn strategy_corpus() {
    let cases = [
        Case {
            name: "1 exact",
            content: "fn main() {\n    println!(\"hi\");\n}\n",
            old: "println!(\"hi\");",
            new: "println!(\"bye\");",
            replace_all: false,
            want_strategy: "exact",
            want_content: "fn main() {\n    println!(\"bye\");\n}\n",
            want_count: 1,
        },
        Case {
            // Trailing whitespace on the file's first line defeats exact; per-line trim matches.
            name: "2 line_trimmed",
            content: "alpha  \nbeta\n",
            old: "alpha\nbeta",
            new: "ALPHA\nBETA",
            replace_all: false,
            want_strategy: "line_trimmed",
            want_content: "ALPHA\nBETA\n",
            want_count: 1,
        },
        Case {
            // Interior double spaces collapse; the matched span covers the original run.
            name: "3 whitespace_normalized",
            content: "let  x   =  1;\nnext\n",
            old: "let x = 1;",
            new: "let x = 2;",
            replace_all: false,
            want_strategy: "whitespace_normalized",
            want_content: "let x = 2;\nnext\n",
            want_count: 1,
        },
        Case {
            // Indent-only drift (model sent 0-indent, file has 2). NOTE (parity finding, true of
            // hermes as well): any `indentation_flexible` (strategy 4) match is also a
            // `line_trimmed` (strategy 2) match — lstrip-equal lines are strip-equal — so with
            // the chain in hermes' order, strategy 2 always claims these. The strategy is ported
            // in its slot for order fidelity; the WINNER here is line_trimmed, and the non-exact
            // re-indent anchors the replacement onto the file's 2-space base either way.
            name: "4 indentation drift (subsumed by line_trimmed)",
            content: "  a();\n  b();\n",
            old: "a();\nb();",
            new: "a2();\nb2();",
            replace_all: false,
            want_strategy: "line_trimmed",
            want_content: "  a2();\n  b2();\n",
            want_count: 1,
        },
        Case {
            // The pattern's newline arrived as a literal backslash-n.
            name: "5 escape_normalized",
            content: "one\ntwo\nthree\n",
            old: "one\\ntwo",
            new: "ONE\nTWO",
            replace_all: false,
            want_strategy: "escape_normalized",
            want_content: "ONE\nTWO\nthree\n",
            want_count: 1,
        },
        Case {
            // Boundary-only whitespace drift (trailing spaces on the file's first line defeat
            // exact). NOTE (parity finding, true of hermes as well): every `trimmed_boundary`
            // (strategy 6) match — boundaries trimmed, middle byte-exact — is also a
            // `line_trimmed` match, so strategy 2 always claims these in-order. The strategy is
            // ported in its slot for order fidelity.
            name: "6 boundary drift (subsumed by line_trimmed)",
            content: "start  \nmid  dle\nend\n",
            old: "start\nmid  dle\nend",
            new: "START\nmid  dle\nEND",
            replace_all: false,
            want_strategy: "line_trimmed",
            want_content: "START\nmid  dle\nEND\n",
            want_count: 1,
        },
        Case {
            // Smart quotes in the file, ASCII in the pattern.
            name: "7 unicode_normalized",
            content: "say \u{201c}hello\u{201d} now\n",
            old: "say \"hello\" now",
            new: "say \"goodbye\" now",
            replace_all: false,
            want_strategy: "unicode_normalized",
            want_content: "say \"goodbye\" now\n",
            want_count: 1,
        },
        Case {
            // First and last lines anchor exactly; the middle differs but stays >= 50% similar
            // (single candidate). Middle drift defeats every earlier strategy.
            name: "8 block_anchor",
            content: "begin block\n    let value = compute_all(input);\nend block\n",
            old: "begin block\n    let value = compute(input);\nend block",
            new: "begin block\n    let value = 42;\nend block",
            replace_all: false,
            want_strategy: "block_anchor",
            want_content: "begin block\n    let value = 42;\nend block\n",
            want_count: 1,
        },
        Case {
            // No anchor lines survive (first line differs), but 1 of 2 lines is >= 0.80 similar
            // — the 50%-of-lines last resort.
            name: "9 context_aware",
            content: "let total_count = 11;\nlet unrelated = 5;\n",
            old: "let total_count = 10;\nsomething_else();",
            new: "let total = 0;\nreset();",
            replace_all: false,
            want_strategy: "context_aware",
            want_content: "let total = 0;\nreset();\n",
            want_count: 1,
        },
        Case {
            name: "replace_all replaces every occurrence",
            content: "a b a b a\n",
            old: "a",
            new: "x",
            replace_all: true,
            want_strategy: "exact",
            want_content: "x b x b x\n",
            want_count: 3,
        },
    ];

    for case in &cases {
        let got = fuzzy_find_and_replace(case.content, case.old, case.new, case.replace_all)
            .unwrap_or_else(|e| panic!("case {:?} failed to match: {e}", case.name));
        assert_eq!(
            got.strategy, case.want_strategy,
            "case {:?} matched via the wrong strategy",
            case.name
        );
        assert_eq!(
            got.content, case.want_content,
            "case {:?} produced wrong content",
            case.name
        );
        assert_eq!(
            got.count, case.want_count,
            "case {:?} wrong count",
            case.name
        );
    }
}

#[test]
fn block_anchor_thresholds_reject_dissimilar_middles() {
    // Anchors match but the middle is completely unrelated: below the 0.50 unique-candidate
    // threshold, the block-anchor strategy must NOT fire — and no later strategy rescues it
    // (context-aware sees only 2 of 3 lines similar... 2/3 >= 50%, so guard with a middle that
    // also defeats per-line similarity on the anchors' trims).
    let content = "anchor top\nZZZZZZZZZZZZZZZZZZZZZZZZ\nanchor bottom\n";
    let old = "anchor top\nlet expected = compute(seed) + offset;\nanchor bottom";
    // 2 of 3 lines (the anchors) are >= 0.80 similar, so context_aware still matches — the chain
    // is deliberately permissive at its tail. What must hold: block_anchor itself rejected the
    // 0.0-similarity middle and the result is attributed to the later strategy.
    let got = fuzzy_find_and_replace(content, old, "replacement\n", false).unwrap();
    assert_eq!(got.strategy, "context_aware");
}

#[test]
fn multi_match_requires_replace_all() {
    let content = "dup\nx\ndup\n";
    let err = fuzzy_find_and_replace(content, "dup", "one", false).unwrap_err();
    assert_eq!(err, FuzzyError::Ambiguous(2));
    let msg = err.to_string();
    assert!(msg.contains("Found 2 matches"), "{msg}");
    assert!(msg.contains("replace_all"), "{msg}");
}

#[test]
fn identical_and_empty_inputs_are_rejected() {
    assert_eq!(
        fuzzy_find_and_replace("x", "", "y", false).unwrap_err(),
        FuzzyError::EmptyOld
    );
    assert_eq!(
        fuzzy_find_and_replace("x", "same", "same", false).unwrap_err(),
        FuzzyError::Identical
    );
}

#[test]
fn escape_drift_is_blocked_on_non_exact_matches() {
    // The file has a plain apostrophe; old/new both carry a backslash-escaped one (transport
    // drift). The fuzzy chain matches via normalization, then the guard blocks the write.
    let content = "  msg = 'don't panic'\n";
    let old = "msg = 'don\\'t panic'";
    let new = "msg = 'don\\'t worry'";
    let err = fuzzy_find_and_replace(content, old, new, false).unwrap_err();
    match err {
        FuzzyError::EscapeDrift(msg) => {
            assert!(msg.contains("Escape-drift detected"), "{msg}");
        }
        other => panic!("expected escape drift, got {other:?}"),
    }
}

#[test]
fn tab_unescape_is_conditional_on_the_matched_region() {
    // The matched region holds a real tab: a literal `\t` in new_string is unescaped.
    let content = "a\tb\n";
    let got = fuzzy_find_and_replace(content, "a\tb", "a\\tc", false).unwrap();
    assert_eq!(got.content, "a\tc\n");

    // The matched region holds the two-character sequence `\t` (no real tab): new_string's
    // literal `\t` is preserved (e.g. patching source that defines sep = "\t").
    let content = "sep = \"\\t\"\n";
    let got = fuzzy_find_and_replace(content, "sep = \"\\t\"", "sep2 = \"\\t\"", false).unwrap();
    assert_eq!(got.content, "sep2 = \"\\t\"\n");
}

#[test]
fn reindent_applies_on_non_exact_matches_only() {
    // 2-space model indent vs 4-space file: the chain matches (via line_trimmed, which subsumes
    // pure indent drift), and new_string is re-anchored onto the file's 4-space base.
    let content = "    if ready:\n        fire()\n";
    let old = "  if ready:\n      fire()";
    let new = "  if ready:\n      fire()\n      log()";
    let got = fuzzy_find_and_replace(content, old, new, false).unwrap();
    assert_eq!(got.strategy, "line_trimmed");
    assert_eq!(
        got.content,
        "    if ready:\n        fire()\n        log()\n"
    );
}

#[test]
fn did_you_mean_fires_only_for_not_found() {
    let content = "fn compute_totals() {\n    let x = 1;\n}\n";
    // Close-but-absent anchor -> hint with numbered context.
    let hint = find_closest_lines("fn compute_total() {", content).expect("a close line exists");
    assert!(hint.contains("   1| fn compute_totals() {"), "{hint}");

    let not_found = FuzzyError::NotFound;
    let suffix = format_no_match_hint(&not_found, "fn compute_total() {", content);
    assert!(
        suffix.starts_with("\n\nDid you mean one of these sections?"),
        "{suffix}"
    );

    // Non-not-found errors never get the hint.
    let ambiguous = FuzzyError::Ambiguous(2);
    assert!(format_no_match_hint(&ambiguous, "fn compute_total() {", content).is_empty());
}
