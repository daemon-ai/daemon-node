// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Cyrillic / non-Latin recall end-to-end through the public `remember`/`recall` surface: the
//! tokenizer, FTS5 mirror, and recall gate must preserve non-Latin scripts so a Cyrillic query
//! recalls the Cyrillic memory it lexically overlaps.
//!
//! The inflection-only routing (FTS5 miss -> `_cyrillic_like_search` fallback) is exercised at the
//! DB-search layer in `engine::recall`'s unit tests, since — matching the Python reference — the
//! `recall()` relevance gate (`min_relevance = 0.15` for a 1-token query, lexical trigram overlap
//! not being a lexical-token match) drops inflection-only hits unless a dense vector carries them.

use daemon_mnemosyne::{Engine, MnemosyneConfig, RememberArgs};

// PARITY: Mnemosyne tests/test_cyrillic_fts.py (E2E adaptation: exact-token Cyrillic recall through
// the public surface, verifying tokenization/FTS do not mangle non-Latin scripts)
#[test]
fn cyrillic_query_recalls_matching_memory_end_to_end() {
    let e = Engine::open_in_memory(MnemosyneConfig::default()).unwrap();
    e.remember(
        "Пользователь предпочитает тёмную тему оформления",
        &RememberArgs::default(),
    )
    .unwrap();
    e.remember(
        "Совершенно посторонний текст про погоду",
        &RememberArgs::default(),
    )
    .unwrap();

    // Query shares the exact Cyrillic tokens "тёмную"/"тему" with the first memory.
    let hits = e.recall("тёмную тему", 5).unwrap();
    assert!(
        hits.iter().any(|h| h.content.contains("тёмную тему")),
        "the matching Cyrillic memory must surface, got: {:?}",
        hits.iter().map(|h| h.content.as_str()).collect::<Vec<_>>()
    );
    assert!(
        !hits.iter().any(|h| h.content.contains("погоду")),
        "the non-overlapping Cyrillic memory must not surface"
    );
}
