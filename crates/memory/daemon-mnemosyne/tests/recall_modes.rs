// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Conformance: the three recall modes (`Base`, `Enhanced`, `Polyphonic`) over one seeded bank.
//!
//! Mirrors the port-spec §16 intent: enhanced + polyphonic must produce ranked results, and base
//! recall must stay unchanged when the opt-in flags are off (no synonym expansion / no fusion).

use daemon_mnemosyne::{Engine, MnemosyneConfig, RecallMode, RememberArgs};

/// Seed a small, deterministic bank with 3-dim embeddings into an engine in the given mode.
fn seeded(mode: RecallMode) -> Engine {
    let cfg = MnemosyneConfig {
        recall_mode: mode,
        ..MnemosyneConfig::default()
    };
    let e = Engine::open_in_memory(cfg).unwrap();
    let rows: &[(&str, [f32; 3])] = &[
        (
            "the database password rotation policy runs monthly",
            [1.0, 0.0, 0.0],
        ),
        (
            "Acme Corporation signed the contract last week",
            [0.0, 1.0, 0.0],
        ),
        ("I prefer dark mode in the editor", [0.0, 0.0, 1.0]),
        ("lunch was margherita pizza on Friday", [0.7, 0.7, 0.0]),
        ("the server crashed due to a memory leak", [0.5, 0.0, 0.5]),
    ];
    for (content, vec) in rows {
        e.remember_with_vector(content, &RememberArgs::default(), Some(vec), "mock")
            .unwrap();
    }
    e
}

#[test]
fn base_recall_is_ranked_but_does_not_expand_synonyms() {
    let e = seeded(RecallMode::Base);

    // A directly-lexical query returns ranked results.
    let hits = e.recall("database password", 5).unwrap();
    assert!(
        !hits.is_empty(),
        "base recall should surface the lexical match"
    );
    assert!(
        hits[0].content.contains("password"),
        "top hit: {}",
        hits[0].content
    );

    // "db" only matches via the `database` synonym group, which base mode must NOT apply.
    assert!(
        e.recall("db", 5).unwrap().is_empty(),
        "base recall must not expand synonyms (opt-in only)"
    );
}

#[test]
fn enhanced_recall_expands_synonyms_and_is_stable() {
    let e = seeded(RecallMode::Enhanced);

    // Same "db" query now resolves through synonym expansion -> the database row.
    let hits = e.recall("db", 5).unwrap();
    assert!(
        !hits.is_empty(),
        "enhanced recall should expand `db` -> database"
    );
    assert!(
        hits[0].content.contains("password"),
        "top hit: {}",
        hits[0].content
    );

    // Repeating the query (served from the semantic cache) stays consistent.
    let again = e.recall("db", 5).unwrap();
    assert_eq!(
        again[0].content, hits[0].content,
        "cached recall must be stable"
    );
}

#[test]
fn polyphonic_recall_fuses_voices_over_seeded_bank() {
    let e = seeded(RecallMode::Polyphonic);

    // "Acme" + a query vector parallel to the Acme row: the vector voice ranks it first, and the
    // graph voice (entity mention) reinforces it; RRF fusion must surface it.
    let hits = e
        .recall_with_vector("Acme", 5, Some(&[0.0, 1.0, 0.0]))
        .unwrap();
    assert!(
        hits.iter().any(|h| h.content.contains("Acme")),
        "polyphonic fusion should surface the Acme row, got: {:?}",
        hits.iter().map(|h| h.content.as_str()).collect::<Vec<_>>()
    );
}
