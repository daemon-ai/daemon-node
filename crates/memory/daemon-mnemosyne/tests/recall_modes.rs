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
fn base_recall_is_ranked_and_gates_short_tokens() {
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

    // "db" alone yields NO results: `_recall_tokens` drops <3-char tokens, so the query carries
    // zero lexical signal (verified against the Python reference: `recall("db")` -> []).
    assert!(
        e.recall("db", 5).unwrap().is_empty(),
        "two-char queries carry no recall tokens"
    );

    // One exact token of two ("rotation", lexical 0.5) clears the 0.15 short-query gate even
    // though "pwd" itself matches nothing (verified against Python: base `pwd rotation` surfaces).
    let hits = e.recall("pwd rotation", 5).unwrap();
    assert!(
        !hits.is_empty() && hits[0].content.contains("password"),
        "partial token overlap should surface the password row, got {hits:?}"
    );
}

#[test]
fn enhanced_recall_matches_python_gating_and_is_stable() {
    let e = seeded(RecallMode::Enhanced);

    // Parity with the Python reference (`recall_enhanced("db")` -> []): expansion rewrites "db"
    // to "(database|db|datastore|data_store)", whose 3 surviving tokens raise the lexical gate to
    // 0.5 while the seeded row matches only 1/3 of them — enhanced mode must NOT loosen the gate.
    assert!(
        e.recall("db", 5).unwrap().is_empty(),
        "enhanced recall keeps the Python lexical gate"
    );

    // A query with real lexical signal resolves, and repeating it (now served from the semantic
    // query cache) stays consistent.
    let hits = e.recall("db password", 5).unwrap();
    assert!(
        !hits.is_empty() && hits[0].content.contains("password"),
        "top hit: {hits:?}"
    );
    let again = e.recall("db password", 5).unwrap();
    assert_eq!(
        again[0].content, hits[0].content,
        "cached recall must be stable"
    );
}

#[test]
fn base_recall_matches_python_reference_matrix() {
    // Golden matrix captured from the live Python engine (BeamMemory over the same 5-row bank):
    //   recall(q, top_k=5) -> top-1 content, or [] when the lexical gate rejects everything.
    let e = seeded(RecallMode::Base);
    let expect_password = "the database password rotation policy runs monthly";
    let expect_dark = "I prefer dark mode in the editor";
    let cases: &[(&str, Option<&str>)] = &[
        ("db", None),
        ("theme", None),
        ("db password", Some(expect_password)),
        ("db rotation", Some(expect_password)),
        ("datastore rotation", Some(expect_password)),
        ("pwd rotation", Some(expect_password)),
        ("password rotation", Some(expect_password)),
        ("dark theme", Some(expect_dark)),
    ];
    for (query, want) in cases {
        let hits = e.recall(query, 5).unwrap();
        match want {
            None => assert!(hits.is_empty(), "{query:?} should return nothing: {hits:?}"),
            Some(content) => {
                assert_eq!(
                    hits.first().map(|h| h.content.as_str()),
                    Some(*content),
                    "{query:?} top hit diverged from the Python reference"
                );
            }
        }
    }
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
