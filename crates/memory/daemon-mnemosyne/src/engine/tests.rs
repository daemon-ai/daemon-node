// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Engine unit tests, moved verbatim out of `engine.rs` (W-MNEMO).

use super::*;
use crate::config::{RecallMode, RecallScope};
use crate::knowledge::{annotations, episodic_graph};
use rusqlite::params;

fn engine() -> Engine {
    Engine::open_in_memory(MnemosyneConfig::default()).expect("engine")
}

#[test]
fn remember_then_recall() {
    let e = engine();
    e.remember(
        "the authentication flow uses JWT tokens",
        &RememberArgs::default(),
    )
    .unwrap();
    e.remember("lunch was pizza", &RememberArgs::default())
        .unwrap();
    let hits = e.recall("authentication flow", 5).unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].content.contains("authentication"));
}

#[test]
fn session_scoping_over_shared_bank() {
    // Two engines over the *same* agent-wide bank, each bound to its own session id (the
    // per-session construction the composition layer's `MnemosyneBanks` performs). Session-scoped
    // rows must not leak across sessions, while `scope='global'` rows are visible to both.
    let dir = std::env::temp_dir().join(format!("mnemosyne-scope-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = |sid: &str| MnemosyneConfig {
        data_dir: dir.clone(),
        session_id: sid.to_string(),
        ..MnemosyneConfig::default()
    };
    let s1 = Engine::open(cfg("s1")).expect("open s1");
    let s2 = Engine::open(cfg("s2")).expect("open s2");

    s1.remember("alpha private to one", &RememberArgs::default())
        .unwrap();
    s2.remember("beta private to two", &RememberArgs::default())
        .unwrap();
    s1.remember(
        "gamma shared globally",
        &RememberArgs {
            scope: "global".to_string(),
            ..Default::default()
        },
    )
    .unwrap();

    // Each session sees its own session-scoped row...
    assert!(!s1.recall("alpha", 5).unwrap().is_empty());
    assert!(!s2.recall("beta", 5).unwrap().is_empty());
    // ...but not the other session's.
    assert!(
        s1.recall("beta", 5).unwrap().is_empty(),
        "s1 must not see s2's row"
    );
    assert!(
        s2.recall("alpha", 5).unwrap().is_empty(),
        "s2 must not see s1's row"
    );
    // The global row is visible to both.
    assert!(!s1.recall("gamma", 5).unwrap().is_empty());
    assert!(
        !s2.recall("gamma", 5).unwrap().is_empty(),
        "global row visible across sessions"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "vec-ext")]
#[test]
fn native_vec_cosine_matches_f32_fallback() {
    let e = engine();
    for (txt, v) in [
        ("alpha vector one", vec![1.0f32, 0.0, 0.0]),
        ("beta vector two", vec![0.0f32, 1.0, 0.0]),
        ("gamma vector three", vec![0.5f32, 0.5, 0.7]),
    ] {
        let id = e.remember(txt, &RememberArgs::default()).unwrap();
        let conn = e.store.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json) VALUES (?1, ?2)",
            params![id, serde_json::to_string(&v).unwrap()],
        )
        .unwrap();
    }
    let query = vec![0.9f32, 0.1, 0.2];
    let conn = e.store.conn.lock().unwrap();
    let native = super::native_cosine_sim_map(&conn, &query).unwrap();
    let stored = super::load_embeddings(&conn).unwrap();
    let manual: std::collections::HashMap<String, f64> = stored
        .iter()
        .map(|(id, v)| (id.clone(), daemon_core::cosine(&query, v) as f64))
        .collect();
    assert_eq!(native.len(), manual.len());
    for (id, m) in &manual {
        let n = native.get(id).copied().unwrap();
        assert!((n - m).abs() < 1e-5, "id={id} native={n} manual={m}");
    }
}

#[test]
fn mutations_write_audit_log_rows() {
    let e = engine();
    let id = e
        .remember("audit me please now", &RememberArgs::default())
        .unwrap();
    e.update(&id, Some("audit me later instead"), None).unwrap();
    e.invalidate(&id, None).unwrap();

    let conn = e.store.conn.lock().unwrap();
    let actions: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT action FROM audit_log ORDER BY event_id ASC")
            .unwrap();
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .flatten()
            .collect();
        rows
    };
    assert!(actions.contains(&"remember".to_string()), "{actions:?}");
    assert!(actions.contains(&"update".to_string()), "{actions:?}");
    assert!(actions.contains(&"invalidate".to_string()), "{actions:?}");
    // The audit rows carry the bank + session for filtering.
    let (bank, session): (String, String) = conn
        .query_row(
            "SELECT bank, session_id FROM audit_log WHERE action='remember' LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(bank, e.config.bank);
    assert_eq!(session, e.config.session_id);
}

#[test]
fn sleep_detects_embedding_cosine_conflict_and_invalidates_older() {
    let e = engine();
    // Two near-identical-but-different memories from the same source, >1h apart, with similar
    // (high-cosine) embeddings and >=2 shared significant tokens, but not near-duplicate text.
    let older_ts = (chrono::Utc::now() - chrono::Duration::hours(48)).to_rfc3339();
    let newer_ts = (chrono::Utc::now() - chrono::Duration::hours(40)).to_rfc3339();
    let va = [1.0f32, 0.02, 0.0];
    let vb = [0.999f32, 0.04, 0.0];
    let conn = e.store.conn.lock().unwrap();
    for (id, ts, content, vec) in [
        (
            "old1",
            &older_ts,
            "Production database runs PostgreSQL version 13 on the primary cluster node",
            va,
        ),
        (
            "new1",
            &newer_ts,
            "Production database migrated to PostgreSQL version 16 across every cluster replica",
            vb,
        ),
    ] {
        conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, metadata_json, veracity, memory_type, scope) \
                 VALUES (?1, ?2, 'conversation', ?3, ?4, 0.5, '{}', 'stated', 'fact', 'session')",
                params![id, content, ts, e.config.session_id],
            )
            .unwrap();
        let emb = serde_json::to_string(&vec).unwrap();
        conn.execute(
                "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?1, ?2, 'mock')",
                params![id, emb],
            )
            .unwrap();
    }
    drop(conn);

    let group = SleepGroup {
        source: "conversation".to_string(),
        ids: vec!["old1".to_string(), "new1".to_string()],
        contents: vec![
            "Production database runs PostgreSQL version 13 on the primary cluster node"
                .to_string(),
            "Production database migrated to PostgreSQL version 16 across every cluster replica"
                .to_string(),
        ],
        scope: "session".to_string(),
        veracity: "stated".to_string(),
        valid_until: None,
    };
    let conflicts = e
        .heuristic_sleep_conflicts(std::slice::from_ref(&group))
        .unwrap();
    assert_eq!(
        conflicts.len(),
        1,
        "expected one conflict, got {conflicts:?}"
    );
    assert_eq!(conflicts[0].older_id, "old1");
    assert_eq!(conflicts[0].newer_id, "new1");

    let resolved = e.resolve_sleep_conflicts(&[group]).unwrap();
    assert_eq!(resolved, 1);
    // The older row is now superseded by the newer.
    let conn = e.store.conn.lock().unwrap();
    let superseded: Option<String> = conn
        .query_row(
            "SELECT superseded_by FROM working_memory WHERE id = 'old1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(superseded.as_deref(), Some("new1"));
}

#[test]
fn sleep_does_not_flag_near_duplicate_or_close_in_time() {
    let e = engine();
    // Near-duplicate content (edit ratio <= 0.3) must NOT be flagged even with high cosine.
    let older_ts = (chrono::Utc::now() - chrono::Duration::hours(48)).to_rfc3339();
    let newer_ts = (chrono::Utc::now() - chrono::Duration::hours(40)).to_rfc3339();
    let v = [1.0f32, 0.0, 0.0];
    let conn = e.store.conn.lock().unwrap();
    for (id, ts, content) in [
        (
            "d1",
            &older_ts,
            "The deployment pipeline uses GitHub Actions for builds",
        ),
        (
            "d2",
            &newer_ts,
            "The deployment pipeline uses GitHub Actions for build",
        ),
    ] {
        conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, metadata_json, veracity, memory_type, scope) \
                 VALUES (?1, ?2, 'conversation', ?3, ?4, 0.5, '{}', 'stated', 'fact', 'session')",
                params![id, content, ts, e.config.session_id],
            )
            .unwrap();
        let emb = serde_json::to_string(&v).unwrap();
        conn.execute(
                "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?1, ?2, 'mock')",
                params![id, emb],
            )
            .unwrap();
    }
    drop(conn);
    let group = SleepGroup {
        source: "conversation".to_string(),
        ids: vec!["d1".to_string(), "d2".to_string()],
        contents: vec![
            "The deployment pipeline uses GitHub Actions for builds".to_string(),
            "The deployment pipeline uses GitHub Actions for build".to_string(),
        ],
        scope: "session".to_string(),
        veracity: "stated".to_string(),
        valid_until: None,
    };
    assert!(
        e.heuristic_sleep_conflicts(&[group]).unwrap().is_empty(),
        "near-duplicate text must not be flagged as a conflict"
    );
}

#[test]
fn memoria_supplement_surfaces_structured_fact_in_recall() {
    // A stored metric fact should be folded into recall as a `memoria` candidate when the query
    // is a structured question with enough lexical overlap (`beam.py` L6006-L6059).
    let e = engine();
    e.remember(
        "The dashboard API response time of 250ms was measured during load testing.",
        &RememberArgs::default(),
    )
    .unwrap();

    let hits = e
        .recall("What is the API response time in production?", 5)
        .unwrap();
    assert!(
        hits.iter().any(|r| r.id.starts_with("memoria_")
            && r.content.contains("[MEMORIA")
            && r.content.contains("250ms")),
        "expected a MEMORIA supplement row, got {hits:?}"
    );
    // The score cap is 0.6 for the memoria row.
    let memoria_row = hits
        .iter()
        .find(|r| r.id.starts_with("memoria_memoria"))
        .expect("memoria row present");
    assert!(
        memoria_row.score <= 0.6 + 1e-9,
        "score {}",
        memoria_row.score
    );
}

#[test]
fn author_and_channel_scope_widen_recall_across_sessions() {
    // Two sessions over a shared bank, each stamping a different author. The default
    // session-scoped recall must not cross sessions, but an author-scoped recall (no channel)
    // widens to all sessions for that author, and a channel-scoped recall sees its channel.
    let dir = std::env::temp_dir().join(format!("mnemosyne-idscope-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = |sid: &str, author: &str, channel: Option<&str>| MnemosyneConfig {
        data_dir: dir.clone(),
        session_id: sid.to_string(),
        author_id: Some(author.to_string()),
        author_type: Some("agent".to_string()),
        channel_id: channel.map(|c| c.to_string()),
        ..MnemosyneConfig::default()
    };
    // Both rows authored by "abdias" but written from different sessions; s2 also on a channel.
    let s1 = Engine::open(cfg("s1", "abdias", None)).expect("open s1");
    let s2 = Engine::open(cfg("s2", "abdias", Some("team-x"))).expect("open s2");
    s1.remember("alpha kubernetes deploy note", &RememberArgs::default())
        .unwrap();
    s2.remember("beta kubernetes rollout note", &RememberArgs::default())
        .unwrap();

    // Default (empty) scope stays session-local: s1 cannot see s2's row.
    let empty = RecallScope::default();
    assert!(
        s1.recall_with_scope(&RecallReq {
            query: "kubernetes",
            top_k: 5,
            query_vector: None,
            scope: &empty,
        })
        .unwrap()
        .iter()
        .all(|r| r.content.contains("alpha")),
        "default scope must remain session-local"
    );

    // Author-only scope widens to every session for that author (the `(1=1)` branch).
    let author_scope = RecallScope {
        author_id: Some("abdias".to_string()),
        ..RecallScope::default()
    };
    let hits = s1
        .recall_with_scope(&RecallReq {
            query: "kubernetes",
            top_k: 5,
            query_vector: None,
            scope: &author_scope,
        })
        .unwrap();
    assert!(
        hits.iter().any(|r| r.content.contains("alpha"))
            && hits.iter().any(|r| r.content.contains("beta")),
        "author scope should surface rows from both sessions, got {hits:?}"
    );

    // A different author sees nothing.
    let other_author = RecallScope {
        author_id: Some("someone-else".to_string()),
        ..RecallScope::default()
    };
    assert!(
        s1.recall_with_scope(&RecallReq {
            query: "kubernetes",
            top_k: 5,
            query_vector: None,
            scope: &other_author,
        })
        .unwrap()
        .is_empty(),
        "unknown author must match no rows"
    );

    // Channel scope surfaces the channel's row.
    let channel_scope = RecallScope {
        channel_id: Some("team-x".to_string()),
        ..RecallScope::default()
    };
    let hits = s1
        .recall_with_scope(&RecallReq {
            query: "kubernetes",
            top_k: 5,
            query_vector: None,
            scope: &channel_scope,
        })
        .unwrap();
    assert!(
        hits.iter().any(|r| r.content.contains("beta")),
        "channel scope should surface the channel row, got {hits:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn vector_recall_surfaces_semantic_match_lexical_misses() {
    let e = engine();
    // A query vector, one near-parallel memory vector (cos ~0.96) and one orthogonal — with
    // content that shares NO tokens with the query, so lexical recall finds nothing.
    let q = [1.0f32, 0.0, 0.0];
    let near = [0.96f32, 0.28, 0.0];
    let far = [0.0f32, 0.0, 1.0];
    e.remember_with_vector("alpha apple", &RememberArgs::default(), Some(&near), "mock")
        .unwrap();
    e.remember_with_vector("beta banana", &RememberArgs::default(), Some(&far), "mock")
        .unwrap();

    // Lexical-only recall for a disjoint query returns nothing.
    assert!(e.recall("zzz", 5).unwrap().is_empty());

    // Vector recall surfaces the semantically-close memory and ranks it first.
    let hits = e.recall_with_vector("zzz", 5, Some(&q)).unwrap();
    assert!(!hits.is_empty(), "vector recall should surface a match");
    assert_eq!(hits[0].content, "alpha apple");
    assert!(
        hits.iter().all(|h| h.content != "beta banana"),
        "orthogonal memory must not pass the vector gate"
    );
}

#[test]
fn get_context_orders_by_importance() {
    let e = engine();
    e.remember(
        "low",
        &RememberArgs {
            importance: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    e.remember(
        "high",
        &RememberArgs {
            importance: 0.9,
            ..Default::default()
        },
    )
    .unwrap();
    let ctx = e.get_context(10).unwrap();
    assert_eq!(ctx[0].content, "high");
}

#[test]
fn lexical_relevance_scores() {
    let q = vec!["auth".to_string(), "flow".to_string()];
    // Both tokens present as whole words + full-query substring -> clamped to 1.0.
    assert!((lexical_relevance(&q, "the auth flow uses jwt") - 1.0).abs() < 1e-9);
    // One exact token of two -> 0.5.
    assert!((lexical_relevance(&q, "the auth subsystem") - 0.5).abs() < 1e-9);
    // A >=4-char substring (no whole-word match, and the full query is not a substring)
    // contributes the 0.4 partial: one of two tokens at 0.4 -> 0.2.
    let q2 = vec!["serialize".to_string(), "absent".to_string()];
    assert!((lexical_relevance(&q2, "the deserializer ran") - 0.2).abs() < 1e-9);
    // Disjoint query -> 0.0; empty query -> 0.0.
    assert_eq!(lexical_relevance(&q, "completely unrelated"), 0.0);
    assert_eq!(lexical_relevance(&[], "anything"), 0.0);
}

#[test]
fn fts_surfaces_row_beyond_recency_window() {
    // Fill the recency/importance window (limit 2000) with high-importance filler that does NOT
    // contain the marker, then add one low-importance row that does. The marker row ranks 2001st
    // by importance, so it is *outside* the fallback scan — only the FTS5 candidate path can
    // surface it. (Under the old full-scan recall this row was unreachable.)
    let e = engine();
    for i in 0..2000 {
        e.remember(
            &format!("filler row number {i}"),
            &RememberArgs {
                importance: 0.9,
                ..Default::default()
            },
        )
        .unwrap();
    }
    e.remember(
        "a unique zqxj marker lives here",
        &RememberArgs {
            importance: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    let hits = e.recall("zqxj", 5).unwrap();
    assert!(
        hits.iter().any(|h| h.content.contains("zqxj")),
        "FTS5 must surface a row outside the recency window"
    );
}

#[test]
fn consolidation_populates_episodic_and_is_idempotent() {
    let e = engine();
    e.remember(
        "blue-green deployment rollout strategy",
        &RememberArgs::default(),
    )
    .unwrap();
    e.remember("margherita pizza for lunch", &RememberArgs::default())
        .unwrap();

    assert_eq!(e.consolidate().unwrap(), 2, "both WM rows promoted");
    assert_eq!(
        e.consolidate().unwrap(),
        0,
        "already-consolidated rows are skipped"
    );

    let conn = e.store.conn.lock().unwrap();
    let n: i64 = conn
        .query_row("SELECT count(*) FROM episodic_memory", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 2);
    let logged: i64 = conn
        .query_row(
            "SELECT count(*) FROM consolidation_log WHERE items_consolidated = 2",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(logged, 1);
}

#[test]
fn episodic_recall_after_consolidation_dedups_cross_tier() {
    let e = engine();
    e.remember(
        "the deployment uses a blue-green rollout",
        &RememberArgs::default(),
    )
    .unwrap();
    e.consolidate().unwrap();

    // The content now lives in BOTH tiers; recall must surface it exactly once (cross-tier dedup).
    let hits = e.recall("deployment rollout", 5).unwrap();
    let matches: Vec<_> = hits
        .iter()
        .filter(|h| h.content.contains("blue-green"))
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "cross-tier duplicate collapsed to one row"
    );
}

#[test]
fn episodic_vector_recall_uses_binary_and_cosine() {
    // Promote two memories with stored embeddings (consolidate also packs MIB binary vectors),
    // then recall by a query vector parallel to one of them with NO lexical overlap. Only the
    // episodic vector + binary path can surface it.
    let e = engine();
    let near = [0.96f32, 0.28, 0.0];
    let far = [0.0f32, 0.0, 1.0];
    e.remember_with_vector("alpha apple", &RememberArgs::default(), Some(&near), "mock")
        .unwrap();
    e.remember_with_vector("beta banana", &RememberArgs::default(), Some(&far), "mock")
        .unwrap();
    e.consolidate().unwrap();

    let q = [1.0f32, 0.0, 0.0];
    let hits = e.recall_with_vector("zzz", 5, Some(&q)).unwrap();
    assert!(
        hits.iter().any(|h| h.content == "alpha apple"),
        "episodic vector recall should surface the semantically-close memory"
    );
    assert!(
        hits.iter().all(|h| h.content != "beta banana"),
        "the orthogonal memory must not pass the vector gate"
    );
}

#[test]
fn remember_extracts_entities_and_facts() {
    let e = engine();
    let id = e
        .remember(
            "Maya works at Acme and uses Postgres",
            &RememberArgs::default(),
        )
        .unwrap();
    let c = e.store.conn.lock().unwrap();

    // Entities became `mentions` annotations.
    let mentions = annotations::query_by_memory(&c, &id, Some("mentions")).unwrap();
    assert!(
        mentions.iter().any(|m| m.value == "Maya"),
        "expected a Maya mention, got {mentions:?}"
    );

    // SPO triples landed in `facts` and were consolidated.
    let fact_rows: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM facts WHERE source_msg_id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap();
    assert!(fact_rows >= 1, "expected at least one extracted fact");
    let works_at: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM consolidated_facts \
                 WHERE subject = 'Maya' AND predicate = 'works_at' AND object = 'Acme'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(works_at, 1, "Maya works_at Acme should be consolidated");
}

#[test]
fn entity_and_fact_match_reorders_recall() {
    let e = engine();
    // The entity-/fact-bearing memory (capitalized "Acme" -> entity + `works_at` fact)...
    e.remember(
        "Maya works at Acme on infrastructure",
        &RememberArgs::default(),
    )
    .unwrap();
    // ...and a lexical-only distractor that mentions "acme" lowercase (no entity extracted).
    e.remember("the acme deadline is approaching", &RememberArgs::default())
        .unwrap();

    // A capitalized-entity query: both rows match lexically, but the entity/fact multipliers
    // must lift the structured memory to the top.
    let hits = e.recall("Acme", 5).unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits[0].content.contains("Maya"),
        "entity+fact match should rank first, got {:?}",
        hits.iter()
            .map(|h| (&h.content, h.score))
            .collect::<Vec<_>>()
    );
}

#[test]
fn cooccurrence_links_memories_sharing_an_entity() {
    let e = engine();
    let a = e
        .remember("Maya leads the Phoenix team", &RememberArgs::default())
        .unwrap();
    let b = e
        .remember("Maya approved the Phoenix budget", &RememberArgs::default())
        .unwrap();
    let c = e.store.conn.lock().unwrap();
    // The two memories share the "Maya"/"Phoenix" entities -> a `references` edge was drawn.
    assert!(episodic_graph::edge_count(&c, &a).unwrap() >= 1);
    let related = episodic_graph::find_related_memories(&c, &a, 2, "", 0.0).unwrap();
    assert!(
        related.iter().any(|r| r.memory_id == b),
        "graph should relate the two Maya/Phoenix memories"
    );
}

#[test]
fn ingest_extracted_merges_llm_entities_and_triples() {
    let e = engine();
    let id = e
        .remember("a routine note", &RememberArgs::default())
        .unwrap();
    let extracted = crate::extract::Extracted {
        entities: vec!["Denis".into()],
        triples: vec![crate::extract::ExtractedTriple {
            subject: "Denis".into(),
            predicate: "manages".into(),
            object: "Atlas".into(),
            confidence: 0.9,
        }],
        facts: vec!["Denis manages the Atlas project".into()],
    };
    e.ingest_extracted(&id, &extracted).unwrap();
    let c = e.store.conn.lock().unwrap();
    let mentions: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1 AND kind = 'mentions' AND value = 'Denis'",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
    assert_eq!(mentions, 1, "LLM entity should land as a mention");
    let triple: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM consolidated_facts WHERE subject='Denis' AND predicate='manages' AND object='Atlas'",
                [],
                |r| r.get(0),
            )
            .unwrap();
    assert_eq!(triple, 1, "LLM triple should be consolidated");
    let fact_ann: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1 AND kind = 'fact'",
            params![id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        fact_ann, 1,
        "LLM statement should land as a fact annotation"
    );
}

#[test]
fn temporal_columns_populated_on_write() {
    let e = engine();
    let id = e
        .remember("ship the release on 2026-05-20", &RememberArgs::default())
        .unwrap();
    let c = e.store.conn.lock().unwrap();
    let (date, precision): (Option<String>, String) = c
        .query_row(
            "SELECT event_date, event_date_precision FROM working_memory WHERE id = ?1",
            params![id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(date.as_deref(), Some("2026-05-20"));
    assert_eq!(precision, "day");
}

#[test]
fn sleep_groups_and_summarizes_with_aaak() {
    let e = engine();
    // Two rows from the same source -> one summary group.
    e.remember("User prefers dark mode", &RememberArgs::default())
        .unwrap();
    e.remember("User prefers tabs over spaces", &RememberArgs::default())
        .unwrap();
    let report = e.sleep(true).expect("forced sleep");
    assert_eq!(report.items_consolidated, 2);
    assert_eq!(report.summaries_created, 1);
    assert_eq!(report.llm_used, 0, "no LLM -> AAAK fallback");
    // A summary episodic row was written, tagged as a sleep consolidation.
    let c = e.store.conn.lock().unwrap();
    let summaries: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM episodic_memory WHERE source = 'sleep_consolidation'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(summaries, 1);
    // The originals are marked consolidated (additive: still present).
    let pending: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM working_memory WHERE consolidated_at IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pending, 0, "all working rows claimed");
}

#[test]
fn sleep_skips_pinned_and_respects_cutoff() {
    let e = engine();
    let id = e
        .remember("recent unpinned note", &RememberArgs::default())
        .unwrap();
    {
        let c = e.store.conn.lock().unwrap();
        c.execute(
            "UPDATE working_memory SET pinned = 1 WHERE id = ?1",
            params![id],
        )
        .unwrap();
    }
    // force=false: the row is fresh (after the cutoff) AND pinned -> nothing consolidates.
    let report = e.sleep(false).expect("sleep");
    assert_eq!(report.items_consolidated, 0);
}

#[test]
fn degrade_episodic_promotes_old_tiers() {
    let e = engine();
    // Seed an episodic row backdated > TIER2_DAYS so tier1->2 fires.
    {
        let c = e.store.conn.lock().unwrap();
        c.execute(
            "INSERT INTO episodic_memory (id, content, session_id, tier, created_at) \
                 VALUES ('old1', 'User prefers Python and Rust over Go', 'default', 1, \
                         datetime('now', '-60 days'))",
            [],
        )
        .unwrap();
        // And one backdated > TIER3_DAYS at tier 2 so tier2->3 fires.
        let long = "x ".repeat(400);
        c.execute(
            "INSERT INTO episodic_memory (id, content, session_id, tier, created_at) \
                 VALUES ('old2', ?1, 'default', 2, datetime('now', '-200 days'))",
            params![long],
        )
        .unwrap();
    }
    let (t1, t2) = e.degrade_episodic().expect("degrade");
    assert_eq!(t1, 1, "tier1 row should promote to tier2");
    assert_eq!(t2, 1, "tier2 row should promote to tier3");
    let c = e.store.conn.lock().unwrap();
    let tier1: i64 = c
        .query_row(
            "SELECT tier FROM episodic_memory WHERE id='old1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(tier1, 2);
    let (tier2, len): (i64, i64) = c
        .query_row(
            "SELECT tier, LENGTH(content) FROM episodic_memory WHERE id='old2'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(tier2, 3);
    assert!(
        len as usize <= TIER3_MAX_CHARS + 8,
        "tier3 content compressed"
    );
}

#[test]
fn tool_backing_methods_round_trip() {
    let e = engine();
    let id = e
        .remember("a fact to manage", &RememberArgs::default())
        .unwrap();
    assert!(e.get(&id).unwrap().is_some());
    assert!(e.update(&id, Some("an updated fact"), Some(0.9)).unwrap());
    assert_eq!(e.get(&id).unwrap().unwrap().content, "an updated fact");

    // Scratchpad CRUD.
    e.scratchpad_write("remember to ship").unwrap();
    assert_eq!(e.scratchpad_read().unwrap().len(), 1);
    assert_eq!(e.scratchpad_clear().unwrap(), 1);
    assert!(e.scratchpad_read().unwrap().is_empty());

    // Triples + canonical.
    e.triple_add(&TripleAdd {
        subject: "Ada",
        predicate: "uses",
        object: "Rust",
        valid_from: None,
        valid_until: None,
        source: "tool",
        confidence: 1.0,
        supersede: true,
    })
    .unwrap();
    assert_eq!(
        e.triple_query(&TripleQuery {
            subject: Some("Ada"),
            predicate: None,
            object: None,
            as_of: None,
        })
        .unwrap()
        .len(),
        1
    );
    let (_row, status) = e
        .canonical_remember(&CanonicalRemember {
            owner_id: "ada",
            category: "identity",
            name: "lang",
            body: "Rust",
            source: "tool",
            confidence: 1.0,
        })
        .unwrap();
    assert_eq!(status, crate::knowledge::canonical::Status::Created);
    assert_eq!(e.canonical_recall("ada", None, None).unwrap().len(), 1);

    // Invalidate drops it from recall surface.
    assert!(e.invalidate(&id, None).unwrap());
    assert!(e.get(&id).unwrap().is_none());

    // Forget hard-deletes.
    let id2 = e.remember("ephemeral", &RememberArgs::default()).unwrap();
    assert!(e.forget(&id2).unwrap());
    assert!(!e.forget(&id2).unwrap(), "already gone");
}

#[test]
fn export_import_round_trips_rows() {
    let e = engine();
    e.remember("portable memory one", &RememberArgs::default())
        .unwrap();
    e.remember("portable memory two", &RememberArgs::default())
        .unwrap();
    let bundle = e.export().unwrap();

    let e2 = engine();
    let n = e2.import(&bundle).unwrap();
    assert_eq!(n, 2, "both working rows imported");
    assert!(!e2.recall("portable memory", 5).unwrap().is_empty());
}

#[test]
fn stats_and_diagnose_report_counts() {
    let e = engine();
    e.remember("count me", &RememberArgs::default()).unwrap();
    let stats = e.stats().unwrap();
    assert_eq!(stats.working, 1);
    let diag = e.diagnose().unwrap();
    assert_eq!(diag.pending_consolidation, 1);
}

#[test]
fn enhanced_recall_uses_synonym_expansion() {
    // Enhanced recall expands "db" -> the `database` synonym group, so a query that shares no
    // surface token with the stored row still surfaces it (base recall alone would miss "db").
    let cfg = MnemosyneConfig {
        recall_mode: RecallMode::Enhanced,
        ..MnemosyneConfig::default()
    };
    let e = Engine::open_in_memory(cfg).unwrap();
    e.remember(
        "the database password rotation is monthly",
        &RememberArgs::default(),
    )
    .unwrap();
    e.remember("lunch was margherita pizza", &RememberArgs::default())
        .unwrap();

    let hits = e.recall("db password", 5).unwrap();
    assert!(
        !hits.is_empty(),
        "enhanced recall should surface via synonym expansion"
    );
    assert!(
        hits[0].content.contains("password"),
        "got: {}",
        hits[0].content
    );
    // A second identical query is served from the cache and stays consistent.
    let again = e.recall("db password", 5).unwrap();
    assert_eq!(again[0].content, hits[0].content);
}

#[test]
fn base_recall_unchanged_when_flags_off() {
    // The default (Base) mode must not synonym-expand: "db" shares no token with the row, so a
    // base recall returns nothing (proving enhanced behavior is opt-in, no base regression).
    let e = engine();
    e.remember(
        "the database password rotation is monthly",
        &RememberArgs::default(),
    )
    .unwrap();
    assert!(
        e.recall("db", 5).unwrap().is_empty(),
        "base recall must not expand synonyms"
    );
}

#[test]
fn polyphonic_recall_fuses_voices() {
    let cfg = MnemosyneConfig {
        recall_mode: RecallMode::Polyphonic,
        ..MnemosyneConfig::default()
    };
    let e = Engine::open_in_memory(cfg).unwrap();
    let acme_vec = [1.0f32, 0.0, 0.0];
    e.remember_with_vector(
        "Acme is a company",
        &RememberArgs::default(),
        Some(&acme_vec),
        "mock",
    )
    .unwrap();
    e.remember_with_vector(
        "unrelated note about pizza",
        &RememberArgs::default(),
        Some(&[0.0, 1.0, 0.0]),
        "mock",
    )
    .unwrap();

    // "Acme" hits the graph/fact voices (fact subject "Acme") and the vector voice (parallel
    // query vector); RRF fusion should surface the Acme row.
    let hits = e.recall_with_vector("Acme", 5, Some(&acme_vec)).unwrap();
    assert!(
        hits.iter().any(|h| h.content == "Acme is a company"),
        "polyphonic fused result"
    );
}
