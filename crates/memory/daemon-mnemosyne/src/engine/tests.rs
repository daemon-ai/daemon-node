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
fn remember_dedups_exact_content_and_refreshes_row() {
    let e = engine();
    let id1 = e
        .remember(
            "the deploy target is us-east-1",
            &RememberArgs {
                importance: 0.9,
                ..Default::default()
            },
        )
        .unwrap();
    // Simulate a consolidated row: dedup must clear the stamp so sleep re-runs.
    e.store
        .conn
        .lock()
        .unwrap()
        .execute(
            "UPDATE working_memory SET consolidated_at = 'x', veracity = 'unknown' WHERE id = ?1",
            params![id1],
        )
        .unwrap();
    let id2 = e
        .remember(
            "the deploy target is us-east-1",
            &RememberArgs {
                importance: 0.2,
                veracity: "stated".to_string(),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        id1, id2,
        "exact-content re-remember returns the existing id"
    );

    let conn = e.store.conn.lock().unwrap();
    let (count, importance, veracity, consolidated_at): (i64, f64, String, Option<String>) = conn
        .query_row(
            "SELECT COUNT(*), MAX(importance), MAX(veracity), MAX(consolidated_at) \
             FROM working_memory WHERE content = 'the deploy target is us-east-1'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert_eq!(count, 1, "no duplicate row");
    assert!((importance - 0.9).abs() < 1e-9, "importance keeps the max");
    assert_eq!(veracity, "stated", "non-unknown veracity upgrades the row");
    assert!(
        consolidated_at.is_none(),
        "dedup clears consolidated_at so the row is sleep-eligible again"
    );
}

#[test]
fn remember_derives_and_clamps_trust_tier() {
    let e = engine();
    for (source, expected) in [
        ("conversation", "STATED"),
        ("mcp", "EXTERNAL_WRITE"),
        ("bulk_import", "IMPORTED"),
        ("sleep_consolidation", "DERIVED"),
        ("somewhere_else", "STATED"),
    ] {
        let id = e
            .remember(
                &format!("trust tier probe via {source}"),
                &RememberArgs {
                    source: source.to_string(),
                    ..Default::default()
                },
            )
            .unwrap();
        let tier: String = e
            .store
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT trust_tier FROM working_memory WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(tier, expected, "source {source}");
    }
    // Explicit-but-bogus tiers clamp to STATED; bogus veracity labels clamp to unknown.
    let id = e
        .remember(
            "explicit tier probe",
            &RememberArgs {
                trust_tier: Some("ROOT".to_string()),
                veracity: "Gospel Truth".to_string(),
                ..Default::default()
            },
        )
        .unwrap();
    let (tier, veracity): (String, String) = e
        .store
        .conn
        .lock()
        .unwrap()
        .query_row(
            "SELECT trust_tier, veracity FROM working_memory WHERE id = ?1",
            params![id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(tier, "STATED");
    assert_eq!(veracity, "unknown");
}

#[test]
fn remember_emits_memory_events_with_stable_device_id() {
    let e = engine();
    let id = e
        .remember("an event-logged memory", &RememberArgs::default())
        .unwrap();
    // Exact-content re-remember logs an UPDATE against the same memory id.
    e.remember("an event-logged memory", &RememberArgs::default())
        .unwrap();
    let conn = e.store.conn.lock().unwrap();
    let rows: Vec<(String, String, String)> = {
        let mut stmt = conn
            .prepare(
                "SELECT operation, memory_id, device_id FROM memory_events ORDER BY timestamp ASC",
            )
            .unwrap();
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .flatten()
            .collect();
        rows
    };
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert_eq!(rows[0].0, "CREATE");
    assert_eq!(rows[1].0, "UPDATE");
    assert!(rows.iter().all(|r| r.1 == id));
    assert!(rows[0].2.starts_with("device-"));
    assert_eq!(rows[0].2, rows[1].2, "device id is stable per bank");
    // The persisted identity matches sync_meta.
    let stored: String = conn
        .query_row(
            "SELECT value FROM sync_meta WHERE key = 'device_id'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored, rows[0].2);
    // Events carry a dedup hash.
    let no_hash: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memory_events WHERE event_hash IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(no_hash, 0);
}

#[test]
fn trim_caps_unconsolidated_working_rows() {
    let e = Engine::open_in_memory(MnemosyneConfig {
        working_memory_max_items: 3,
        ..Default::default()
    })
    .unwrap();
    // A consolidated row is exempt from the cap ("originals stay").
    let kept = e
        .remember("consolidated original zero", &RememberArgs::default())
        .unwrap();
    e.store
        .conn
        .lock()
        .unwrap()
        .execute(
            "UPDATE working_memory SET consolidated_at = 'x', timestamp = '2000-01-01T00:00:00' \
             WHERE id = ?1",
            params![kept],
        )
        .unwrap();
    for i in 0..5 {
        e.remember(&format!("fresh row number {i}"), &RememberArgs::default())
            .unwrap();
    }
    let conn = e.store.conn.lock().unwrap();
    let unconsolidated: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM working_memory WHERE consolidated_at IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(unconsolidated, 3, "trim caps not-yet-consolidated rows");
    let exempt: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM working_memory WHERE id = ?1",
            params![kept],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(exempt, 1, "consolidated rows survive the trim");
}

#[test]
fn remember_writes_occurred_on_and_has_source_annotations() {
    let e = engine();
    let id = e
        .remember(
            "annotated for provenance",
            &RememberArgs {
                source: "toolbelt".to_string(),
                ..Default::default()
            },
        )
        .unwrap();
    let conn = e.store.conn.lock().unwrap();
    let occurred = annotations::query_by_memory(&conn, &id, Some("occurred_on")).unwrap();
    assert_eq!(occurred.len(), 1);
    assert_eq!(occurred[0].value.len(), 10, "YYYY-MM-DD grain");
    let has_source = annotations::query_by_memory(&conn, &id, Some("has_source")).unwrap();
    assert_eq!(has_source.len(), 1);
    assert_eq!(has_source[0].value, "toolbelt");
    // Conversational sources skip has_source (`beam.py` L3487).
    drop(conn);
    let id2 = e
        .remember("plain conversational note", &RememberArgs::default())
        .unwrap();
    let conn = e.store.conn.lock().unwrap();
    let none = annotations::query_by_memory(&conn, &id2, Some("has_source")).unwrap();
    assert!(none.is_empty());
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
            filters: Default::default(),
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
            filters: Default::default(),
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
            filters: Default::default(),
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
            filters: Default::default(),
        })
        .unwrap();
    assert!(
        hits.iter().any(|r| r.content.contains("beta")),
        "channel scope should surface the channel row, got {hits:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn wm_vector_signal_blends_but_lexical_gates() {
    let e = engine();
    let q = [1.0f32, 0.0, 0.0];
    let near = [0.96f32, 0.28, 0.0];
    let far = [0.0f32, 0.0, 1.0];
    // Two rows with identical lexical relevance for the query; only the vectors differ.
    e.remember_with_vector(
        "apple pie recipe",
        &RememberArgs::default(),
        Some(&near),
        "mock",
    )
    .unwrap();
    e.remember_with_vector(
        "apple tart recipe",
        &RememberArgs::default(),
        Some(&far),
        "mock",
    )
    .unwrap();

    // Working-memory candidates are lexically gated (`beam.py` L5313): a vector-only match with
    // zero lexical relevance is dropped — pure semantic matches surface via the episodic tier.
    assert!(e.recall_with_vector("zzz", 5, Some(&q)).unwrap().is_empty());

    // With lexical parity, the dense blend (`base*0.8 + sim*0.2`, `beam.py` L5321-L5323) breaks
    // the tie toward the semantically-close row.
    let hits = e.recall_with_vector("apple recipe", 5, Some(&q)).unwrap();
    assert_eq!(hits.len(), 2, "both rows pass the lexical gate");
    assert_eq!(hits[0].content, "apple pie recipe");
    assert!(hits[0].dense_score > hits[1].dense_score);
}

#[test]
fn wm_score_matches_python_reference_formula() {
    // Golden check against the live Python engine (BeamMemory.recall on a fresh bank):
    //   remember("the database password rotation policy runs monthly"); recall("database password")
    //   -> score 0.53232, keyword_score 1.0, fts_score 1.0, dense_score 0.0.
    // Composition: base = 1.0*0.48 + 0.5*0.2 + 1.0^2*0.08 = 0.66; score = base * (0.32 + 0.68*decay)
    // * veracity(unknown)=0.8. With decay ~= 1.0 for a fresh row -> ~0.5282 (Python's 0.53232
    // reflects its local-vs-utc timestamp skew inflating decay slightly above 1.0).
    let e = engine();
    e.remember(
        "the database password rotation policy runs monthly",
        &RememberArgs::default(),
    )
    .unwrap();
    let hits = e.recall("database password", 5).unwrap();
    assert_eq!(hits.len(), 1);
    let h = &hits[0];
    assert_eq!(h.keyword_score, 1.0);
    assert_eq!(h.fts_score, 1.0);
    assert_eq!(h.dense_score, 0.0);
    assert!(
        (h.score - 0.5282).abs() < 0.005,
        "score {} should match the Python formula chain",
        h.score
    );
}

#[test]
fn recall_diagnostics_record_paths_and_fallbacks() {
    let e = engine();
    // Miss on an EMPTY bank: both fallbacks fire with zero scanned rows -> truly empty.
    assert!(e.recall("zzz qqq", 5).unwrap().is_empty());
    e.remember("kubernetes deploy pipeline", &RememberArgs::default())
        .unwrap();
    // Hit: the FTS path contributes the kept WM row.
    assert!(!e.recall("kubernetes pipeline", 5).unwrap().is_empty());
    // Miss over a non-empty bank: the fallback scan credits its scanned candidates even though
    // the relevance gate kept none (`beam.py` L5366-L5370), so this is NOT truly empty.
    assert!(e.recall("zzz qqq", 5).unwrap().is_empty());

    let snap = e.recall_diagnostics().snapshot();
    assert_eq!(snap.totals.calls, 3);
    assert_eq!(snap.totals.calls_truly_empty, 1);
    let wm_fts = &snap.by_tier[0];
    assert_eq!(wm_fts.0, "wm_fts");
    assert_eq!(wm_fts.1.calls_with_hits, 1);
    let wm_fallback = &snap.by_tier[2];
    assert_eq!(wm_fallback.0, "wm_fallback");
    assert_eq!(
        wm_fallback.1.total_hits, 1,
        "the non-empty miss scanned one fallback candidate"
    );
    assert_eq!(snap.totals.calls_using_wm_fallback, 2);
    assert!((snap.totals.wm_fallback_rate - 2.0 / 3.0).abs() < 1e-9);

    e.recall_diagnostics().reset();
    assert_eq!(e.recall_diagnostics().snapshot().totals.calls, 0);
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
    use crate::recall::lexical::lexical_relevance;
    let q = vec!["auth".to_string(), "flow".to_string()];
    // Both tokens present as whole words + full-query substring -> clamped to 1.0.
    assert!((lexical_relevance(&q, "the auth flow uses jwt", "auth flow") - 1.0).abs() < 1e-9);
    // One exact token of two -> 0.5.
    assert!((lexical_relevance(&q, "the auth subsystem", "auth flow") - 0.5).abs() < 1e-9);
    // A >=4-char substring (no whole-word match, and the full query is not a substring)
    // contributes the 0.4 partial: one of two tokens at 0.4 -> 0.2.
    let q2 = vec!["serialize".to_string(), "absent".to_string()];
    assert!(
        (lexical_relevance(&q2, "the deserializer ran", "serialize absent") - 0.2).abs() < 1e-9
    );
    // Disjoint query -> 0.0; empty query -> 0.0.
    assert_eq!(
        lexical_relevance(&q, "completely unrelated", "auth flow"),
        0.0
    );
    assert_eq!(lexical_relevance(&[], "anything", ""), 0.0);
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
    // `mentions` annotations are opt-in (`beam.py` `extract_entities=False` default); the SPO
    // fact/graph pipeline is always on.
    let id = e
        .remember(
            "Maya works at Acme and uses Postgres",
            &RememberArgs {
                extract_entities: true,
                ..Default::default()
            },
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
    // The entity-bearing memory: `extract_entities` stores `mentions` annotations (Maya, Acme),
    // which is what the entity-aware recall pass matches on (`beam.py` `_find_memories_by_entity`
    // reads mention annotations — without them the 1.3x boost never fires and the two rows tie).
    e.remember(
        "Maya works at Acme on infrastructure",
        &RememberArgs {
            extract_entities: true,
            ..Default::default()
        },
    )
    .unwrap();
    // ...and a lexical-only distractor that mentions "acme" lowercase (no entity extracted).
    e.remember("the acme deadline is approaching", &RememberArgs::default())
        .unwrap();

    // A capitalized-entity query: both rows match lexically, but the 1.3x entity multiplier must
    // lift the annotated memory to the top.
    let hits = e.recall("Acme", 5).unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits[0].content.contains("Maya"),
        "entity match should rank first, got {:?}",
        hits.iter()
            .map(|h| (&h.content, h.score))
            .collect::<Vec<_>>()
    );
    assert!(
        hits[0].entity_match,
        "the winning row must be marked entity_match"
    );
}

#[test]
fn cooccurrence_links_memories_sharing_an_entity() {
    // Proactive linking is env-gated in Python (`MNEMOSYNE_PROACTIVE_LINKING=1`) and config-gated
    // here; entity-overlap linking additionally requires the `mentions` annotations.
    let e = Engine::open_in_memory(MnemosyneConfig {
        proactive_linking: true,
        ..Default::default()
    })
    .unwrap();
    let args = RememberArgs {
        extract_entities: true,
        ..Default::default()
    };
    let a = e.remember("Maya leads the Phoenix team", &args).unwrap();
    let b = e
        .remember("Maya approved the Phoenix budget", &args)
        .unwrap();
    let c = e.store.conn.lock().unwrap();
    // The second memory shares the "Maya"/"Phoenix" entities -> a `references` edge was drawn
    // from the newer memory to the older one.
    assert!(episodic_graph::edge_count(&c, &b).unwrap() >= 1);
    let related = episodic_graph::find_related_memories(&c, &b, 2, "", 0.0).unwrap();
    assert!(
        related.iter().any(|r| r.memory_id == a),
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
        facts: vec![
            "Denis manages the Atlas project".into(),
            "too short".into(), // <= MIN_FACT_LENGTH chars — dropped by filter_facts
        ],
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
fn finish_sleep_embeds_and_binarizes_llm_summary() {
    let e = engine();
    e.remember("User decided to migrate the service to Rust", &{
        RememberArgs {
            source: "conversation".to_string(),
            ..Default::default()
        }
    })
    .unwrap();
    e.remember("User decided the migration ships next quarter", &{
        RememberArgs {
            source: "conversation".to_string(),
            ..Default::default()
        }
    })
    .unwrap();
    let groups = e.sleep_plan(true).expect("plan");
    assert_eq!(groups.len(), 1);

    // The async seam supplies an LLM summary (with a <think> block to strip) and its embedding.
    let vec: Vec<f32> = vec![0.9, -0.4, 0.2, -0.1];
    let mut summaries = std::collections::HashMap::new();
    summaries.insert(
        groups[0].source.clone(),
        GroupSummary {
            text: "<think>reasoning</think>We decided to migrate the service to Rust next quarter"
                .to_string(),
            llm: true,
            embedding: Some(vec.clone()),
            model: "mock-embed".to_string(),
        },
    );
    let report = e.finish_sleep(&groups, &summaries).expect("finish");
    assert_eq!(report.summaries_created, 1);
    assert_eq!(report.llm_used, 1);

    let c = e.store.conn.lock().unwrap();
    let (id, content, memory_type, binary): (String, String, String, Option<Vec<u8>>) = c
        .query_row(
            "SELECT id, content, memory_type, binary_vector FROM episodic_memory \
             WHERE source = 'sleep_consolidation'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert!(
        !content.contains("<think>"),
        "think block must be stripped: {content}"
    );
    assert_eq!(
        memory_type, "decision",
        "summary must be typed-memory classified"
    );
    // The embedding row and MIB binary vector make the summary visible to vector recall.
    let (emb_json, model): (String, String) = c
        .query_row(
            "SELECT embedding_json, model FROM memory_embeddings WHERE memory_id = ?1",
            params![id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .expect("summary embedding row");
    assert_eq!(serde_json::from_str::<Vec<f32>>(&emb_json).unwrap(), vec);
    assert_eq!(model, "mock-embed");
    assert_eq!(
        binary.as_deref(),
        Some(crate::binary_vectors::maximally_informative_binarization(&vec).as_slice()),
        "episodic row must carry the MIB binary vector"
    );
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

    // Invalidate drops it from the recall surface, but `get` stays a pure read that still returns
    // the row (`beam.py` `get` L3855-L3911 applies no validity filter).
    assert!(e.invalidate(&id, None).unwrap());
    assert!(e.get(&id).unwrap().is_some());
    assert!(e.recall("updated fact", 5).unwrap().is_empty());

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

// parity: test_consolidate_fact_sibling_races.py::test_concurrent_resolve_conflict_different_winners_deterministic (tests/test_consolidate_fact_sibling_races.py:56)
// parity: test_consolidate_fact_sibling_races.py::TestReviewHardening::test_first_writer_wins_logs_warning (tests/test_consolidate_fact_sibling_races.py:436)
#[test]
fn resolve_conflict_first_writer_wins() {
    use crate::knowledge::veracity::consolidate_fact;
    let e = engine();
    {
        let c = e.store.conn.lock().unwrap();
        consolidate_fact(&c, "Alice", "is", "engineer", "stated", "src_a").unwrap();
        consolidate_fact(&c, "Alice", "is", "manager", "inferred", "src_b").unwrap();
    }
    let pending = e.pending_conflicts().unwrap();
    assert_eq!(pending.len(), 1, "setup failure: expected one conflict");
    let conflict = &pending[0];

    // First resolution wins…
    e.resolve_conflict(
        conflict.conflict_id,
        true,
        &conflict.newer_fact_id,
        &conflict.older_fact_id,
    )
    .unwrap();
    // …and a second, opposite resolution of the SAME conflict must be a no-op — the two
    // competing writers otherwise leave BOTH facts superseded (the pre-fix race shape).
    e.resolve_conflict(
        conflict.conflict_id,
        true,
        &conflict.older_fact_id,
        &conflict.newer_fact_id,
    )
    .unwrap();

    let c = e.store.conn.lock().unwrap();
    let superseded: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM consolidated_facts \
             WHERE subject = 'Alice' AND superseded_by IS NOT NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        superseded <= 1,
        "both facts superseded ({superseded}/2) — conflicting resolutions left an incoherent state"
    );
    let resolved_against: Option<String> = c
        .query_row(
            "SELECT superseded_by FROM consolidated_facts WHERE id = ?1",
            params![conflict.older_fact_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        resolved_against.as_deref(),
        Some(conflict.newer_fact_id.as_str()),
        "the FIRST resolution must remain durable"
    );
}

// parity: test_consolidate_fact_id_collision.py::TestReviewHardening::test_resolve_conflict_rejects_ambiguous_winning_id (tests/test_consolidate_fact_id_collision.py:389)
#[test]
fn resolve_conflict_rejects_foreign_fact_ids() {
    use crate::knowledge::veracity::consolidate_fact;
    let e = engine();
    {
        let c = e.store.conn.lock().unwrap();
        consolidate_fact(&c, "Iris", "is", "the lead", "stated", "m1").unwrap();
        consolidate_fact(&c, "Iris", "is", "the manager", "inferred", "m2").unwrap();
        // An unrelated fact that must never be dragged into the resolution.
        consolidate_fact(&c, "Zoe", "is", "the CFO", "stated", "m3").unwrap();
    }
    let pending = e.pending_conflicts().unwrap();
    assert_eq!(pending.len(), 1, "setup failure: expected one conflict");
    let conflict = &pending[0];
    let foreign = crate::knowledge::veracity::compute_fact_id("Zoe", "is", "the CFO");

    // A winner id that belongs to NEITHER side of the conflict must not supersede anything and
    // must leave the conflict unresolved.
    e.resolve_conflict(
        conflict.conflict_id,
        true,
        &foreign,
        &conflict.older_fact_id,
    )
    .unwrap();

    let c = e.store.conn.lock().unwrap();
    let superseded: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NOT NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        superseded, 0,
        "a foreign winning id must not supersede any fact"
    );
    let unresolved: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM conflicts WHERE resolution IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(unresolved, 1, "the conflict must remain unresolved");
}

// parity: test_e6a_followup_gaps.py::TestForgetCascadeToAnnotations::test_forget_deletes_annotations_for_memory_id (tests/test_e6a_followup_gaps.py:47)
#[test]
fn forget_cascades_annotations_and_embeddings() {
    let e = engine();
    let id = e
        .remember_with_vector(
            "Alice met Bob in San Francisco.",
            &RememberArgs {
                extract_entities: true,
                ..Default::default()
            },
            Some(&[0.5, 0.5, 0.0]),
            "mock",
        )
        .unwrap();
    {
        let c = e.store.conn.lock().unwrap();
        let pre: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(pre > 0, "setup failure: no annotations to forget");
        let emb: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(emb, 1, "setup failure: no embedding to forget");
    }

    assert!(e.forget(&id).unwrap(), "forget() found no memory");

    let c = e.store.conn.lock().unwrap();
    let post: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(post, 0, "annotations for forgotten memory still present");
    let emb: i64 = c
        .query_row(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(emb, 0, "embedding for forgotten memory still present");
}

// parity: test_e6a_followup_gaps.py::TestForgetCascadeToAnnotations::test_forget_doesnt_touch_other_memories_annotations (tests/test_e6a_followup_gaps.py:66)
#[test]
fn forget_leaves_other_memories_annotations_intact() {
    let e = engine();
    let args = RememberArgs {
        extract_entities: true,
        ..Default::default()
    };
    let id_to_forget = e.remember("Alice met Bob.", &args).unwrap();
    let id_to_keep = e.remember("Charlie met Dana.", &args).unwrap();

    let count = |id: &str| -> i64 {
        e.store
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap()
    };
    let keep_before = count(&id_to_keep);
    assert!(
        keep_before > 0,
        "setup failure: no annotations on the kept row"
    );

    assert!(e.forget(&id_to_forget).unwrap());

    assert_eq!(count(&id_to_forget), 0);
    assert_eq!(
        count(&id_to_keep),
        keep_before,
        "forget destroyed another memory's annotations"
    );
}

// parity: test_e6a_followup_gaps.py::TestForgetCrossSessionDoesNotLeakAnnotations::test_wrong_session_forget_does_not_touch_annotations (tests/test_e6a_followup_gaps.py:131)
// parity: test_e6a_followup_gaps.py::TestForgetCrossSessionDoesNotLeakAnnotations::test_correct_session_forget_still_works (tests/test_e6a_followup_gaps.py:148)
#[test]
fn cross_session_forget_is_denied_and_preserves_annotations() {
    let dir = std::env::temp_dir().join(format!("mnemosyne-xforget-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = |sid: &str| MnemosyneConfig {
        data_dir: dir.clone(),
        session_id: sid.to_string(),
        ..MnemosyneConfig::default()
    };
    let s_a = Engine::open(cfg("session-a")).expect("open session-a");
    let id = s_a
        .remember(
            "Alice met Bob in Paris.",
            &RememberArgs {
                extract_entities: true,
                ..Default::default()
            },
        )
        .unwrap();
    let count = |e: &Engine| -> i64 {
        e.store
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap()
    };
    let pre = count(&s_a);
    assert!(pre > 0, "setup failure: no annotations");

    // The session-scoped DELETE is the authorization boundary: a cross-session forget matches
    // no row, so the cascade must not fire (`beam.py` `forget_working` L3913).
    let s_b = Engine::open(cfg("session-b")).expect("open session-b");
    assert!(
        !s_b.forget(&id).unwrap(),
        "cross-session forget should report nothing deleted"
    );
    assert_eq!(
        count(&s_b),
        pre,
        "cross-session forget destroyed annotations"
    );

    // The owning session still forgets normally.
    assert!(s_a.forget(&id).unwrap());
    assert_eq!(count(&s_a), 0);

    let _ = std::fs::remove_dir_all(&dir);
}

// parity: test_e6a_followup_gaps.py::TestForgetCascadeIsAtomic::test_failed_cascade_rolls_back_working_memory_delete (tests/test_e6a_followup_gaps.py:178)
#[test]
fn forget_cascade_failure_rolls_back_row_delete() {
    let e = engine();
    let id = e
        .remember(
            "Atomic cascade test.",
            &RememberArgs {
                extract_entities: true,
                ..Default::default()
            },
        )
        .unwrap();
    // Break the cascade mid-way: the annotations DELETE will fail after the working_memory
    // DELETE succeeded. Python wraps the cascade in one transaction and rolls back.
    e.store
        .conn
        .lock()
        .unwrap()
        .execute("DROP TABLE annotations", [])
        .unwrap();

    assert!(
        e.forget(&id).is_err(),
        "a broken cascade must surface the error"
    );

    let survives: i64 = e
        .store
        .conn
        .lock()
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM working_memory WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        survives, 1,
        "working_memory row not rolled back after cascade failure"
    );
}

/// A [`BatchItem`] with just content (+ optional source/veracity), the common test shape.
fn batch_item(content: &str, source: Option<&str>, veracity: Option<&str>) -> BatchItem {
    BatchItem {
        content: content.to_string(),
        source: source.map(String::from),
        veracity: veracity.map(String::from),
        ..Default::default()
    }
}

/// Annotation kinds for one memory id, sorted (the `_annotation_rows` fixture,
/// tests/test_e2_remember_batch_enrichment.py:46).
fn annotation_kinds(e: &Engine, id: &str) -> Vec<String> {
    let c = e.store.conn.lock().unwrap();
    let mut stmt = c
        .prepare("SELECT DISTINCT kind FROM annotations WHERE memory_id = ?1 ORDER BY kind")
        .unwrap();
    let rows = stmt.query_map(params![id], |r| r.get(0)).unwrap();
    rows.collect::<std::result::Result<Vec<String>, _>>()
        .unwrap()
}

fn gist_count(e: &Engine, id: &str) -> i64 {
    e.store
        .conn
        .lock()
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM gists WHERE memory_id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap()
}

// parity: test_e2_remember_batch_enrichment.py::test_remember_batch_writes_temporal_annotations_for_every_row (tests/test_e2_remember_batch_enrichment.py:82)
// parity: test_e2_remember_batch_enrichment.py::test_remember_batch_extracts_gists_and_consolidated_facts (tests/test_e2_remember_batch_enrichment.py:122)
#[test]
fn remember_batch_enriches_every_row_with_annotations_gists_and_facts() {
    let e = engine();
    let ids = e
        .remember_batch(
            &[
                batch_item("Alice is the lead engineer", Some("convo"), None),
                batch_item("Bob is a contractor", Some("convo"), None),
            ],
            &RememberBatchArgs::default(),
        )
        .unwrap();
    assert_eq!(ids.len(), 2);
    for id in &ids {
        assert!(
            annotation_kinds(&e, id).contains(&"occurred_on".to_string()),
            "{id}: missing occurred_on — the temporal enrichment didn't fire"
        );
        assert!(
            gist_count(&e, id) >= 1,
            "{id}: missing gist — graph ingestion didn't fire"
        );
    }
    let facts: i64 = e
        .store
        .conn
        .lock()
        .unwrap()
        .query_row("SELECT COUNT(*) FROM consolidated_facts", [], |r| r.get(0))
        .unwrap();
    assert!(
        facts > 0,
        "consolidated_facts empty — the veracity consolidator wasn't consulted by the batch path"
    );
}

// parity: test_e2_remember_batch_enrichment.py::test_per_row_source_flows_to_has_source_annotation (tests/test_e2_remember_batch_enrichment.py:188)
// parity: test_e2_remember_batch_enrichment.py::TestReviewHardening::test_meta_by_id_dict_survives_python_o (tests/test_e2_remember_batch_enrichment.py:531)
// parity: test_e2_remember_batch_enrichment.py::test_per_row_veracity_threads_into_consolidated_facts (tests/test_e2_remember_batch_enrichment.py:151)
#[test]
fn remember_batch_threads_per_row_source_and_veracity() {
    let e = engine();
    let ids = e
        .remember_batch(
            &[
                batch_item("First from wiki", Some("wiki"), Some("stated")),
                batch_item("Second from email", Some("email"), Some("inferred")),
                batch_item("Third from doc", Some("doc"), None),
            ],
            &RememberBatchArgs::default(),
        )
        .unwrap();
    // Each row's has_source annotation carries its OWN source value, regardless of order.
    let c = e.store.conn.lock().unwrap();
    for (id, expected) in ids.iter().zip(["wiki", "email", "doc"]) {
        let value: String = c
            .query_row(
                "SELECT value FROM annotations WHERE memory_id = ?1 AND kind = 'has_source'",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(value, expected, "{id}: has_source mismatch");
    }
    // Per-row veracity landed on the rows themselves (stated / inferred / the unknown default).
    let veracities: Vec<String> = ids
        .iter()
        .map(|id| {
            c.query_row(
                "SELECT veracity FROM working_memory WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap()
        })
        .collect();
    assert_eq!(veracities, vec!["stated", "inferred", "unknown"]);
}

// parity: test_e2_remember_batch_enrichment.py::test_extract_entities_off_by_default (tests/test_e2_remember_batch_enrichment.py:213)
// parity: test_e2_remember_batch_enrichment.py::test_extract_entities_true_populates_mentions (tests/test_e2_remember_batch_enrichment.py:227)
#[test]
fn remember_batch_entity_extraction_is_opt_in() {
    let e = engine();
    let items = [batch_item(
        "Alice and Bob worked on the auth refactor",
        None,
        None,
    )];
    let off = e
        .remember_batch(&items, &RememberBatchArgs::default())
        .unwrap();
    assert!(
        !annotation_kinds(&e, &off[0]).contains(&"mentions".to_string()),
        "default-off entity extraction leaked a mentions annotation"
    );

    let on = e
        .remember_batch(
            &items,
            &RememberBatchArgs {
                extract_entities: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        annotation_kinds(&e, &on[0]).contains(&"mentions".to_string()),
        "extract_entities=true should produce mentions annotations"
    );
}

// parity: test_e2_remember_batch_enrichment.py::test_remember_batch_parity_with_remember_for_annotations (tests/test_e2_remember_batch_enrichment.py:284)
// parity: test_e2_remember_batch_enrichment.py::test_remember_batch_parity_with_remember_for_gists (tests/test_e2_remember_batch_enrichment.py:314)
#[test]
fn remember_batch_matches_single_remember_enrichment() {
    let content = "Frank is a database administrator";
    let single = engine();
    let single_id = single
        .remember(
            content,
            &RememberArgs {
                source: "wiki".to_string(),
                ..Default::default()
            },
        )
        .unwrap();

    let batched = engine();
    let batch_ids = batched
        .remember_batch(
            &[batch_item(content, Some("wiki"), None)],
            &RememberBatchArgs::default(),
        )
        .unwrap();

    assert_eq!(
        annotation_kinds(&single, &single_id),
        annotation_kinds(&batched, &batch_ids[0]),
        "annotation kinds diverge between remember() and remember_batch()"
    );
    assert_eq!(
        gist_count(&single, &single_id),
        gist_count(&batched, &batch_ids[0]),
        "gist counts diverge between remember() and remember_batch()"
    );
}

// parity: test_e2_remember_batch_enrichment.py::test_enrichment_exception_does_not_break_batch (tests/test_e2_remember_batch_enrichment.py:339)
#[test]
fn remember_batch_survives_per_row_enrichment_failure() {
    let e = engine();
    // Break the graph half of the enrichment pipeline for every row; inserts and the temporal
    // annotations must still land for the whole batch.
    e.store
        .conn
        .lock()
        .unwrap()
        .execute("DROP TABLE gists", [])
        .unwrap();

    let ids = e
        .remember_batch(
            &[
                batch_item("ok row 1", None, None),
                batch_item("row with boom inside", None, None),
                batch_item("ok row 3", None, None),
            ],
            &RememberBatchArgs::default(),
        )
        .expect("a broken enrichment helper must not fail the batch");
    assert_eq!(ids.len(), 3);

    let wm: i64 = e
        .store
        .conn
        .lock()
        .unwrap()
        .query_row("SELECT COUNT(*) FROM working_memory", [], |r| r.get(0))
        .unwrap();
    assert_eq!(wm, 3, "enrichment failure tore down working_memory inserts");
    for id in &ids {
        assert!(
            annotation_kinds(&e, id).contains(&"occurred_on".to_string()),
            "{id}: enrichment loop short-circuited — later rows lost their temporal annotation"
        );
    }
}

// parity: test_e2_remember_batch_enrichment.py::TestReviewHardening::test_remember_batch_emits_memory_added_event_per_row (tests/test_e2_remember_batch_enrichment.py:493)
#[test]
fn remember_batch_emits_memory_added_event_per_row() {
    let e = engine();
    let stream = e.enable_streaming();
    let ids = e
        .remember_batch(
            &[
                batch_item("Event row A", None, None),
                batch_item("Event row B", None, None),
                batch_item("Event row C", None, None),
            ],
            &RememberBatchArgs::default(),
        )
        .unwrap();

    let buffer = stream.get_buffer(None, None);
    let added: Vec<&crate::streaming::MemoryEvent> = buffer
        .iter()
        .filter(|ev| ev.event_type == crate::streaming::EventType::MemoryAdded)
        .collect();
    assert_eq!(added.len(), 3, "one MEMORY_ADDED per batch row: {buffer:?}");
    let event_ids: std::collections::HashSet<&str> =
        added.iter().map(|ev| ev.memory_id.as_str()).collect();
    assert_eq!(
        event_ids,
        ids.iter().map(String::as_str).collect(),
        "event memory_ids must match the returned batch ids"
    );
    // The always-on event log saw them too.
    let creates: i64 = e
        .store
        .conn
        .lock()
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM memory_events WHERE operation = 'CREATE'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(creates, 3);
}

// parity: test_e2_remember_batch_enrichment.py (`force_veracity` + `trust_tier` kwargs, beam.py:3047-3080)
#[test]
fn remember_batch_force_veracity_and_imported_trust_tier() {
    let e = engine();
    let ids = e
        .remember_batch(
            &[
                batch_item("row that self-elevates", None, Some("stated")),
                batch_item("row without a label", None, None),
            ],
            &RememberBatchArgs {
                veracity: Some("tool".to_string()),
                force_veracity: true,
                ..Default::default()
            },
        )
        .unwrap();
    let c = e.store.conn.lock().unwrap();
    for id in &ids {
        let (veracity, tier): (String, String) = c
            .query_row(
                "SELECT veracity, trust_tier FROM working_memory WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            veracity, "tool",
            "force_veracity must override per-item labels uniformly"
        );
        assert_eq!(
            tier, "IMPORTED",
            "batch ingest defaults to the IMPORTED tier"
        );
    }
}

/// Recall with per-call weight overrides through the public filter surface (`beam.py`
/// `recall(vec_weight=..., fts_weight=..., importance_weight=...)`).
fn recall_weighted(
    e: &Engine,
    query: &str,
    weights: (Option<f64>, Option<f64>, Option<f64>),
) -> Vec<MemoryRow> {
    let scope = RecallScope::default();
    e.recall_with_scope(&RecallReq {
        query,
        top_k: 5,
        query_vector: None,
        scope: &scope,
        filters: crate::config::RecallFilters {
            vec_weight: weights.0,
            fts_weight: weights.1,
            importance_weight: weights.2,
            ..Default::default()
        },
    })
    .unwrap()
}

// parity: test_configurable_scoring.py::TestRecallConfigurableWeights::test_high_importance_weight_boosts_high_importance_memories (tests/test_configurable_scoring.py:147)
// parity: test_configurable_scoring.py::TestPublicRecallConfigurableWeights::test_mnemosyne_recall_accepts_weight_params (tests/test_configurable_scoring.py:241)
#[test]
fn recall_importance_weight_override_reorders_results() {
    let e = engine();
    // A: strong lexical match, low importance. B: weaker lexical match, high importance.
    e.remember(
        "critical alert generic text",
        &RememberArgs {
            importance: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    e.remember(
        "critical system status",
        &RememberArgs {
            importance: 0.9,
            ..Default::default()
        },
    )
    .unwrap();

    // Keyword-dominated weights: the exact lexical match ranks first.
    let low_iw = recall_weighted(&e, "critical alert", (Some(0.5), Some(0.45), Some(0.05)));
    assert_eq!(low_iw.len(), 2, "both rows pass the gate: {low_iw:?}");
    assert!(
        low_iw[0].content.contains("generic"),
        "keyword-dominant weights must rank the lexical match first, got {low_iw:?}"
    );

    // Importance-dominated weights: the high-importance row overtakes it end-to-end.
    let high_iw = recall_weighted(&e, "critical alert", (Some(0.1), Some(0.1), Some(0.8)));
    assert!(
        high_iw[0].content.contains("system status"),
        "importance-dominant weights must reorder the ranking, got {high_iw:?}"
    );
}

// parity: test_configurable_scoring.py::TestNormalizeWeights::test_env_var_override (tests/test_configurable_scoring.py:83) — env vars map to the injected config in Rust
// parity: test_configurable_scoring.py::TestRecallConfigurableWeights::test_env_vars_affect_scoring (tests/test_configurable_scoring.py:185)
#[test]
fn configured_recall_weights_change_ranking_end_to_end() {
    // The Rust analog of `MNEMOSYNE_{VEC,FTS,IMPORTANCE}_WEIGHT`: the host injects
    // `recall_weights` through MnemosyneConfig, and recall uses them as its defaults.
    let e = Engine::open_in_memory(MnemosyneConfig {
        recall_weights: (0.1, 0.1, 0.8),
        ..MnemosyneConfig::default()
    })
    .unwrap();
    e.remember(
        "critical alert generic text",
        &RememberArgs {
            importance: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    e.remember(
        "critical system status",
        &RememberArgs {
            importance: 0.9,
            ..Default::default()
        },
    )
    .unwrap();

    let hits = e.recall("critical alert", 5).unwrap();
    assert!(
        hits[0].content.contains("system status"),
        "configured importance-heavy defaults must drive the ranking, got {hits:?}"
    );
    assert!(hits[0].importance >= 0.5, "top hit is the important row");
}

// parity: test_configurable_scoring.py::TestRecallConfigurableWeights::test_explicit_params_override_env_in_recall (tests/test_configurable_scoring.py:202)
#[test]
fn recall_explicit_weight_overrides_beat_configured_defaults() {
    // Importance-heavy configured defaults (the env-var layer in Python)…
    let e = Engine::open_in_memory(MnemosyneConfig {
        recall_weights: (0.1, 0.1, 0.8),
        ..MnemosyneConfig::default()
    })
    .unwrap();
    e.remember(
        "critical alert generic text",
        &RememberArgs {
            importance: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    e.remember(
        "critical system status",
        &RememberArgs {
            importance: 0.9,
            ..Default::default()
        },
    )
    .unwrap();
    // …drive the default ranking toward the important row…
    let defaults = e.recall("critical alert", 5).unwrap();
    assert!(
        defaults[0].content.contains("system status"),
        "config defaults rank the important row first, got {defaults:?}"
    );
    // …but explicit per-call weights take precedence over the configured defaults.
    let overridden = recall_weighted(&e, "critical alert", (Some(0.5), Some(0.45), Some(0.05)));
    assert!(
        overridden[0].content.contains("generic"),
        "explicit per-call weights must override the configured defaults, got {overridden:?}"
    );
}

// parity: test_configurable_scoring.py::TestRecallConfigurableWeights::test_zero_all_weights_uses_defaults_in_recall (tests/test_configurable_scoring.py:227)
// parity: test_configurable_scoring.py::TestEdgeCases::test_invalid_negative_param_clamped (tests/test_configurable_scoring.py:307)
// parity: test_configurable_scoring.py::TestEdgeCases::test_very_high_fts_weight (tests/test_configurable_scoring.py:298)
#[test]
fn recall_weight_override_edge_cases_never_break_recall() {
    let e = engine();
    e.remember(
        "exact text match phrase",
        &RememberArgs {
            importance: 0.5,
            ..Default::default()
        },
    )
    .unwrap();

    // All-zero overrides fall back to the defaults instead of zeroing every score.
    let zeros = recall_weighted(&e, "exact text match", (Some(0.0), Some(0.0), Some(0.0)));
    assert!(
        !zeros.is_empty(),
        "all-zero weights must fall back to defaults"
    );

    // Negative components are clamped, not propagated.
    let negative = recall_weighted(&e, "exact text", (Some(-0.5), Some(1.0), Some(0.5)));
    assert!(!negative.is_empty(), "negative weights must be clamped");

    // A pure-FTS weighting still surfaces the text match.
    let fts_only = recall_weighted(&e, "exact text match", (Some(0.0), Some(1.0), Some(0.0)));
    assert!(
        !fts_only.is_empty() && fts_only[0].content.contains("exact"),
        "fts-only weighting surfaces the text match, got {fts_only:?}"
    );
}

// parity: test_configurable_scoring.py::TestRecallConfigurableWeights::test_results_include_score_breakdown (tests/test_configurable_scoring.py:171)
// parity: test_configurable_scoring.py::TestRecallConfigurableWeights::test_weight_params_dont_break_temporal_scoring (tests/test_configurable_scoring.py:217)
#[test]
fn recall_weight_overrides_report_breakdown_and_compose_with_temporal() {
    let e = engine();
    e.remember(
        "Test content for scoring breakdown",
        &RememberArgs::default(),
    )
    .unwrap();

    let hits = recall_weighted(&e, "test content", (Some(0.4), Some(0.4), Some(0.2)));
    assert_eq!(hits.len(), 1);
    // The row carries the per-signal breakdown fields (`dense_score`/`fts_score`/`score`).
    assert!(hits[0].score > 0.0);
    assert!(hits[0].fts_score > 0.0, "fts signal populated: {hits:?}");
    assert_eq!(hits[0].dense_score, 0.0, "no vector supplied");

    // Weight overrides coexist with the Phase-3 temporal boost knobs.
    let scope = RecallScope::default();
    let temporal = e
        .recall_with_scope(&RecallReq {
            query: "test content",
            top_k: 5,
            query_vector: None,
            scope: &scope,
            filters: crate::config::RecallFilters {
                vec_weight: Some(0.4),
                fts_weight: Some(0.3),
                importance_weight: Some(0.3),
                temporal_weight: 0.5,
                query_time: Some("2099-01-01".to_string()),
                ..Default::default()
            },
        })
        .unwrap();
    assert!(
        !temporal.is_empty(),
        "temporal boost + weight overrides must compose"
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

/// Recall with a temporal-boost configuration through the scoped recall surface.
fn recall_temporal(
    e: &Engine,
    query: &'static str,
    temporal_weight: f64,
    temporal_halflife: Option<f64>,
) -> Vec<MemoryRow> {
    let scope = RecallScope::default();
    e.recall_with_scope(&RecallReq {
        query,
        top_k: 5,
        query_vector: None,
        scope: &scope,
        filters: crate::config::RecallFilters {
            temporal_weight,
            temporal_halflife,
            ..Default::default()
        },
    })
    .unwrap()
}

// PARITY: Mnemosyne tests/test_temporal_recall.py::TestTemporalRecallEndToEnd::test_temporal_boost_recent_vs_old
#[test]
fn temporal_boost_ranks_recent_over_old_end_to_end() {
    let e = engine();
    let test_args = RememberArgs {
        source: "test".to_string(),
        ..Default::default()
    };
    e.remember("Meeting about project alpha", &test_args)
        .unwrap();
    e.remember("Meeting about project beta", &test_args)
        .unwrap();
    {
        let conn = e.store.conn.lock().unwrap();
        let old = (chrono::Utc::now() - chrono::Duration::days(5))
            .format("%Y-%m-%dT%H:%M:%S")
            .to_string();
        let recent = (chrono::Utc::now() - chrono::Duration::hours(2))
            .format("%Y-%m-%dT%H:%M:%S")
            .to_string();
        conn.execute(
            "UPDATE working_memory SET timestamp = ?1 WHERE content LIKE '%alpha%'",
            params![old],
        )
        .unwrap();
        conn.execute(
            "UPDATE working_memory SET timestamp = ?1 WHERE content LIKE '%beta%'",
            params![recent],
        )
        .unwrap();
    }

    let hits = recall_temporal(&e, "meeting", 0.5, None);
    let score = |needle: &str| {
        hits.iter()
            .find(|h| h.content.contains(needle))
            .map(|h| h.score)
    };
    let alpha = score("alpha");
    let beta = score("beta");
    assert!(
        alpha.is_some() && beta.is_some(),
        "both memories must surface: {hits:?}"
    );
    assert!(
        beta.unwrap() > alpha.unwrap(),
        "recent memory must outrank stale with temporal boost: beta={beta:?} alpha={alpha:?}"
    );
}

// PARITY: Mnemosyne tests/test_temporal_recall.py::TestTemporalRecallEndToEnd::test_temporal_halflife_override
#[test]
fn temporal_halflife_override_changes_boost_end_to_end() {
    let e = engine();
    e.remember(
        "Memory from two days ago",
        &RememberArgs {
            source: "test".to_string(),
            ..Default::default()
        },
    )
    .unwrap();
    {
        let conn = e.store.conn.lock().unwrap();
        let two_days = (chrono::Utc::now() - chrono::Duration::days(2))
            .format("%Y-%m-%dT%H:%M:%S")
            .to_string();
        conn.execute(
            "UPDATE working_memory SET timestamp = ?1 WHERE content LIKE '%two days ago%'",
            params![two_days],
        )
        .unwrap();
    }

    // Only the per-call temporal_halflife differs; the base recency decay is identical, so any
    // score delta is attributable to the temporal boost knob (`beam.py` L5137-L5141).
    let score_short = recall_temporal(&e, "memory", 0.5, Some(6.0))
        .first()
        .map(|h| h.score)
        .unwrap_or(0.0);
    let score_long = recall_temporal(&e, "memory", 0.5, Some(168.0))
        .first()
        .map(|h| h.score)
        .unwrap_or(0.0);
    assert!(
        score_long > score_short,
        "a longer temporal halflife must boost a 2-day-old memory more: long={score_long} short={score_short}"
    );
}

// ---- A/B scoring toggles (`tests/test_ab_toggles.py`) ----

/// Seed one episodic row plus the graph edges + fact that let a recall claim the bonuses.
fn seed_episodic_with_graph_and_fact(e: &Engine) {
    let conn = e.store.conn.lock().unwrap();
    let ts = crate::util::now_iso();
    conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance) \
         VALUES ('ep-bonus', 'deploy production rollout plan', 'consolidation', ?1, 'default', 0.5)",
        params![ts],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO graph_edges (source, target, edge_type) VALUES ('ep-bonus', 'ep-other', 'related')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO graph_edges (source, target, edge_type) VALUES ('ep-other', 'ep-bonus', 'related')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO facts (fact_id, session_id, source_msg_id, subject, predicate, object) \
         VALUES ('fact-1', 'default', 'ep-bonus', 'team', 'deploys', 'production')",
        [],
    )
    .unwrap();
}

fn ep_score(e: &Engine, query: &str) -> Option<f64> {
    e.recall(query, 5)
        .unwrap()
        .into_iter()
        .find(|r| r.id == "ep-bonus")
        .map(|r| r.score)
}

// PARITY: Mnemosyne tests/test_ab_toggles.py::TestLinearBonusToggles::test_graph_bonus_disabled_does_not_apply
#[test]
fn graph_bonus_toggle_alters_episodic_recall_score() {
    // veracity multiplier defanged so only the graph bonus differs, matching the Python fixture.
    let on = Engine::open_in_memory(MnemosyneConfig {
        veracity_multiplier: false,
        ..MnemosyneConfig::default()
    })
    .unwrap();
    seed_episodic_with_graph_and_fact(&on);
    let off = Engine::open_in_memory(MnemosyneConfig {
        veracity_multiplier: false,
        graph_bonus: false,
        ..MnemosyneConfig::default()
    })
    .unwrap();
    seed_episodic_with_graph_and_fact(&off);

    let on_score = ep_score(&on, "deploy production rollout").expect("on hit");
    let off_score = ep_score(&off, "deploy production rollout").expect("off hit");
    assert!(
        on_score > off_score,
        "graph_bonus toggle must lift the score: on={on_score} off={off_score}"
    );
}

// PARITY: Mnemosyne tests/test_ab_toggles.py::TestLinearBonusToggles::test_fact_bonus_disabled_does_not_apply
#[test]
fn fact_bonus_toggle_alters_episodic_recall_score() {
    let base = |fact_bonus: bool| MnemosyneConfig {
        veracity_multiplier: false,
        graph_bonus: false,
        fact_bonus,
        ..MnemosyneConfig::default()
    };
    let on = Engine::open_in_memory(base(true)).unwrap();
    seed_episodic_with_graph_and_fact(&on);
    let off = Engine::open_in_memory(base(false)).unwrap();
    seed_episodic_with_graph_and_fact(&off);

    let on_score = ep_score(&on, "deploys production").expect("on hit");
    let off_score = ep_score(&off, "deploys production").expect("off hit");
    assert!(
        on_score > off_score,
        "fact_bonus toggle must lift the score: on={on_score} off={off_score}"
    );
}

// PARITY: Mnemosyne tests/test_ab_toggles.py::TestVeracityMultiplierToggle::test_disabled_makes_stated_unknown_score_equal
// PARITY: Mnemosyne tests/test_ab_toggles.py::TestVeracityMultiplierToggle::test_enabled_makes_stated_outrank_unknown
#[test]
fn veracity_multiplier_toggle_controls_stated_vs_unknown_ranking() {
    let seed = |e: &Engine| {
        let conn = e.store.conn.lock().unwrap();
        let ts = crate::util::now_iso();
        conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, veracity) \
             VALUES ('ep-stated', 'the user prefers dark mode', 'consolidation', ?1, 'default', 0.5, 'stated')",
            params![ts],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, veracity) \
             VALUES ('ep-unknown', 'the user prefers dark mode', 'consolidation', ?1, 'default', 0.5, 'unknown')",
            params![ts],
        )
        .unwrap();
    };
    let score_of = |e: &Engine, id: &str| {
        e.recall("dark mode", 10)
            .unwrap()
            .into_iter()
            .find(|r| r.id == id)
            .map(|r| r.score)
    };
    // Defang graph/fact/binary bonuses so only the veracity multiplier differs.
    let cfg = |veracity_multiplier: bool| MnemosyneConfig {
        veracity_multiplier,
        graph_bonus: false,
        fact_bonus: false,
        binary_bonus: false,
        ..MnemosyneConfig::default()
    };

    let off = Engine::open_in_memory(cfg(false)).unwrap();
    seed(&off);
    let s_stated = score_of(&off, "ep-stated").expect("stated off");
    let s_unknown = score_of(&off, "ep-unknown").expect("unknown off");
    assert!(
        (s_stated - s_unknown).abs() < 1e-9,
        "multiplier OFF: stated/unknown must score identically: {s_stated} vs {s_unknown}"
    );

    let on = Engine::open_in_memory(cfg(true)).unwrap();
    seed(&on);
    let s_stated_on = score_of(&on, "ep-stated").expect("stated on");
    let s_unknown_on = score_of(&on, "ep-unknown").expect("unknown on");
    assert!(
        s_stated_on > s_unknown_on,
        "multiplier ON: stated (1.0) must outrank unknown (0.8): {s_stated_on} vs {s_unknown_on}"
    );
}

// PARITY: Mnemosyne tests/test_ab_toggles.py::TestCrossTierDedupToggle (disabled_returns_input_unchanged + enabled_dedups_normally)
#[test]
fn cross_tier_dedup_toggle_controls_summary_source_collapse() {
    let seed = |e: &Engine| {
        let conn = e.store.conn.lock().unwrap();
        let ts = crate::util::now_iso();
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) \
             VALUES ('wm-src', 'deployment script for prod release', 'conversation', ?1, 'default', 0.5)",
            params![ts],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, summary_of) \
             VALUES ('ep-sum', 'Summary deployment script for prod release', 'consolidation', ?1, 'default', 0.5, 'wm-src')",
            params![ts],
        )
        .unwrap();
    };
    let pair_count = |e: &Engine| {
        e.recall("deployment", 10)
            .unwrap()
            .into_iter()
            .filter(|r| r.id == "wm-src" || r.id == "ep-sum")
            .count()
    };

    let on = Engine::open_in_memory(MnemosyneConfig::default()).unwrap();
    seed(&on);
    assert_eq!(
        pair_count(&on),
        1,
        "cross-tier dedup ON collapses the summary<->source pair to one"
    );

    let off = Engine::open_in_memory(MnemosyneConfig {
        cross_tier_dedup: false,
        ..MnemosyneConfig::default()
    })
    .unwrap();
    seed(&off);
    assert_eq!(
        pair_count(&off),
        2,
        "cross-tier dedup OFF leaves both the summary and its source"
    );
}

// ---- Polyphonic voice A/B toggles (`tests/test_ab_toggles.py::TestPolyphonicVoiceToggles`) ----

/// Build a polyphonic engine with a single vector-bearing row, toggling one voice.
fn poly_engine_with_vector_row(voice_vector: bool, voice_temporal: bool) -> Engine {
    let e = Engine::open_in_memory(MnemosyneConfig {
        recall_mode: RecallMode::Polyphonic,
        voice_vector,
        voice_temporal,
        ..MnemosyneConfig::default()
    })
    .unwrap();
    e.remember_with_vector(
        "quarterly revenue figures attachment",
        &RememberArgs::default(),
        Some(&[1.0, 0.0, 0.0]),
        "mock",
    )
    .unwrap();
    e
}

// PARITY: Mnemosyne tests/test_ab_toggles.py::TestPolyphonicVoiceToggles::test_vector_voice_disabled_returns_empty
#[test]
fn voice_vector_toggle_gates_the_vector_voice() {
    // Query "zzz" carries no lexical/graph/fact/temporal signal, so the aligned query vector is the
    // ONLY voice that can surface the row. Disabling the vector voice must drop it entirely.
    let enabled = poly_engine_with_vector_row(true, true);
    let hits_on = enabled
        .recall_with_vector("zzz", 5, Some(&[1.0, 0.0, 0.0]))
        .unwrap();
    assert!(
        hits_on
            .iter()
            .any(|h| h.content.contains("quarterly revenue")),
        "vector voice ON must surface the aligned row: {hits_on:?}"
    );

    let disabled = poly_engine_with_vector_row(false, true);
    let hits_off = disabled
        .recall_with_vector("zzz", 5, Some(&[1.0, 0.0, 0.0]))
        .unwrap();
    assert!(
        !hits_off
            .iter()
            .any(|h| h.content.contains("quarterly revenue")),
        "disabling the vector voice must drop its sole contribution: {hits_off:?}"
    );
}

// PARITY: Mnemosyne tests/test_ab_toggles.py::TestPolyphonicVoiceToggles::test_temporal_voice_disabled_returns_empty
#[test]
fn voice_temporal_toggle_gates_the_temporal_voice() {
    // A temporal-keyword query with no vector surfaces recent working rows via the temporal voice
    // alone; disabling that voice must surface nothing.
    let enabled = poly_engine_with_vector_row(true, true);
    let hits_on = enabled.recall("notes from last week", 5).unwrap();
    assert!(
        !hits_on.is_empty(),
        "temporal voice ON must surface recent rows on a temporal-cue query"
    );

    let disabled = poly_engine_with_vector_row(true, false);
    let hits_off = disabled.recall("notes from last week", 5).unwrap();
    assert!(
        hits_off.is_empty(),
        "disabling the temporal voice must drop its sole contribution: {hits_off:?}"
    );
}
