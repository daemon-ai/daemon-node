// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `mnemosyne_*` tool surface — port of the `__init__.py` JSON tool dispatch (L1750+).
//!
//! These tools are *not* part of the §11 [`MemoryProvider`](daemon_core::memory::MemoryProvider)
//! seam (which is about context, not dispatch). A host registers them through the §12
//! [`ToolRegistry`](daemon_core::tools) and routes calls to [`dispatch`]. The defs are
//! session-independent; the dispatch resolves the per-session [`Engine`] via the caller.
//!
//! Factored out of [`crate::provider`] so the surface is one table + one match rather than a giant
//! method. Embedding (remember/recall) and summarization (sleep) happen at this async seam before
//! the synchronous [`Engine`] is touched.

use crate::embeddings::Embedder;
use crate::engine::{
    CanonicalRemember, Engine, GraphLink, RecallReq, RememberArgs, SleepGroup, TripleAdd,
    TripleEnd, TripleQuery, ValidateArgs,
};
use crate::extract::Extractor;
use daemon_core::tools::ToolDef;
use serde_json::{json, Value};
use std::collections::HashMap;

fn def(name: &str, schema: &str) -> ToolDef {
    ToolDef {
        name: name.to_string(),
        schema: schema.to_string(),
    }
}

/// All `mnemosyne_*` tool defs (the `sync` feature adds three replication wrappers). Order is stable
/// so the registry/probe enumeration is deterministic.
pub fn defs() -> Vec<ToolDef> {
    #[allow(unused_mut)]
    let mut defs = vec![
        def(
            "mnemosyne_remember",
            r#"{"type":"object","properties":{"content":{"type":"string"},"importance":{"type":"number"},"scope":{"type":"string"},"veracity":{"type":"string"}},"required":["content"]}"#,
        ),
        def(
            "mnemosyne_recall",
            r#"{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer"},"author_id":{"type":"string"},"author_type":{"type":"string"},"channel_id":{"type":"string"}},"required":["query"]}"#,
        ),
        def(
            "mnemosyne_get",
            r#"{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}"#,
        ),
        def(
            "mnemosyne_update",
            r#"{"type":"object","properties":{"id":{"type":"string"},"content":{"type":"string"},"importance":{"type":"number"}},"required":["id"]}"#,
        ),
        def(
            "mnemosyne_forget",
            r#"{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}"#,
        ),
        def(
            "mnemosyne_invalidate",
            r#"{"type":"object","properties":{"id":{"type":"string"},"replacement_id":{"type":"string"}},"required":["id"]}"#,
        ),
        def(
            "mnemosyne_validate",
            r#"{"type":"object","properties":{"id":{"type":"string"},"action":{"type":"string"},"validator":{"type":"string"},"new_content":{"type":"string"},"note":{"type":"string"}},"required":["id","action"]}"#,
        ),
        def(
            "mnemosyne_sleep",
            r#"{"type":"object","properties":{"force":{"type":"boolean"}}}"#,
        ),
        def("mnemosyne_stats", r#"{"type":"object","properties":{}}"#),
        def("mnemosyne_diagnose", r#"{"type":"object","properties":{}}"#),
        def(
            "mnemosyne_triple_add",
            r#"{"type":"object","properties":{"subject":{"type":"string"},"predicate":{"type":"string"},"object":{"type":"string"},"valid_from":{"type":"string"},"valid_until":{"type":"string"},"source":{"type":"string"},"confidence":{"type":"number"},"supersede":{"type":"boolean"}},"required":["subject","predicate","object"]}"#,
        ),
        def(
            "mnemosyne_triple_end",
            r#"{"type":"object","properties":{"subject":{"type":"string"},"predicate":{"type":"string"},"object":{"type":"string"}},"required":["subject","predicate"]}"#,
        ),
        def(
            "mnemosyne_triple_query",
            r#"{"type":"object","properties":{"subject":{"type":"string"},"predicate":{"type":"string"},"object":{"type":"string"},"as_of":{"type":"string"}}}"#,
        ),
        def(
            "mnemosyne_remember_canonical",
            r#"{"type":"object","properties":{"owner_id":{"type":"string"},"category":{"type":"string"},"name":{"type":"string"},"body":{"type":"string"},"source":{"type":"string"},"confidence":{"type":"number"}},"required":["owner_id","category","name","body"]}"#,
        ),
        def(
            "mnemosyne_recall_canonical",
            r#"{"type":"object","properties":{"owner_id":{"type":"string"},"category":{"type":"string"},"name":{"type":"string"}},"required":["owner_id"]}"#,
        ),
        def(
            "mnemosyne_scratchpad_write",
            r#"{"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}"#,
        ),
        def(
            "mnemosyne_scratchpad_read",
            r#"{"type":"object","properties":{}}"#,
        ),
        def(
            "mnemosyne_scratchpad_clear",
            r#"{"type":"object","properties":{}}"#,
        ),
        def(
            "mnemosyne_graph_query",
            r#"{"type":"object","properties":{"id":{"type":"string"},"depth":{"type":"integer"}},"required":["id"]}"#,
        ),
        def(
            "mnemosyne_graph_link",
            r#"{"type":"object","properties":{"source":{"type":"string"},"target":{"type":"string"},"edge_type":{"type":"string"},"weight":{"type":"number"}},"required":["source","target"]}"#,
        ),
        def("mnemosyne_export", r#"{"type":"object","properties":{}}"#),
        def(
            "mnemosyne_import",
            r#"{"type":"object","properties":{"bundle":{"type":"object"}},"required":["bundle"]}"#,
        ),
        def(
            "mnemosyne_shared_remember",
            r#"{"type":"object","properties":{"content":{"type":"string"},"importance":{"type":"number"}},"required":["content"]}"#,
        ),
        def(
            "mnemosyne_shared_recall",
            r#"{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer"},"author_id":{"type":"string"},"author_type":{"type":"string"},"channel_id":{"type":"string"}},"required":["query"]}"#,
        ),
        def(
            "mnemosyne_shared_forget",
            r#"{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}"#,
        ),
        def(
            "mnemosyne_shared_stats",
            r#"{"type":"object","properties":{}}"#,
        ),
    ];
    #[cfg(feature = "sync")]
    defs.extend(sync_defs());
    defs
}

#[cfg(feature = "sync")]
fn sync_defs() -> Vec<ToolDef> {
    vec![
        def(
            "mnemosyne_sync_status",
            r#"{"type":"object","properties":{}}"#,
        ),
        def(
            "mnemosyne_sync_export",
            r#"{"type":"object","properties":{}}"#,
        ),
        def(
            "mnemosyne_sync_import",
            r#"{"type":"object","properties":{"bundle":{"type":"object"}},"required":["bundle"]}"#,
        ),
    ]
}

fn s<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

fn err(e: impl std::fmt::Display) -> String {
    json!({"status": "error", "error": e.to_string()}).to_string()
}

/// Run a sleep pass, summarizing via the LLM when present, else the engine's AAAK fallback.
/// Shared by the `mnemosyne_sleep` tool and the provider's session-boundary/auto-sleep hooks.
pub async fn run_sleep(
    engine: &Engine,
    extractor: &Extractor,
    force: bool,
) -> crate::Result<crate::engine::SleepReport> {
    if !extractor.available() {
        return engine.sleep(force);
    }
    let groups: Vec<SleepGroup> = engine.sleep_plan(force)?;
    if groups.is_empty() {
        return engine.finish_sleep(&groups, &HashMap::new());
    }
    // Heuristic embedding-cosine conflict detection, before summarization (`beam.py` L7705-L7731).
    // When the LLM gate is on, each pair is validated before invalidation; otherwise all detected
    // pairs are invalidated (older superseded by newer).
    if let Ok(conflicts) = engine.heuristic_sleep_conflicts(&groups) {
        let gate = engine.llm_conflict_detection() && extractor.available();
        for c in &conflicts {
            let confirmed = if gate {
                crate::knowledge::conflict::validate_conflict_pair(
                    extractor,
                    &c.older_content,
                    &c.newer_content,
                )
                .await
                .map(|v| v.is_conflict)
                .unwrap_or(false)
            } else {
                true
            };
            if confirmed {
                let _ = engine.invalidate(&c.older_id, Some(&c.newer_id));
            }
        }
    }
    let mut summaries = HashMap::new();
    for group in &groups {
        let prompt = Engine::summary_prompt(&group.contents);
        if let Some(text) = extractor.summarize(prompt).await {
            summaries.insert(group.source.clone(), text);
        }
    }
    engine.finish_sleep(&groups, &summaries)
}

/// Dispatch one `mnemosyne_*` tool by name, returning a JSON string result.
pub async fn dispatch(
    engine: &Engine,
    embedder: &Embedder,
    extractor: &Extractor,
    name: &str,
    args: Value,
) -> String {
    match name {
        "mnemosyne_remember" | "mnemosyne_shared_remember" => {
            let content = s(&args, "content").unwrap_or("");
            let importance = args.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5);
            let scope = if name == "mnemosyne_shared_remember" {
                "global".to_string()
            } else {
                s(&args, "scope").unwrap_or("session").to_string()
            };
            let veracity = s(&args, "veracity").unwrap_or("unknown").to_string();
            let vector = embedder.embed_query(content).await;
            let model = embedder.model().unwrap_or("");
            match engine.remember_with_vector(
                content,
                &RememberArgs {
                    source: "conversation".to_string(),
                    importance,
                    scope,
                    veracity,
                    ..Default::default()
                },
                vector.as_deref(),
                model,
            ) {
                Ok(id) => json!({"status": "ok", "memory_id": id}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_recall" | "mnemosyne_shared_recall" => {
            let query = s(&args, "query").unwrap_or("");
            let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
            let query_vec = embedder.embed_query(query).await;
            // Multi-agent identity overrides default to the engine's configured scope (`beam.py`
            // `recall` author/channel params L5030-L5032 default to the instance attrs).
            let mut scope = engine.config_scope();
            if let Some(v) = s(&args, "author_id") {
                scope.author_id = Some(v.to_string());
            }
            if let Some(v) = s(&args, "author_type") {
                scope.author_type = Some(v.to_string());
            }
            if let Some(v) = s(&args, "channel_id") {
                scope.channel_id = Some(v.to_string());
            }
            match engine.recall_with_scope(&RecallReq {
                query,
                top_k,
                query_vector: query_vec.as_deref(),
                scope: &scope,
            }) {
                Ok(rows) => {
                    let results: Vec<Value> = rows
                        .iter()
                        .map(|r| json!({"id": r.id, "content": r.content, "score": r.score}))
                        .collect();
                    json!({"query": query, "count": results.len(), "results": results}).to_string()
                }
                Err(e) => err(e),
            }
        }
        "mnemosyne_get" => match engine.get(s(&args, "id").unwrap_or("")) {
            Ok(Some(r)) => {
                json!({"status":"ok","id":r.id,"content":r.content,"importance":r.importance,"source":r.source,"timestamp":r.timestamp}).to_string()
            }
            Ok(None) => json!({"status": "not_found"}).to_string(),
            Err(e) => err(e),
        },
        "mnemosyne_update" => {
            let id = s(&args, "id").unwrap_or("");
            let content = s(&args, "content");
            let importance = args.get("importance").and_then(|v| v.as_f64());
            match engine.update(id, content, importance) {
                Ok(updated) => json!({"status": "ok", "updated": updated}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_forget" | "mnemosyne_shared_forget" => {
            match engine.forget(s(&args, "id").unwrap_or("")) {
                Ok(deleted) => json!({"status": "ok", "deleted": deleted}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_invalidate" => {
            match engine.invalidate(s(&args, "id").unwrap_or(""), s(&args, "replacement_id")) {
                Ok(changed) => json!({"status": "ok", "invalidated": changed}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_validate" => {
            match engine.validate(&ValidateArgs {
                id: s(&args, "id").unwrap_or(""),
                action: s(&args, "action").unwrap_or("confirm"),
                validator: s(&args, "validator"),
                new_content: s(&args, "new_content"),
                note: s(&args, "note"),
            }) {
                Ok(found) => json!({"status": "ok", "found": found}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_sleep" => {
            let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
            match run_sleep(engine, extractor, force).await {
                Ok(report) => json!({"status": "ok", "report": report}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_stats" | "mnemosyne_shared_stats" => match engine.stats() {
            Ok(stats) => json!({"status": "ok", "stats": stats}).to_string(),
            Err(e) => err(e),
        },
        "mnemosyne_diagnose" => match engine.diagnose() {
            Ok(d) => json!({"status": "ok", "diagnostics": d}).to_string(),
            Err(e) => err(e),
        },
        "mnemosyne_triple_add" => {
            match engine.triple_add(&TripleAdd {
                subject: s(&args, "subject").unwrap_or(""),
                predicate: s(&args, "predicate").unwrap_or(""),
                object: s(&args, "object").unwrap_or(""),
                valid_from: s(&args, "valid_from"),
                valid_until: s(&args, "valid_until"),
                source: s(&args, "source").unwrap_or("tool"),
                confidence: args.get("confidence").and_then(|v| v.as_f64()).unwrap_or(1.0),
                supersede: args.get("supersede").and_then(|v| v.as_bool()).unwrap_or(true),
            }) {
                Ok(id) => json!({"status": "ok", "row_id": id}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_triple_end" => {
            match engine.triple_end(&TripleEnd {
                subject: s(&args, "subject").unwrap_or(""),
                predicate: s(&args, "predicate").unwrap_or(""),
                object: s(&args, "object"),
                valid_until: s(&args, "valid_until"),
            }) {
                Ok(n) => json!({"status": "ok", "closed": n}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_triple_query" => {
            match engine.triple_query(&TripleQuery {
                subject: s(&args, "subject"),
                predicate: s(&args, "predicate"),
                object: s(&args, "object"),
                as_of: s(&args, "as_of"),
            }) {
                Ok(rows) => {
                    let triples: Vec<Value> = rows
                        .iter()
                        .map(|t| json!({"subject":t.subject,"predicate":t.predicate,"object":t.object,"valid_from":t.valid_from,"valid_until":t.valid_until}))
                        .collect();
                    json!({"status": "ok", "count": triples.len(), "triples": triples}).to_string()
                }
                Err(e) => err(e),
            }
        }
        "mnemosyne_remember_canonical" => {
            match engine.canonical_remember(&CanonicalRemember {
                owner_id: s(&args, "owner_id").unwrap_or(""),
                category: s(&args, "category").unwrap_or(""),
                name: s(&args, "name").unwrap_or(""),
                body: s(&args, "body").unwrap_or(""),
                source: s(&args, "source").unwrap_or("tool"),
                confidence: args.get("confidence").and_then(|v| v.as_f64()).unwrap_or(1.0),
            }) {
                Ok((row, status)) => json!({"status": "ok", "outcome": format!("{status:?}"), "version": row.version}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_recall_canonical" => {
            match engine.canonical_recall(
                s(&args, "owner_id").unwrap_or(""),
                s(&args, "category"),
                s(&args, "name"),
            ) {
                Ok(rows) => {
                    let facts: Vec<Value> = rows
                        .iter()
                        .map(|r| json!({"category":r.category,"name":r.name,"body":r.body,"version":r.version}))
                        .collect();
                    json!({"status": "ok", "count": facts.len(), "facts": facts}).to_string()
                }
                Err(e) => err(e),
            }
        }
        "mnemosyne_scratchpad_write" => {
            match engine.scratchpad_write(s(&args, "content").unwrap_or("")) {
                Ok(id) => json!({"status": "ok", "id": id}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_scratchpad_read" => match engine.scratchpad_read() {
            Ok(notes) => {
                let items: Vec<Value> = notes
                    .iter()
                    .map(|(id, content)| json!({"id": id, "content": content}))
                    .collect();
                json!({"status": "ok", "count": items.len(), "notes": items}).to_string()
            }
            Err(e) => err(e),
        },
        "mnemosyne_scratchpad_clear" => match engine.scratchpad_clear() {
            Ok(n) => json!({"status": "ok", "cleared": n}).to_string(),
            Err(e) => err(e),
        },
        "mnemosyne_graph_query" => {
            let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
            match engine.graph_query(s(&args, "id").unwrap_or(""), depth) {
                Ok(rels) => {
                    let related: Vec<Value> = rels
                        .iter()
                        .map(|r| json!({"id": r.memory_id, "edge_type": r.edge_type, "weight": r.weight, "depth": r.depth}))
                        .collect();
                    json!({"status": "ok", "count": related.len(), "related": related}).to_string()
                }
                Err(e) => err(e),
            }
        }
        "mnemosyne_graph_link" => {
            match engine.graph_link(&GraphLink {
                source: s(&args, "source").unwrap_or(""),
                target: s(&args, "target").unwrap_or(""),
                edge_type: s(&args, "edge_type").unwrap_or("related_to"),
                weight: args.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0),
            }) {
                Ok(()) => json!({"status": "ok"}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_export" => match engine.export() {
            Ok(bundle) => json!({"status": "ok", "bundle": bundle}).to_string(),
            Err(e) => err(e),
        },
        "mnemosyne_import" => match args.get("bundle") {
            Some(bundle) => match engine.import(bundle) {
                Ok(n) => json!({"status": "ok", "imported": n}).to_string(),
                Err(e) => err(e),
            },
            None => err("missing 'bundle'"),
        },
        #[cfg(feature = "sync")]
        "mnemosyne_sync_status" => match engine.diagnose() {
            Ok(d) => json!({"status": "ok", "replication": "local-only", "diagnostics": d}).to_string(),
            Err(e) => err(e),
        },
        #[cfg(feature = "sync")]
        "mnemosyne_sync_export" => match engine.export() {
            Ok(bundle) => json!({"status": "ok", "bundle": bundle}).to_string(),
            Err(e) => err(e),
        },
        #[cfg(feature = "sync")]
        "mnemosyne_sync_import" => match args.get("bundle") {
            Some(bundle) => match engine.import(bundle) {
                Ok(n) => json!({"status": "ok", "imported": n}).to_string(),
                Err(e) => err(e),
            },
            None => err("missing 'bundle'"),
        },
        _ => json!({"status": "unknown_tool", "tool": name}).to_string(),
    }
}
