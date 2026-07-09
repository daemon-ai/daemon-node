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
    CanonicalRemember, Engine, GraphLink, GroupSummary, RecallReq, RememberArgs, SleepGroup,
    TripleAdd, TripleEnd, TripleQuery,
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
            r#"{"type":"object","properties":{"content":{"type":"string"},"importance":{"type":"number"},"source":{"type":"string"},"scope":{"type":"string"},"valid_until":{"type":"string"},"extract_entities":{"type":"boolean"},"extract":{"type":"boolean"},"metadata":{"type":"object"},"veracity":{"type":"string"}},"required":["content"]}"#,
        ),
        def(
            "mnemosyne_recall",
            r#"{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},"temporal_weight":{"type":"number"},"query_time":{"type":"string"},"temporal_halflife":{"type":"number"},"vec_weight":{"type":"number"},"fts_weight":{"type":"number"},"importance_weight":{"type":"number"},"from_date":{"type":"string"},"to_date":{"type":"string"},"source":{"type":"string"},"topic":{"type":"string"},"veracity":{"type":"string"},"memory_type":{"type":"string"},"author_id":{"type":"string"},"author_type":{"type":"string"},"channel_id":{"type":"string"}},"required":["query"]}"#,
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
            r#"{"type":"object","properties":{"memory_id":{"type":"string"},"action":{"type":"string","enum":["attest","update","invalidate","delete"]},"validator":{"type":"string"},"new_content":{"type":"string"},"note":{"type":"string"},"bank":{"type":"string","enum":["private","surface"]}},"required":["memory_id","action"]}"#,
        ),
        def(
            "mnemosyne_sleep",
            r#"{"type":"object","properties":{"force":{"type":"boolean"},"dry_run":{"type":"boolean"}}}"#,
        ),
        def("mnemosyne_stats", r#"{"type":"object","properties":{}}"#),
        def(
            "mnemosyne_diagnose",
            r#"{"type":"object","properties":{
                "repair_vec_working":{"type":"boolean","description":"Idempotently backfill missing derived vectors (episodic MIB binaries) from stored embeddings"},
                "dry_run":{"type":"boolean","description":"With repair_vec_working, report what would be repaired without writing"}}}"#,
        ),
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
            r#"{"type":"object","properties":{"content":{"type":"string"},"kind":{"type":"string","enum":["meta","preference","correction","identity"]},"importance":{"type":"number"},"metadata":{"type":"object"},"veracity":{"type":"string"}},"required":["content"]}"#,
        ),
        def(
            "mnemosyne_shared_recall",
            r#"{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}"#,
        ),
        def(
            "mnemosyne_shared_forget",
            r#"{"type":"object","properties":{"memory_id":{"type":"string"}},"required":["memory_id"]}"#,
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
    // The three replication tools (`sync_adapter.py` ALL_SYNC_TOOL_SCHEMAS): all arg-less.
    vec![
        def(
            "mnemosyne_sync_push",
            r#"{"type":"object","properties":{}}"#,
        ),
        def(
            "mnemosyne_sync_pull",
            r#"{"type":"object","properties":{}}"#,
        ),
        def(
            "mnemosyne_sync_status",
            r#"{"type":"object","properties":{}}"#,
        ),
    ]
}

fn s<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

fn err(e: impl std::fmt::Display) -> String {
    json!({"status": "error", "error": e.to_string()}).to_string()
}

/// Everything a tool call can touch: the private bank plus the optional shared-surface bank
/// (`__init__.py` `_surface_beam`). Built per call by [`crate::MnemosyneProvider::call_tool`].
pub struct ToolCx<'a> {
    /// The private (per-profile) bank engine.
    pub engine: &'a Engine,
    /// The embedding seam (empty in keyword-only mode).
    pub embedder: &'a Embedder,
    /// The LLM seam (empty when no provider is injected).
    pub extractor: &'a Extractor,
    /// The shared-surface engine; `None` = init failed/unavailable.
    pub surface: Option<&'a Engine>,
}

impl ToolCx<'_> {
    fn surface(&self) -> std::result::Result<&Engine, String> {
        self.surface
            .ok_or_else(|| json!({"error": "shared surface DB is not initialized"}).to_string())
    }
}

/// The valid shared-surface kinds (`_handle_shared_remember` L1984).
const SURFACE_KINDS: &[&str] = &["meta", "preference", "correction", "identity"];

/// Stable shared-surface id: `sf_` + sha256 of the normalized content (`_surface_hash` L1936).
fn surface_hash(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let normalized = content
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let digest = Sha256::digest(format!("surface:v1:{normalized}").as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    format!("sf_{}", &hex[..24])
}

/// Prefix surface content with its kind label unless already labeled (`_surface_label` L1942).
fn surface_label(content: &str, kind: &str) -> String {
    let lower = content.to_lowercase();
    for prefix in [
        "surface meta:",
        "surface preference:",
        "surface correction:",
        "surface identity:",
        "surface fact:",
    ] {
        if lower.starts_with(prefix) {
            return content.to_string();
        }
    }
    let label = match kind {
        "preference" => "Surface preference",
        "correction" => "Surface correction",
        "identity" => "Surface identity",
        _ => "Surface meta",
    };
    format!("{label}: {content}")
}

/// One recall row as the tool JSON shape (Python `dict(row)` result rows), tagged with its bank.
fn row_json(r: &crate::engine::MemoryRow, bank: &str) -> Value {
    let mut v = json!({
        "id": r.id,
        "content": r.content,
        "score": r.score,
        "importance": r.importance,
        "source": r.source,
        "timestamp": r.timestamp,
        "veracity": r.veracity,
        "trust_tier": r.trust_tier,
        "bank": bank,
    });
    if bank == "surface" {
        v["shared_surface"] = json!(true);
    }
    v
}

/// Run a sleep pass, summarizing via the LLM when present (AAAK fallback otherwise) and embedding
/// each final summary at this async seam so [`Engine::finish_sleep`] can persist the consolidation
/// output's vector + MIB binary (`beam.py` `consolidate_to_episodic` L4005-L4032). Shared by the
/// `mnemosyne_sleep` tool and the provider's session-boundary/auto-sleep hooks.
pub async fn run_sleep(
    engine: &Engine,
    embedder: &Embedder,
    extractor: &Extractor,
    force: bool,
) -> crate::Result<crate::engine::SleepReport> {
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
                crate::knowledge::conflict::validate_conflict_pair_logged(
                    extractor,
                    engine,
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
    let mut summaries: HashMap<String, GroupSummary> = HashMap::new();
    for group in &groups {
        let llm_text = if extractor.available() {
            // Optional pre-compression of the summarization input (`beam.py` L7736-L7743):
            // consulted only when a host has materialized the plugin manager AND enabled the
            // compression plugin (identity transform while no backend is linked).
            use crate::plugins::MnemosynePlugin as _;
            let lines = match engine.plugins_if_active() {
                Some(pm) if pm.compression().enabled() => {
                    pm.compression().compress_lines(group.contents.clone())
                }
                _ => group.contents.clone(),
            };
            extractor.summarize(Engine::summary_prompt(&lines)).await
        } else {
            None
        };
        let (text, llm) = match llm_text.map(|t| crate::util::strip_think(&t)) {
            Some(t) if !t.is_empty() => (t, true),
            _ => (group.aaak_summary(), false),
        };
        // Embed the *final* summary text (LLM or AAAK); `None` in keyword-only mode.
        let embedding = embedder.embed_query(&text).await;
        summaries.insert(
            group.source.clone(),
            GroupSummary {
                text,
                llm,
                embedding,
                model: embedder.model().unwrap_or("").to_string(),
            },
        );
    }
    engine.finish_sleep(&groups, &summaries)
}

/// Dispatch one `mnemosyne_*` tool by name, returning a JSON string result.
pub async fn dispatch(cx: &ToolCx<'_>, name: &str, args: Value) -> String {
    let engine = cx.engine;
    let embedder = cx.embedder;
    let extractor = cx.extractor;
    match name {
        "mnemosyne_remember" => {
            let content = s(&args, "content").unwrap_or("");
            if content.is_empty() {
                return json!({"error": "content is required"}).to_string();
            }
            let importance = args.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5);
            // Tool writes default to source "user" (`_handle_remember` L1838), not the
            // engine's "conversation" default.
            let source = s(&args, "source").unwrap_or("user").to_string();
            let scope = s(&args, "scope").unwrap_or("session").to_string();
            let veracity = crate::knowledge::veracity::clamp_veracity(
                s(&args, "veracity").unwrap_or(""),
                "mnemosyne_remember",
            );
            let extract_entities = args
                .get("extract_entities")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let extract = args.get("extract").and_then(|v| v.as_bool()).unwrap_or(false);
            let metadata = args.get("metadata").filter(|m| m.is_object()).cloned();
            let vector = embedder.embed_query(content).await;
            let model = embedder.model().unwrap_or("");
            let id = match engine.remember_with_vector(
                content,
                &RememberArgs {
                    source,
                    importance,
                    scope,
                    veracity: veracity.clone(),
                    metadata: metadata.clone(),
                    valid_until: s(&args, "valid_until").map(String::from),
                    extract_entities,
                    extract,
                    ..Default::default()
                },
                vector.as_deref(),
                model,
            ) {
                Ok(id) => id,
                Err(e) => return err(e),
            };
            // `extract=true` LLM fact extraction happens at this async seam (`beam.py` runs it
            // inline in `remember`; the sync engine records the request).
            if extract && extractor.available() {
                if let Some(extracted) = extractor.extract(content).await {
                    let _ = engine.ingest_extracted(&id, &extracted);
                }
            }
            json!({
                "status": "stored",
                "memory_id": id,
                "content_preview": content.chars().take(100).collect::<String>(),
                "extract_entities": extract_entities,
                "extract": extract,
                "metadata": metadata,
                "veracity": veracity,
            })
            .to_string()
        }
        "mnemosyne_recall" => {
            let query = s(&args, "query").unwrap_or("");
            if query.is_empty() {
                return json!({"error": "query is required"}).to_string();
            }
            // Python's tool arg is `limit` (`_handle_recall` L1878); accept `top_k` too.
            let top_k = args
                .get("limit")
                .or_else(|| args.get("top_k"))
                .and_then(|v| v.as_u64())
                .unwrap_or(5) as usize;
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
            let temporal_weight = args
                .get("temporal_weight")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            // Row filters + temporal knobs + per-call weight overrides (forwarded only when the
            // caller actually supplied them — issue #45).
            let filters = crate::config::RecallFilters {
                from_date: s(&args, "from_date").map(String::from),
                to_date: s(&args, "to_date").map(String::from),
                source: s(&args, "source").map(String::from),
                topic: s(&args, "topic").map(String::from),
                veracity: s(&args, "veracity").map(String::from),
                memory_type: s(&args, "memory_type").map(String::from),
                temporal_weight,
                query_time: s(&args, "query_time").map(String::from),
                temporal_halflife: args.get("temporal_halflife").and_then(|v| v.as_f64()),
                vec_weight: args.get("vec_weight").and_then(|v| v.as_f64()),
                fts_weight: args.get("fts_weight").and_then(|v| v.as_f64()),
                importance_weight: args.get("importance_weight").and_then(|v| v.as_f64()),
            };
            let rows = match engine.recall_with_scope(&RecallReq {
                query,
                top_k,
                query_vector: query_vec.as_deref(),
                scope: &scope,
                filters,
            }) {
                Ok(rows) => rows,
                Err(e) => return err(e),
            };
            let mut results: Vec<Value> = rows.iter().map(|r| row_json(r, "private")).collect();
            // Optionally merge shared-surface results by score (`_handle_recall` L1906-L1925).
            let surface_read = engine.config().shared_surface_read;
            if surface_read {
                if let Some(surface) = cx.surface {
                    if let Ok(srows) = surface.recall_with_vector(query, top_k, query_vec.as_deref())
                    {
                        results.extend(srows.iter().map(|r| row_json(r, "surface")));
                        results.sort_by(|a, b| {
                            let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            sb.total_cmp(&sa)
                        });
                        results.truncate(top_k);
                    }
                }
            }
            json!({
                "query": query,
                "count": results.len(),
                "temporal_weight": temporal_weight,
                "shared_surface_read": surface_read,
                "results": results,
            })
            .to_string()
        }
        "mnemosyne_shared_remember" => {
            let surface = match cx.surface() {
                Ok(sf) => sf,
                Err(e) => return e,
            };
            let content = s(&args, "content").unwrap_or("").trim().to_string();
            if content.is_empty() {
                return json!({"error": "content is required"}).to_string();
            }
            if content.starts_with("[USER]") || content.starts_with("[ASSISTANT]") {
                return json!({"error": "raw conversation content is not allowed in shared memory"})
                    .to_string();
            }
            let kind = s(&args, "kind").unwrap_or("meta").trim().to_lowercase();
            if !SURFACE_KINDS.contains(&kind.as_str()) {
                return json!({"error": "kind must be one of: meta, preference, correction, identity"})
                    .to_string();
            }
            let importance = args
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.8)
                .clamp(0.0, 1.0);
            let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));
            if !metadata.is_object() {
                return json!({"error": "metadata must be an object"}).to_string();
            }
            let veracity = crate::knowledge::veracity::clamp_veracity(
                s(&args, "veracity").unwrap_or(""),
                "mnemosyne_shared_remember",
            );
            let surface_content = surface_label(&content, &kind);
            let stable_id = surface_hash(&surface_content);
            let mut meta = metadata;
            meta["shared_memory"] = json!(true);
            meta["surface_kind"] = json!(kind);
            meta["write_path"] = json!("manual_tool");
            meta["source_profile_session"] = json!(engine.session_id());
            let existing = surface.find_existing(&surface_content).ok().flatten();
            let vector = embedder.embed_query(&surface_content).await;
            let model = embedder.model().unwrap_or("");
            let memory_id = match surface.remember_with_vector(
                &surface_content,
                &RememberArgs {
                    source: "surface_manual".to_string(),
                    importance,
                    scope: "global".to_string(),
                    veracity: veracity.clone(),
                    metadata: Some(meta),
                    memory_id: Some(stable_id),
                    ..Default::default()
                },
                vector.as_deref(),
                model,
            ) {
                Ok(id) => id,
                Err(e) => return err(e),
            };
            // The tool-level audit event lands in the PRIVATE bank's audit_log with bank="surface"
            // (`_audit_event` calls, `audit.py` — the log is co-located with the provider DB).
            engine.audit_tool(
                "shared_remember",
                Some(&memory_id),
                "surface",
                "mnemosyne_shared_remember",
                Some(&json!({"kind": kind, "existing": existing.is_some()})),
            );
            json!({
                "status": if existing.is_some() { "existing_shared" } else { "stored_shared" },
                "memory_id": memory_id,
                "content_preview": surface_content.chars().take(120).collect::<String>(),
                "shared_db": surface.config().bank_db_path().display().to_string(),
                "kind": kind,
                "veracity": veracity,
            })
            .to_string()
        }
        "mnemosyne_shared_recall" => {
            let surface = match cx.surface() {
                Ok(sf) => sf,
                Err(e) => return e,
            };
            let query = s(&args, "query").unwrap_or("");
            if query.is_empty() {
                return json!({"error": "query is required"}).to_string();
            }
            let top_k = args
                .get("limit")
                .or_else(|| args.get("top_k"))
                .and_then(|v| v.as_u64())
                .unwrap_or(5) as usize;
            let query_vec = embedder.embed_query(query).await;
            match surface.recall_with_vector(query, top_k, query_vec.as_deref()) {
                Ok(rows) => {
                    let results: Vec<Value> = rows.iter().map(|r| row_json(r, "surface")).collect();
                    json!({
                        "query": query,
                        "count": results.len(),
                        "shared_db": surface.config().bank_db_path().display().to_string(),
                        "results": results,
                    })
                    .to_string()
                }
                Err(e) => err(e),
            }
        }
        "mnemosyne_shared_forget" => {
            let surface = match cx.surface() {
                Ok(sf) => sf,
                Err(e) => return e,
            };
            let memory_id = s(&args, "memory_id").or_else(|| s(&args, "id")).unwrap_or("");
            if memory_id.is_empty() {
                return json!({"error": "memory_id is required"}).to_string();
            }
            match surface.forget(memory_id) {
                Ok(deleted) => {
                    if deleted {
                        engine.audit_tool(
                            "shared_forget",
                            Some(memory_id),
                            "surface",
                            "mnemosyne_shared_forget",
                            None,
                        );
                    }
                    json!({
                        "status": if deleted { "deleted" } else { "not_found" },
                        "memory_id": memory_id,
                        "shared_db": surface.config().bank_db_path().display().to_string(),
                    })
                    .to_string()
                }
                Err(e) => err(e),
            }
        }
        "mnemosyne_shared_stats" => {
            let surface = match cx.surface() {
                Ok(sf) => sf,
                Err(e) => return e,
            };
            match surface.stats() {
                Ok(stats) => json!({
                    "provider": "mnemosyne_shared",
                    "shared_db": surface.config().bank_db_path().display().to_string(),
                    "stats": stats,
                })
                .to_string(),
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
        "mnemosyne_forget" => match engine.forget(s(&args, "id").unwrap_or("")) {
            Ok(deleted) => json!({"status": "ok", "deleted": deleted}).to_string(),
            Err(e) => err(e),
        },
        "mnemosyne_invalidate" => {
            match engine.invalidate(s(&args, "id").unwrap_or(""), s(&args, "replacement_id")) {
                Ok(changed) => json!({"status": "ok", "invalidated": changed}).to_string(),
                Err(e) => err(e),
            }
        }
        "mnemosyne_validate" => {
            // Collaborative attestation over either bank (`_handle_validate`, `__init__.py`
            // L2091-L2207): any agent can attest/update/invalidate/delete any memory; the
            // original author_id is preserved and the ring buffer keeps the last 3 validations.
            let memory_id = s(&args, "memory_id").or_else(|| s(&args, "id")).unwrap_or("");
            let action = s(&args, "action").unwrap_or("");
            let bank = s(&args, "bank").unwrap_or("private");
            let new_content = s(&args, "new_content").filter(|c| !c.is_empty());
            let note = s(&args, "note").filter(|n| !n.is_empty());
            if memory_id.is_empty() {
                return json!({"error": "memory_id is required"}).to_string();
            }
            if !matches!(action, "attest" | "update" | "invalidate" | "delete") {
                return json!({"error": format!("unknown action: {action}")}).to_string();
            }
            if !matches!(bank, "private" | "surface") {
                return json!({"error": format!("unknown bank: {bank}")}).to_string();
            }
            if action == "update" && new_content.is_none() {
                return json!({"error": "new_content is required for action='update'"}).to_string();
            }
            // Validator identity: explicit arg, else the configured author identity — the Rust
            // analog of Python's `self._agent_identity` — else "unknown".
            let validator = s(&args, "validator")
                .filter(|v| !v.is_empty())
                .map(str::to_string)
                .or_else(|| engine.config().author_id.clone())
                .unwrap_or_else(|| "unknown".to_string());
            let target = if bank == "surface" {
                match cx.surface() {
                    Ok(sf) => sf,
                    Err(e) => return e,
                }
            } else {
                engine
            };
            match target.validate_action(memory_id, action, &validator, new_content, note) {
                Ok(Some(outcome)) => {
                    // Tool-level audit into the private provider log, bank-stamped (`_audit_event`).
                    engine.audit_tool(
                        &format!("validate_{action}"),
                        Some(memory_id),
                        bank,
                        "mnemosyne_validate",
                        None,
                    );
                    json!({
                        "status": format!("validation_{action}"),
                        "memory_id": memory_id,
                        "bank": bank,
                        "validator": validator,
                        "author_id": outcome.author_id,
                        "previous_content": outcome.previous_content.chars().take(200).collect::<String>(),
                    })
                    .to_string()
                }
                Ok(None) => json!({
                    "error": "memory_not_found",
                    "memory_id": memory_id,
                    "bank": bank,
                })
                .to_string(),
                Err(e) => json!({
                    "error": "validation_failed",
                    "reason": e.to_string(),
                    "memory_id": memory_id,
                })
                .to_string(),
            }
        }
        "mnemosyne_sleep" => {
            let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
            let dry_run = args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false);
            if dry_run {
                // Report the would-be plan without claiming rows (`beam.py` sleep L7639).
                return match engine.sleep_plan_dry_run(force) {
                    Ok(groups) => {
                        let items: usize = groups.iter().map(|g| g.ids.len()).sum();
                        let group_summaries: Vec<Value> = groups
                            .iter()
                            .map(|g| json!({"source": g.source, "items": g.ids.len()}))
                            .collect();
                        json!({
                            "status": "dry_run",
                            "would_consolidate": items,
                            "groups": group_summaries,
                        })
                        .to_string()
                    }
                    Err(e) => err(e),
                };
            }
            match run_sleep(engine, embedder, extractor, force).await {
                Ok(report) => {
                    engine.audit_tool(
                        "sleep",
                        None,
                        "private",
                        "mnemosyne_sleep",
                        Some(&json!({"status": "consolidated"})),
                    );
                    json!({"status": "ok", "report": report}).to_string()
                }
                Err(e) => err(e),
            }
        }
        "mnemosyne_stats" => match engine.stats() {
            Ok(stats) => json!({"status": "ok", "stats": stats}).to_string(),
            Err(e) => err(e),
        },
        // Full diagnostics scan (`_handle_diagnose`): summary + JSONL log + optional idempotent
        // vector repair. The arg keeps Python's wire name `repair_vec_working` even though the
        // Rust repair targets the §7 stores (episodic MIB binaries), so clients stay compatible.
        "mnemosyne_diagnose" => {
            let mut summary = crate::diagnose::run_diagnostics(
                engine,
                embedder,
                extractor,
                crate::diagnose::DiagnoseOptions {
                    repair_vec_working: args
                        .get("repair_vec_working")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    dry_run: args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false),
                },
            );
            if engine.is_persistent() {
                summary["active_provider_db_path"] =
                    json!(engine.config().bank_db_path().display().to_string());
            }
            summary.to_string()
        }
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
                    // Full-row output like Python's dict(row) results (source/confidence/
                    // valid_from included).
                    let facts: Vec<Value> = rows
                        .iter()
                        .map(|r| {
                            json!({
                                "category": r.category,
                                "name": r.name,
                                "body": r.body,
                                "version": r.version,
                                "source": r.source,
                                "confidence": r.confidence,
                                "valid_from": r.valid_from,
                            })
                        })
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
        "mnemosyne_sync_push" | "mnemosyne_sync_pull" | "mnemosyne_sync_status" => {
            sync_tool(engine, name).await
        }
        _ => json!({"status": "unknown_tool", "tool": name}).to_string(),
    }
}

/// The three replication tools (`sync_adapter.py` `handle_tool_call`), on [`crate::sync`] +
/// [`MnemosyneConfig`](crate::MnemosyneConfig) knobs instead of env vars.
///
/// Divergences from the Python adapter (spec §12.1): cursors persist per remote under the same
/// `sync_meta` keys [`crate::sync::SyncEngine::sync_with`] uses (Python's tools share one global
/// `last_sync_cursor` between push and pull, which cross-contaminates the two directions and any
/// second remote), and the pull request sends `since` — the parameter the server actually reads —
/// where Python sent `since_token` and silently full-pulled every time.
#[cfg(feature = "sync")]
async fn sync_tool(engine: &Engine, name: &str) -> String {
    use crate::sync::{SyncEncryption, SyncEngine};

    // Python truncates cursors for display: `cursor[:30] + "..."`.
    fn trunc(cursor: &str) -> String {
        if cursor.chars().count() > 30 {
            format!("{}...", cursor.chars().take(30).collect::<String>())
        } else {
            cursor.to_string()
        }
    }

    let cfg = engine.config();
    let encryption = match cfg.sync_key.as_deref() {
        Some(source) => match SyncEncryption::from_key_source(source) {
            Ok(enc) => enc,
            Err(e) => return err(format!("Sync adapter not available: {e}")),
        },
        None => None,
    };
    let encrypt_enabled = encryption.is_some();
    let se = match SyncEngine::new(engine, None, encryption) {
        Ok(se) => se,
        Err(e) => return err(format!("Sync adapter not available: {e}")),
    };
    let remote = cfg.sync_remote.clone().unwrap_or_default();
    let token = cfg.sync_token.as_deref();
    let no_remote = || err("No remote configured. Set sync_remote in the Mnemosyne config.");

    match name {
        // `_handle_push`: send everything past the per-remote push cursor.
        "mnemosyne_sync_push" => {
            if remote.is_empty() {
                return no_remote();
            }
            let cursor_key = format!("last_push_cursor_{remote}");
            let cursor = se.meta_get(&cursor_key).ok().flatten();
            let changes = match se.pull_changes(cursor.as_deref(), 500) {
                Ok(c) => c,
                Err(e) => return err(e),
            };
            let events = changes["events"].as_array().cloned().unwrap_or_default();
            if events.is_empty() {
                return json!({"status": "ok", "pushed": 0, "message": "No local changes to push."})
                    .to_string();
            }
            let resp =
                SyncEngine::http_post(&remote, "/sync/push", &json!({"events": events}), token)
                    .await;
            if resp["status"] != "ok" {
                return resp.to_string();
            }
            // Advance to the last *local* timestamp actually sent (the server's `next_cursor`
            // is its wall clock — meaningless against this bank's log).
            let next = changes["next_cursor"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            if !next.is_empty() {
                let _ = se.meta_set(&cursor_key, &next);
            }
            json!({
                "status": "ok",
                "pushed": resp["accepted"],
                "duplicates": resp.get("duplicates").cloned().unwrap_or(json!(0)),
                "conflicts": resp.get("conflicts").cloned().unwrap_or(json!(0)),
                "next_cursor": trunc(&next),
            })
            .to_string()
        }
        // `_handle_pull`: fetch since the per-remote pull cursor and apply locally.
        "mnemosyne_sync_pull" => {
            if remote.is_empty() {
                return no_remote();
            }
            let cursor_key = format!("last_sync_cursor_{remote}");
            let cursor = se.meta_get(&cursor_key).ok().flatten();
            let resp =
                SyncEngine::http_post(&remote, "/sync/pull", &json!({"since": cursor}), token)
                    .await;
            if resp["status"] != "ok" {
                return resp.to_string();
            }
            let incoming = resp["events"].as_array().cloned().unwrap_or_default();
            if incoming.is_empty() {
                return json!({"status": "ok", "pulled": 0, "message": "No remote changes to pull."})
                    .to_string();
            }
            let applied = match se.push_changes(&incoming) {
                Ok(stats) => stats,
                Err(e) => return err(e),
            };
            let next = resp["next_cursor"].as_str().unwrap_or_default().to_string();
            if !next.is_empty() {
                let _ = se.meta_set(&cursor_key, &next);
            }
            json!({
                "status": "ok",
                "pulled": applied["accepted"],
                "duplicates": applied["duplicates"],
                "conflicts": applied["conflicts"],
                "next_cursor": trunc(&next),
            })
            .to_string()
        }
        // `_handle_status`: local identity/counters + the configured transport knobs.
        _ => {
            let cursor_key = format!("last_sync_cursor_{remote}");
            let cursor = se.meta_get(&cursor_key).ok().flatten().unwrap_or_default();
            let local_events = match se.get_status(None) {
                Ok(status) => status["total_events"].clone(),
                Err(e) => return err(e),
            };
            json!({
                "status": "ok",
                "device_id": se.device_id,
                "remote": if remote.is_empty() { "(unconfigured)".to_string() } else { remote },
                "encryption": if encrypt_enabled { "enabled" } else { "disabled" },
                "mode": cfg.sync_mode,
                "local_events": local_events,
                "last_cursor": if cursor.is_empty() { "none".to_string() } else { trunc(&cursor) },
            })
            .to_string()
        }
    }
}
