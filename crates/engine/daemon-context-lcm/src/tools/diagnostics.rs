// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `lcm_status` (§10.6, `LCM:tools.py:1532-1651`) and `lcm_doctor` (§10.6,
//! `LCM:tools.py:1654-1935`), plus the metadata-only payload/pattern scanners they share
//! (`LCM:ingest_protection.py:985-1275`).
//!
//! Rust adaptations (documented per-site): the engine has no side channel, no separate provider
//! label, and no env-override preset provenance — those fields report their inert values rather
//! than being dropped, so operators see the same key set as the Python plugin.

use super::ToolCx;
use crate::externalize;
use crate::protection::{
    self, contains_data_uri_base64, contains_long_base64_run, GENERIC_BASE64_MIN_CHARS,
    HEARTBEAT_NOISE_MAX_CHARS, QUARANTINED_ASSISTANT_MIN_CHARS,
};
use crate::store::{RiskField, RiskRow};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::path::Path;

/// The bounded sample size of every payload-risk list (`scan_sqlite_payload_risks(limit=5)`).
const RISK_SAMPLE_LIMIT: usize = 5;

// ---- shared serialization helpers ----------------------------------------------------------------

/// Round to one decimal place (the Python diagnostics' `round(x, 1)`).
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Round to four decimal places (the Python `round(engine.cache_read_ratio, 4)`).
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Serialize a diagnostic struct into a JSON `detail` value (defaulting to `null` on the
/// impossible serialization failure).
fn to_detail<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// Render a float the way Python's `str()` does inside f-strings (`0.0` keeps its decimal).
fn py_float(x: f64) -> String {
    if x == x.trunc() && x.is_finite() {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

// ---- runtime identity / presets ------------------------------------------------------------------

/// `get_runtime_identity` (`LCM:engine.py:2351-2400`), adapted: the daemon engine is a crate, not
/// a plugin checkout, so the plugin/git fields carry the crate identity and the `hermes_home`,
/// `session_platform`, and side-channel fields are omitted (the daemon has neither).
pub(super) fn runtime_identity(cx: &ToolCx<'_>) -> Value {
    let lifecycle = cx.store.get_lifecycle(cx.session_id).ok().flatten();
    json!({
        "engine": "lcm",
        "plugin_name": env!("CARGO_PKG_NAME"),
        "plugin_version": env!("CARGO_PKG_VERSION"),
        "database_path": cx.config.db_path().map(|p| p.display().to_string()).unwrap_or_default(),
        "database_path_source": "config",
        "session_id": cx.session_id,
        "session_bound": !cx.session_id.is_empty(),
        // Conversation identity: the port keys lifecycle rows by the session id itself.
        "conversation_id": cx.session_id,
        "lifecycle_current_session_id": lifecycle
            .as_ref()
            .and_then(|l| l.current_session_id.clone())
            .unwrap_or_default(),
        "lifecycle_last_finalized_session_id": lifecycle
            .as_ref()
            .and_then(|l| l.last_finalized_session_id.clone())
            .unwrap_or_default(),
    })
}

/// `preset_status_payload` (`LCM:presets.py:253-313`), adapted: the daemon has no env-override
/// channel, so the override/provenance/dry-run fields are inert (empty), and the suggested preset
/// carries this port's [`crate::presets::Preset`] fields.
fn preset_status_payload(cx: &ToolCx<'_>) -> Value {
    let context_length = cx.context_length.unwrap_or(0);
    let preset = crate::presets::suggest_preset_for_engine(context_length);
    let reason = match preset {
        Some(p) if p.name == "codex_gpt_long_context" => {
            "context-window match for GPT/Codex candidate; verify provider/model family before applying"
                .to_string()
        }
        Some(_) => {
            "context-window match for GPT/Codex Spark candidate; verify provider/model family before applying"
                .to_string()
        }
        None => format!("no shipped benchmarked preset matches context_length {context_length}"),
    };
    json!({
        "read_only": true,
        "runtime_mutation": false,
        "reason": reason,
        "match_confidence": if preset.is_some() { "context-only" } else { "none" },
        "suggested_preset": preset.map(|p| json!({
            "name": p.name,
            "description": p.description,
            "context_threshold": p.context_threshold,
            "fresh_tail_count": p.fresh_tail_count,
            "leaf_chunk_tokens": p.leaf_chunk_tokens,
        })),
        "provenance": {},
        "explicit_overrides": {},
        "invalid_overrides": {},
        "dry_run_delta": [],
    })
}

/// `sensitive_pattern_status` (`LCM:ingest_protection.py:171-185`): metadata-only redaction
/// posture. `source` carries the config-level provenance (`default` until the host overrides the
/// pattern list — the daemon analog of Python's env-var provenance).
pub(super) fn sensitive_pattern_status(config: &crate::config::LcmConfig) -> Value {
    let (configured, active, unknown) = configured_sensitive_pattern_names(config);
    let enabled = config.sensitive_patterns_enabled;
    let lossless_recovery: Value = if enabled && !active.is_empty() {
        json!(false)
    } else {
        Value::Null
    };
    json!({
        "sensitive_patterns_enabled": enabled,
        "enabled": enabled,
        "sensitive_patterns": configured,
        "patterns": configured,
        "active_patterns": if enabled { active } else { Vec::new() },
        "unknown_patterns": unknown,
        "source": config.sensitive_patterns_source,
        "placeholder_format": "[LCM sensitive redaction: name=<pattern>; chars=<n>; bytes=<n>; sha256=<16 for non-password>]",
        "lossless_recovery": lossless_recovery,
    })
}

/// `_configured_sensitive_pattern_names` (`LCM:ingest_protection.py:188-214`):
/// `(configured, active, unknown)` — `all`/`default` expand to the full catalog.
fn configured_sensitive_pattern_names(
    config: &crate::config::LcmConfig,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    const CATALOG: [&str; 4] = [
        "api_key",
        "bearer_token",
        "password_assignment",
        "private_key",
    ];
    let names: Vec<String> = config
        .sensitive_patterns
        .iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    let mut configured: Vec<String> = Vec::new();
    let mut active: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    for name in names {
        let normalized = name.to_lowercase().trim().to_string();
        if normalized == "all" || normalized == "default" {
            for catalog_name in CATALOG {
                if !configured.iter().any(|c| c == catalog_name) {
                    configured.push(catalog_name.to_string());
                }
                if !active.iter().any(|a| a == catalog_name) {
                    active.push(catalog_name.to_string());
                }
            }
            continue;
        }
        configured.push(normalized.clone());
        if protection::is_known_sensitive_pattern(&normalized) {
            if !active.contains(&normalized) {
                active.push(normalized);
            }
        } else if !unknown.contains(&normalized) {
            unknown.push(normalized);
        }
    }
    (configured, active, unknown)
}

// ---- payload-risk / externalized-payload scanners ------------------------------------------------

/// `make_row` (`LCM:ingest_protection.py:1052-1069`): one bounded, metadata-only risk sample.
fn make_risk_row(row: &RiskRow, field: &str, length_key: &str, category: &str) -> Value {
    let value = row.value.as_deref().unwrap_or("");
    let mut m = Map::new();
    m.insert("store_id".into(), json!(row.store_id));
    m.insert("session_id".into(), json!(row.session_id));
    m.insert("source".into(), json!(row.source));
    m.insert("role".into(), json!(row.role));
    m.insert("field".into(), json!(field));
    m.insert("length".into(), json!(row.length));
    m.insert(length_key.into(), json!(row.length));
    m.insert("suspicious_category".into(), json!(category));
    let refs = externalize::extract_ingest_refs(value);
    let reference = refs
        .first()
        .cloned()
        .or_else(|| externalize::extract_ref(value));
    if let Some(reference) = reference {
        m.insert("externalized_ref".into(), json!(reference));
    }
    Value::Object(m)
}

/// `scan_sqlite_payload_risks` (`LCM:ingest_protection.py:1044-1242`): bounded diagnostics for
/// suspicious inline payload storage; never returns raw payload text.
fn scan_sqlite_payload_risks(cx: &ToolCx<'_>) -> Value {
    let limit = RISK_SAMPLE_LIMIT;
    let candidate_cap = ((limit * 20).max(limit)) as i64;
    let store = cx.store;

    let largest_content: Vec<Value> = store
        .largest_field_rows(RiskField::Content, limit as i64)
        .unwrap_or_default()
        .iter()
        .map(|r| make_risk_row(r, "content", "content_len", "largest_content"))
        .collect();
    let largest_tool_calls: Vec<Value> = store
        .largest_field_rows(RiskField::ToolCalls, limit as i64)
        .unwrap_or_default()
        .iter()
        .map(|r| make_risk_row(r, "tool_calls", "tool_calls_len", "largest_tool_calls"))
        .collect();

    // Broad SQL pre-filter, then the conservative regex classifier — avoids false positives from
    // rows that merely quote a `data:%;base64,%` scaffold.
    let data_uri_content: Vec<Value> = store
        .data_uri_candidate_rows(RiskField::Content, candidate_cap)
        .unwrap_or_default()
        .iter()
        .filter(|r| r.value.as_deref().is_some_and(contains_data_uri_base64))
        .take(limit)
        .map(|r| make_risk_row(r, "content", "content_len", "data_uri_base64"))
        .collect();
    let data_uri_tool_calls: Vec<Value> = store
        .data_uri_candidate_rows(RiskField::ToolCalls, candidate_cap)
        .unwrap_or_default()
        .iter()
        .filter(|r| r.value.as_deref().is_some_and(contains_data_uri_base64))
        .take(limit)
        .map(|r| make_risk_row(r, "tool_calls", "tool_calls_len", "data_uri_base64"))
        .collect();

    let mut generic_rows: Vec<Value> = Vec::new();
    for row in store
        .long_payload_rows(GENERIC_BASE64_MIN_CHARS as i64, candidate_cap)
        .unwrap_or_default()
    {
        for (field, value) in [("content", &row.content), ("tool_calls", &row.tool_calls)] {
            let Some(value) = value.as_deref() else {
                continue;
            };
            if contains_long_base64_run(value) {
                let mut m = Map::new();
                m.insert("store_id".into(), json!(row.store_id));
                m.insert("session_id".into(), json!(row.session_id));
                m.insert("source".into(), json!(row.source));
                m.insert("role".into(), json!(row.role));
                m.insert("field".into(), json!(field));
                m.insert("length".into(), json!(value.chars().count()));
                m.insert("suspicious_category".into(), json!("base64_like"));
                let refs = externalize::extract_ingest_refs(value);
                let reference = refs
                    .first()
                    .cloned()
                    .or_else(|| externalize::extract_ref(value));
                if let Some(reference) = reference {
                    m.insert("externalized_ref".into(), json!(reference));
                }
                generic_rows.push(Value::Object(m));
                break;
            }
        }
        if generic_rows.len() >= limit {
            break;
        }
    }

    let quarantined_assistant: Vec<Value> = store
        .quarantined_assistant_rows(limit as i64)
        .unwrap_or_default()
        .iter()
        .map(|r| make_risk_row(r, "content", "content_len", "quarantined_assistant_output"))
        .collect();

    let mut repetitive_assistant: Vec<Value> = Vec::new();
    for row in store
        .repetitive_assistant_candidate_rows(QUARANTINED_ASSISTANT_MIN_CHARS as i64, candidate_cap)
        .unwrap_or_default()
    {
        if let Some(value) = row.value.as_deref() {
            if let Some(reason) = protection::assistant_output_quarantine_reason(value) {
                repetitive_assistant.push(make_risk_row(&row, "content", "content_len", &reason));
            }
        }
        if repetitive_assistant.len() >= limit {
            break;
        }
    }

    let mut heartbeat_noise: Vec<Value> = Vec::new();
    for row in store
        .heartbeat_candidate_rows(HEARTBEAT_NOISE_MAX_CHARS as i64, candidate_cap)
        .unwrap_or_default()
    {
        if let Some(reason) = protection::heartbeat_noise_reason(row.value.as_deref().unwrap_or(""))
        {
            heartbeat_noise.push(make_risk_row(&row, "content", "content_len", &reason));
        }
        if heartbeat_noise.len() >= limit {
            break;
        }
    }

    json!({
        "largest_content_rows": largest_content,
        "largest_tool_calls_rows": largest_tool_calls,
        "suspicious_data_uri_content_rows": data_uri_content,
        "suspicious_data_uri_tool_calls_rows": data_uri_tool_calls,
        "suspicious_base64_like_rows": generic_rows,
        "quarantined_assistant_rows": quarantined_assistant,
        "suspicious_repetitive_assistant_rows": repetitive_assistant,
        "heartbeat_noise_rows": heartbeat_noise,
    })
}

/// `externalized_payload_stats` (`LCM:ingest_protection.py:1244-1275`): directory-level metadata
/// for the externalization side channel. An ephemeral bank reports an empty dir with zero counts.
fn externalized_payload_stats(cx: &ToolCx<'_>) -> Value {
    let dir = cx.config.externalization_dir();
    let mut count = 0u64;
    let mut total_bytes = 0u64;
    let mut total_chars = 0u64;
    let mut latest_path = String::new();
    let mut latest_mtime = 0.0f64;
    if let Some(dir) = &dir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") || !path.is_file() {
                    continue;
                }
                count += 1;
                if let Ok(meta) = entry.metadata() {
                    total_bytes += meta.len();
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs_f64())
                        .unwrap_or(0.0);
                    if mtime > latest_mtime {
                        latest_mtime = mtime;
                        latest_path = path.display().to_string();
                    }
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(record) = externalize::read_payload_record(dir, &name) {
                    let chars = record
                        .get("content_chars")
                        .or_else(|| record.get("chars"))
                        .and_then(Value::as_u64)
                        .unwrap_or_else(|| {
                            record
                                .get("content")
                                .and_then(Value::as_str)
                                .map(|c| c.chars().count() as u64)
                                .unwrap_or(0)
                        });
                    total_chars += chars;
                }
            }
        }
    }
    json!({
        "externalized_payload_dir": dir.map(|d| d.display().to_string()).unwrap_or_default(),
        "externalized_payload_count": count,
        "externalized_payload_bytes": total_bytes,
        "externalized_payload_chars": total_chars,
        "latest_externalized_payload_path": latest_path,
        "latest_externalized_payload_mtime": latest_mtime,
    })
}

/// `scan_externalized_payload_integrity` (`LCM:ingest_protection.py:985-1041`): compare payload
/// refs stored in `messages` against the JSON files on disk. Metadata-only.
fn scan_externalized_payload_integrity(cx: &ToolCx<'_>) -> Value {
    let dir = cx.config.externalization_dir();
    let existing_files: BTreeSet<String> = dir
        .as_ref()
        .and_then(|d| std::fs::read_dir(d).ok())
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.path().is_file())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.ends_with(".json").then_some(name)
                })
                .collect()
        })
        .unwrap_or_default();

    let mut referenced_refs: BTreeSet<String> = BTreeSet::new();
    let mut first_location_by_ref: std::collections::HashMap<String, Value> =
        std::collections::HashMap::new();
    for row in cx.store.externalized_ref_rows().unwrap_or_default() {
        for (field, value) in [("content", &row.content), ("tool_calls", &row.tool_calls)] {
            let Some(value) = value.as_deref() else {
                continue;
            };
            for reference in externalize::extract_all_refs(value) {
                referenced_refs.insert(reference.clone());
                first_location_by_ref
                    .entry(reference.clone())
                    .or_insert_with(|| {
                        json!({
                            "store_id": row.store_id,
                            "session_id": row.session_id,
                            "source": row.source,
                            "role": row.role,
                            "field": field,
                            "externalized_ref": reference,
                        })
                    });
            }
        }
    }

    let missing_refs: Vec<&String> = referenced_refs
        .iter()
        .filter(|r| !existing_files.contains(*r))
        .collect();
    let existing_ref_count = referenced_refs.len() - missing_refs.len();
    let unreferenced_files: Vec<&String> = existing_files
        .iter()
        .filter(|f| !referenced_refs.contains(*f))
        .collect();

    json!({
        "externalized_payload_refs_total": referenced_refs.len(),
        "externalized_payload_refs_existing": existing_ref_count,
        "externalized_payload_refs_missing": missing_refs.len(),
        "externalized_payload_files_unreferenced": unreferenced_files.len(),
        "missing_externalized_payload_refs": missing_refs
            .iter()
            .take(RISK_SAMPLE_LIMIT)
            .filter_map(|r| first_location_by_ref.get(*r).cloned())
            .collect::<Vec<_>>(),
        "unreferenced_externalized_payload_files": unreferenced_files
            .iter()
            .take(RISK_SAMPLE_LIMIT)
            .map(|r| json!({"externalized_ref": r}))
            .collect::<Vec<_>>(),
    })
}

// ---- 10.6 lcm_status ------------------------------------------------------------------------------

pub(super) fn status(cx: &ToolCx<'_>) -> String {
    let session_id = cx.session_id;
    if session_id.is_empty() {
        return json!({
            "error": "No active session",
            "runtime_identity": runtime_identity(cx),
        })
        .to_string();
    }

    let store_messages = cx.store.message_count(session_id).unwrap_or(0);
    let store_tokens = cx.store.get_session_token_total(session_id).unwrap_or(0);

    let all_nodes = cx
        .store
        .get_session_nodes(session_id, None, i64::MAX)
        .unwrap_or_default();
    let mut depths: std::collections::BTreeMap<i64, (i64, i64, i64)> =
        std::collections::BTreeMap::new();
    for node in &all_nodes {
        let entry = depths.entry(node.depth).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += node.token_count;
        entry.2 += node.source_token_count;
    }
    let total_dag_tokens: i64 = depths.values().map(|d| d.1).sum();
    let total_source_tokens: i64 = depths.values().map(|d| d.2).sum();
    let compression_ratio = if total_dag_tokens > 0 {
        format!(
            "{}:1",
            py_float(round1(total_source_tokens as f64 / total_dag_tokens as f64))
        )
    } else {
        "0:1".to_string()
    };
    let mut depths_obj = Map::new();
    for (depth, (count, tokens, source_tokens)) in &depths {
        depths_obj.insert(
            format!("d{depth}"),
            json!({"count": count, "tokens": tokens, "source_tokens": source_tokens}),
        );
    }

    let source_lineage = match cx.store.source_stats(Some(session_id)) {
        Ok(stats) => to_detail(&stats),
        Err(e) => json!({"error": e.to_string()}),
    };
    let lifecycle = cx
        .store
        .get_lifecycle(session_id)
        .ok()
        .flatten()
        .map(|row| to_detail(&row))
        .unwrap_or(Value::Null);
    let lifecycle_fragmentation = match cx.store.lifecycle_fragmentation_stats() {
        Ok(frag) => to_detail(&frag),
        Err(e) => json!({"error": e.to_string(), "read_only": true}),
    };

    json!({
        "session_id": session_id,
        "compression_count": cx.compaction_count,
        "last_compression_status": cx.last_compression_status,
        "last_compression_noop_reason": cx.last_compression_noop_reason,
        "model": cx.model,
        // The daemon injects one aux provider without a provider label (`LCM:engine.py` reads it
        // from the gateway); reported empty.
        "provider": "",
        "context_length": cx.context_length.unwrap_or(0),
        "context_length_source": cx.context_length_source,
        "context_threshold": cx.config.context_threshold,
        "threshold_tokens": cx.threshold_tokens.unwrap_or(0),
        "last_prompt_tokens": cx.last_prompt_tokens,
        "last_input_tokens": cx.usage.last_input_tokens,
        "last_output_tokens": cx.usage.last_output_tokens,
        "last_cache_read_tokens": cx.usage.last_cache_read_tokens,
        "last_cache_write_tokens": cx.usage.last_cache_write_tokens,
        "last_reasoning_tokens": cx.usage.last_reasoning_tokens,
        "cache_metrics_available": cx.usage.cache_metrics_available,
        "cache_read_ratio": round4(cx.usage.cache_read_ratio()),
        "store": {
            "messages": store_messages,
            "estimated_tokens": store_tokens,
            // Low-disk degradation: FTS was dropped on a full-disk rebuild; search is LIKE-only
            // until a later repair pass (reopen / `/lcm doctor repair apply`) rebuilds cleanly.
            "fts_degraded": cx.store.is_degraded(),
        },
        "dag": {
            "total_nodes": all_nodes.len(),
            "total_tokens": total_dag_tokens,
            "compression_ratio": compression_ratio,
            "depths": depths_obj,
        },
        // Mirrors the Python config block (`LCM:tools.py:1610-1626`) field-for-field.
        "config": {
            "fresh_tail_count": cx.config.fresh_tail_count,
            "leaf_chunk_tokens": cx.config.leaf_chunk_tokens,
            "dynamic_leaf_chunk_enabled": cx.config.dynamic_leaf_chunk_enabled,
            "dynamic_leaf_chunk_max": cx.config.dynamic_leaf_chunk_max,
            "cache_friendly_condensation_enabled": cx.config.cache_friendly_condensation_enabled,
            "cache_friendly_min_debt_groups": cx.config.cache_friendly_min_debt_groups,
            "deferred_maintenance_enabled": cx.config.deferred_maintenance_enabled,
            "deferred_maintenance_max_passes": cx.config.deferred_maintenance_max_passes,
            "critical_budget_pressure_ratio": cx.config.critical_budget_pressure_ratio,
            "context_threshold": cx.config.context_threshold,
            "max_depth": cx.config.incremental_max_depth,
            "condensation_fanin": cx.config.condensation_fanin,
            "summary_model": if cx.config.summary_model.is_empty() { "(auxiliary)" } else { &cx.config.summary_model },
            "summary_timeout_ms": cx.config.summary_timeout_ms,
            "expansion_model": if cx.config.expansion_model.is_empty() { "(summary model)" } else { &cx.config.expansion_model },
        },
        "session_filters": {
            "ignored": cx.session_ignored,
            "stateless": cx.session_stateless,
            "ignore_session_patterns": cx.config.ignore_session_patterns,
            "ignore_session_patterns_source": cx.config.ignore_session_patterns_source,
            "stateless_session_patterns": cx.config.stateless_session_patterns,
            "stateless_session_patterns_source": cx.config.stateless_session_patterns_source,
            "ignore_message_patterns": cx.config.ignore_message_patterns,
            "ignore_message_patterns_source": cx.config.ignore_message_patterns_source,
            "ignored_message_count": cx.ignored_message_count,
            // The daemon runs one engine per foreground session — no cron side channel exists.
            "side_channel_active": false,
        },
        "source_lineage": source_lineage,
        "ingest_protection": sensitive_pattern_status(cx.config),
        "preset_suggestion": preset_status_payload(cx),
        // The Rust ingest reconciles deterministically per incarnation and keeps no report — the
        // Python field starts as `{}` too (`LCM:engine.py:_last_ingest_reconciliation`).
        "ingest_reconciliation": {},
        "runtime_identity": runtime_identity(cx),
        "lifecycle": lifecycle,
        "lifecycle_fragmentation": lifecycle_fragmentation,
    })
    .to_string()
}

// ---- 10.6 lcm_doctor ------------------------------------------------------------------------------

/// Append one check row and fold its status into the running worst level.
fn push_check(checks: &mut Vec<Value>, name: &str, status: &str, detail: Value) {
    checks.push(json!({"check": name, "status": status, "detail": detail}));
}

pub(super) fn doctor(cx: &ToolCx<'_>) -> String {
    let mut checks: Vec<Value> = Vec::new();
    let session_id = cx.session_id;

    // 1. Database integrity.
    match cx.store.integrity_check() {
        Ok(result) => push_check(
            &mut checks,
            "database_integrity",
            if result == "ok" { "pass" } else { "fail" },
            json!(result),
        ),
        Err(e) => push_check(
            &mut checks,
            "database_integrity",
            "fail",
            json!(e.to_string()),
        ),
    }

    // Schema core tables.
    match cx.store.schema_health() {
        Ok(schema) => push_check(
            &mut checks,
            "schema_core_tables",
            if schema.missing.is_empty() {
                "pass"
            } else {
                "fail"
            },
            to_detail(&schema),
        ),
        Err(e) => push_check(
            &mut checks,
            "schema_core_tables",
            "fail",
            json!(e.to_string()),
        ),
    }

    // 1b. FTS5 deep integrity, per index (malformed inverted indexes point at the exact table).
    match cx.store.fts_integrity() {
        Ok(reports) => {
            for (name, report) in ["messages_fts_integrity", "nodes_fts_integrity"]
                .iter()
                .zip(reports.iter())
            {
                let status = if report.status == "unchecked" {
                    "warn"
                } else {
                    report.status.as_str()
                };
                push_check(&mut checks, name, status, json!(report.detail));
            }
        }
        Err(e) => {
            for name in ["messages_fts_integrity", "nodes_fts_integrity"] {
                push_check(&mut checks, name, "fail", json!(e.to_string()));
            }
        }
    }

    // 2. SQLite storage posture and payload diagnostics.
    match cx.store.storage_posture() {
        Ok(posture) => {
            let db_path = Path::new(&posture.database_path);
            let database_exists = !posture.database_path.is_empty() && db_path.exists();
            let database_size_bytes = if database_exists {
                std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0)
            } else {
                0
            };
            let wal_size_bytes = if posture.database_path.is_empty() {
                0
            } else {
                std::fs::metadata(format!("{}-wal", posture.database_path))
                    .map(|m| m.len())
                    .unwrap_or(0)
            };
            push_check(
                &mut checks,
                "sqlite_storage",
                if posture.quick_check == "ok" {
                    "pass"
                } else {
                    "fail"
                },
                json!({
                    "database_path": posture.database_path,
                    "database_exists": database_exists,
                    "journal_mode": posture.journal_mode,
                    "quick_check": posture.quick_check,
                    "database_size_bytes": database_size_bytes,
                    "wal_size_bytes": wal_size_bytes,
                    "in_memory": posture.in_memory,
                }),
            );
        }
        Err(e) => push_check(&mut checks, "sqlite_storage", "fail", json!(e.to_string())),
    }

    // payload_storage: risks + directory stats + ref/file integrity, merged like Python's
    // `{**payload_risks, **externalized_stats, **externalized_integrity}`.
    {
        let payload_risks = scan_sqlite_payload_risks(cx);
        let externalized_stats = externalized_payload_stats(cx);
        let externalized_integrity = scan_externalized_payload_integrity(cx);
        let suspicious_count = [
            "suspicious_data_uri_content_rows",
            "suspicious_data_uri_tool_calls_rows",
            "suspicious_base64_like_rows",
            "suspicious_repetitive_assistant_rows",
            "heartbeat_noise_rows",
        ]
        .iter()
        .map(|key| {
            payload_risks
                .get(key)
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0)
        })
        .sum::<usize>();
        let missing_refs = externalized_integrity
            .get("externalized_payload_refs_missing")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let mut detail = Map::new();
        for part in [payload_risks, externalized_stats, externalized_integrity] {
            if let Value::Object(map) = part {
                detail.extend(map);
            }
        }
        push_check(
            &mut checks,
            "payload_storage",
            if suspicious_count > 0 || missing_refs > 0 {
                "warn"
            } else {
                "pass"
            },
            Value::Object(detail),
        );
    }

    // Sensitive pattern handling.
    {
        let protection_status = sensitive_pattern_status(cx.config);
        let enabled = protection_status["enabled"].as_bool().unwrap_or(false);
        let active_empty = protection_status["active_patterns"]
            .as_array()
            .is_none_or(Vec::is_empty);
        let unknown_nonempty = protection_status["unknown_patterns"]
            .as_array()
            .is_some_and(|a| !a.is_empty());
        let status = if (enabled && active_empty) || unknown_nonempty {
            "warn"
        } else {
            "pass"
        };
        push_check(
            &mut checks,
            "sensitive_pattern_handling",
            status,
            protection_status,
        );
    }

    // 3. FTS index sync (session-scoped, like the Python check).
    match cx.store.session_fts_sync_counts(session_id) {
        Ok((msg_count, fts_count)) => push_check(
            &mut checks,
            "fts_index_sync",
            if fts_count >= msg_count {
                "pass"
            } else {
                "warn"
            },
            json!(format!(
                "{fts_count} session FTS rows, {msg_count} session messages"
            )),
        ),
        Err(e) => push_check(&mut checks, "fts_index_sync", "fail", json!(e.to_string())),
    }

    // 3b. Orphaned DAG nodes (D0 nodes referencing missing store rows).
    match cx.store.orphaned_session_node_count(session_id) {
        Ok(orphaned) => push_check(
            &mut checks,
            "orphaned_dag_nodes",
            if orphaned == 0 { "pass" } else { "warn" },
            if orphaned > 0 {
                json!(format!("{orphaned} nodes reference missing store messages"))
            } else {
                json!("all nodes have valid sources")
            },
        ),
        Err(e) => push_check(
            &mut checks,
            "orphaned_dag_nodes",
            "fail",
            json!(e.to_string()),
        ),
    }

    // Summary quality.
    match cx.store.summary_quality_stats(session_id) {
        Ok(sq) => {
            let degraded = sq.extreme_ratio_nodes + sq.tiny_large_source_nodes;
            let mut detail = to_detail(&sq);
            if let Value::Object(ref mut map) = detail {
                map.insert(
                    "tiny_large_source_threshold".to_string(),
                    json!({"source_token_count_min": 100_000, "token_count_max": 500}),
                );
                map.insert(
                    "recommendation".to_string(),
                    json!(if degraded > 0 {
                        "Inspect worst_nodes with lcm_expand; tiny summaries for very large sources often indicate degraded fallback summarization."
                    } else {
                        "summary compression ratios are within the diagnostic thresholds"
                    }),
                );
            }
            push_check(
                &mut checks,
                "summary_quality",
                if degraded > 0 { "warn" } else { "pass" },
                detail,
            );
        }
        Err(e) => push_check(&mut checks, "summary_quality", "fail", json!(e.to_string())),
    }

    // 4. Configuration validation.
    {
        let c = cx.config;
        let mut warnings: Vec<String> = Vec::new();
        if c.fresh_tail_count < 2 {
            warnings.push("fresh_tail_count < 2 may cause aggressive compaction".to_string());
        }
        if c.context_threshold > 0.95 {
            warnings.push("context_threshold > 0.95 leaves very little headroom".to_string());
        }
        if c.context_threshold < 0.3 {
            warnings.push("context_threshold < 0.3 triggers compaction very early".to_string());
        }
        if c.condensation_fanin < 2 {
            warnings.push("condensation_fanin < 2 creates excessive depth growth".to_string());
        }
        if c.incremental_max_depth == 0 {
            warnings.push("incremental_max_depth=0 disables condensation entirely".to_string());
        }
        push_check(
            &mut checks,
            "config_validation",
            if warnings.is_empty() { "pass" } else { "warn" },
            if warnings.is_empty() {
                json!("all settings within normal ranges")
            } else {
                json!(warnings)
            },
        );
    }

    // 5. Source-lineage hygiene (detail-only; bank-wide like Python's `get_source_stats()`).
    match cx.store.source_stats(None) {
        Ok(stats) => {
            let mut detail = to_detail(&stats);
            if let Value::Object(ref mut map) = detail {
                map.insert(
                    "normalization_mode".to_string(),
                    json!("backcompat-normalization"),
                );
            }
            push_check(&mut checks, "source_lineage_hygiene", "pass", detail);
        }
        Err(e) => push_check(
            &mut checks,
            "source_lineage_hygiene",
            "fail",
            json!(e.to_string()),
        ),
    }

    // 6. Lifecycle/session fragmentation.
    match cx.store.lifecycle_fragmentation_stats() {
        Ok(frag) => push_check(
            &mut checks,
            "lifecycle_fragmentation",
            if frag.is_fragmented() { "warn" } else { "pass" },
            to_detail(&frag),
        ),
        Err(e) => push_check(
            &mut checks,
            "lifecycle_fragmentation",
            "fail",
            json!(e.to_string()),
        ),
    }

    // 7. Context pressure (only when the model window is known).
    if let Some(context_length) = cx.context_length.filter(|n| *n > 0) {
        let usage_pct = round1(cx.last_prompt_tokens as f64 / context_length as f64 * 100.0);
        let threshold_pct = round1(cx.config.context_threshold * 100.0);
        push_check(
            &mut checks,
            "context_pressure",
            if usage_pct < threshold_pct {
                "pass"
            } else {
                "warn"
            },
            json!(format!(
                "{}% used, compaction triggers at {}%",
                py_float(usage_pct),
                py_float(threshold_pct)
            )),
        );
    }

    let mut overall = "healthy";
    if checks.iter().any(|c| c["status"] == "fail") {
        overall = "unhealthy";
    } else if checks.iter().any(|c| c["status"] == "warn") {
        overall = "warnings";
    }

    json!({
        "overall": overall,
        "runtime_identity": runtime_identity(cx),
        "checks": checks,
    })
    .to_string()
}
