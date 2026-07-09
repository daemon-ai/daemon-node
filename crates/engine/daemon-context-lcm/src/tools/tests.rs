// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Contract tests for the seven `lcm_*` tools — exercised through [`dispatch`] over a seeded store.

use super::*;
use crate::store::{NewMessage, NewNode, SourceType, Store};
use crate::tokens::Tokenizer;
use daemon_core::ScriptedProvider;
use serde_json::Value;

struct Fixture {
    store: Store,
    tokenizer: Tokenizer,
    aux: ScriptedProvider,
}

impl Fixture {
    fn new(aux_reply: &str) -> Self {
        let store = Store::open_in_memory().unwrap();
        // Two raw messages + a D0 summary node referencing them, in session "s1".
        let ids = store
            .append_batch(
                "s1",
                &[
                    NewMessage {
                        role: "user".into(),
                        content: Some("deploy the blue green rollout to production".into()),
                        ..Default::default()
                    },
                    NewMessage {
                        role: "assistant".into(),
                        content: Some("acknowledged: rolling out blue green now".into()),
                        ..Default::default()
                    },
                ],
                1_000.0,
            )
            .unwrap();
        store
            .add_node(&NewNode {
                session_id: "s1".into(),
                depth: 0,
                summary: "Summary: discussed a blue/green production rollout.".into(),
                token_count: 9,
                source_token_count: 40,
                source_ids: ids,
                source_type: SourceType::Messages,
                created_at: 1_000.0,
                earliest_at: Some(1_000.0),
                latest_at: Some(1_000.0),
                expand_hint: "Expand for details about: rollout".into(),
            })
            .unwrap();
        // A second session row, to prove scoping.
        store
            .append_batch(
                "s2",
                &[NewMessage {
                    role: "user".into(),
                    content: Some("unrelated lunch plans".into()),
                    ..Default::default()
                }],
                2_000.0,
            )
            .unwrap();
        Self {
            store,
            tokenizer: Tokenizer::heuristic(),
            aux: ScriptedProvider::new(Vec::new(), aux_reply.to_string()),
        }
    }

    fn cx(&self) -> ToolCx<'_> {
        self.cx_with(&CONFIG)
    }

    fn cx_with<'a>(&'a self, config: &'a LcmConfig) -> ToolCx<'a> {
        ToolCx {
            store: &self.store,
            config,
            tokenizer: &self.tokenizer,
            aux: &self.aux,
            session_id: "s1",
            threshold_tokens: Some(350),
            context_length: Some(200_000),
            last_prompt_tokens: 0,
            compaction_count: 2,
            session_ignored: false,
            session_stateless: false,
            ignored_message_count: 0,
            usage: crate::provider::UsageMetrics::default(),
            model: "test-model",
            context_length_source: "model_info",
            last_compression_status: "idle",
            last_compression_noop_reason: "",
            ingest_reconciliation: &Value::Null,
        }
    }
}

use crate::config::LcmConfig;
use std::sync::LazyLock;
static CONFIG: LazyLock<LcmConfig> = LazyLock::new(LcmConfig::in_memory);

async fn call(fx: &Fixture, name: &str, args: Value) -> Value {
    let s = dispatch(&fx.cx(), name, args).await;
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("tool {name} returned non-JSON: {e}: {s}"))
}

async fn call_with(fx: &Fixture, config: &LcmConfig, name: &str, args: Value) -> Value {
    let s = dispatch(&fx.cx_with(config), name, args).await;
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("tool {name} returned non-JSON: {e}: {s}"))
}

// ---- lcm_grep -------------------------------------------------------------------------------------

#[tokio::test]
async fn grep_returns_message_and_summary_hits_for_current_session() {
    let fx = Fixture::new("");
    let out = call(
        &fx,
        "lcm_grep",
        json!({"query": "blue green", "sort": "recency"}),
    )
    .await;
    assert_eq!(out["session_scope"], "current");
    let results = out["results"].as_array().unwrap();
    assert!(results.iter().any(|r| r["type"] == "message"));
    assert!(results.iter().any(|r| r["type"] == "summary"));
    assert!(results.iter().all(|r| r["from_current_session"] == true));
    // The omission marker only appears when a raw filter suppressed summaries.
    assert!(out.get("summary_results_omitted").is_none());
}

#[tokio::test]
async fn grep_omits_summaries_when_a_raw_filter_is_set() {
    let fx = Fixture::new("");
    let out = call(
        &fx,
        "lcm_grep",
        json!({"query": "blue green", "role": "user"}),
    )
    .await;
    let results = out["results"].as_array().unwrap();
    assert!(results.iter().all(|r| r["type"] == "message"));
    assert!(results.iter().all(|r| r["role"] == "user"));
    assert_eq!(out["summary_results_omitted"], true);
    assert_eq!(out["role"], "user");
}

#[tokio::test]
async fn grep_all_scope_is_raw_only_across_sessions() {
    let fx = Fixture::new("");
    let out = call(
        &fx,
        "lcm_grep",
        json!({"query": "lunch", "session_scope": "all"}),
    )
    .await;
    let results = out["results"].as_array().unwrap();
    assert_eq!(out["session_scope"], "all");
    assert!(results.iter().any(|r| r["session_id"] == "s2"));
    assert!(results.iter().all(|r| r["type"] == "message"));
    assert!(results
        .iter()
        .filter(|r| r["session_id"] == "s2")
        .all(|r| r["from_current_session"] == false));
}

#[tokio::test]
async fn grep_requires_session_id_for_session_scope() {
    let fx = Fixture::new("");
    let out = call(
        &fx,
        "lcm_grep",
        json!({"query": "x", "session_scope": "session"}),
    )
    .await;
    assert_eq!(out["error"], "session_scope=session requires session_id");
}

#[tokio::test]
async fn grep_rejects_naive_iso_time_and_accepts_aware_iso_time() {
    let fx = Fixture::new("");
    let naive = call(
        &fx,
        "lcm_grep",
        json!({"query": "blue", "time_from": "2026-01-01T00:00:00"}),
    )
    .await;
    assert_eq!(
        naive["error"],
        "time_from ISO timestamp must include a timezone offset or Z"
    );

    let aware = call(
        &fx,
        "lcm_grep",
        json!({"query": "blue", "time_from": "1970-01-01T00:00:01Z"}),
    )
    .await;
    assert_eq!(aware["time_from"], 1.0);
    assert_eq!(aware["summary_results_omitted"], true);
}

#[tokio::test]
async fn grep_clamps_oversized_limits_and_reports_the_original() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_grep", json!({"query": "blue", "limit": 500})).await;
    assert_eq!(out["limit"], 200);
    assert_eq!(out["limit_clamped_from"], 500);
}

#[tokio::test]
async fn grep_preserves_current_scope_for_unknown_session_scope() {
    let fx = Fixture::new("");
    let out = call(
        &fx,
        "lcm_grep",
        json!({"query": "blue", "session_scope": "everything"}),
    )
    .await;
    assert_eq!(out["session_scope"], "current");
    assert_eq!(out["ignored_session_scope"], "everything");
    assert!(out["scope_note"].as_str().unwrap().contains("Unsupported"));
}

// ---- lcm_load_session -----------------------------------------------------------------------------

#[tokio::test]
async fn load_session_pages_with_cursor_and_char_truncation() {
    let fx = Fixture::new("");
    let out = call(
        &fx,
        "lcm_load_session",
        json!({"session_id": "s1", "limit": 1, "max_content_chars": 5}),
    )
    .await;
    assert_eq!(out["has_more"], true);
    assert_eq!(out["total_messages"], 2);
    assert_eq!(out["returned_messages"], 1);
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["content_truncated"], true);
    assert_eq!(msgs[0]["next_content_offset"], 5);
    assert_eq!(msgs[0]["from_current_session"], true);
    // The cursor advances; the next page returns the second row.
    let cursor = out["next_cursor"].clone();
    let page2 = call(
        &fx,
        "lcm_load_session",
        json!({"session_id": "s1", "limit": 10, "after_store_id": cursor}),
    )
    .await;
    assert_eq!(page2["has_more"], false);
    assert!(page2["next_cursor"].is_null());
    assert_eq!(
        page2["messages"].as_array().unwrap()[0]["role"],
        "assistant"
    );
}

#[tokio::test]
async fn load_session_filters_by_roles_and_time_window() {
    let fx = Fixture::new("");
    let out = call(
        &fx,
        "lcm_load_session",
        json!({"session_id": "s1", "roles": ["assistant"]}),
    )
    .await;
    assert_eq!(out["total_messages"], 1);
    assert_eq!(out["roles"], json!(["assistant"]));
    let msgs = out["messages"].as_array().unwrap();
    assert!(msgs.iter().all(|m| m["role"] == "assistant"));

    let windowed = call(
        &fx,
        "lcm_load_session",
        json!({"session_id": "s1", "time_from": 5000.0}),
    )
    .await;
    assert_eq!(windowed["total_messages"], 0);
    assert_eq!(windowed["returned_messages"], 0);
}

#[tokio::test]
async fn load_session_rejects_bad_arguments_with_python_error_text() {
    let fx = Fixture::new("");
    let missing = call(&fx, "lcm_load_session", json!({})).await;
    assert_eq!(missing["error"], "session_id is required");

    let bad_limit = call(
        &fx,
        "lcm_load_session",
        json!({"session_id": "s1", "limit": "many"}),
    )
    .await;
    assert_eq!(bad_limit["error"], "limit must be an integer");

    let bad_roles = call(
        &fx,
        "lcm_load_session",
        json!({"session_id": "s1", "roles": ["user", ""]}),
    )
    .await;
    assert_eq!(
        bad_roles["error"],
        "roles must contain only non-empty strings"
    );

    let bad_window = call(
        &fx,
        "lcm_load_session",
        json!({"session_id": "s1", "time_from": 10, "time_to": 1}),
    )
    .await;
    assert_eq!(
        bad_window["error"],
        "time_to must be greater than or equal to time_from"
    );
}

// ---- lcm_describe ---------------------------------------------------------------------------------

#[tokio::test]
async fn describe_overview_and_node_subtree() {
    let fx = Fixture::new("");
    let overview = call(&fx, "lcm_describe", json!({})).await;
    assert_eq!(overview["session_id"], "s1");
    assert_eq!(overview["store_message_count"], 2);
    assert_eq!(overview["depths"]["d0"]["count"], 1);
    assert_eq!(
        overview["depths"]["d0"]["nodes"].as_array().unwrap().len(),
        1
    );

    let node = call(&fx, "lcm_describe", json!({"node_id": 1})).await;
    assert_eq!(node["node_id"], 1);
    assert_eq!(node["source_type"], "messages");
    assert_eq!(node["num_sources"], 2);
    assert!(node["expand_hint"].as_str().unwrap().contains("rollout"));
}

#[tokio::test]
async fn describe_missing_node_uses_python_error_text() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_describe", json!({"node_id": 99})).await;
    assert_eq!(out["error"], "Node 99 not found in current session");
}

// ---- lcm_expand -----------------------------------------------------------------------------------

#[tokio::test]
async fn expand_store_id_recovers_exact_content_cross_session() {
    let fx = Fixture::new("");
    // store_id 3 is the s2 message; expand works cross-session.
    let out = call(&fx, "lcm_expand", json!({"store_id": 3})).await;
    assert_eq!(out["source_type"], "raw_message");
    assert_eq!(out["session_id"], "s2");
    assert_eq!(out["from_current_session"], false);
    assert!(out["content"].as_str().unwrap().contains("lunch"));
}

#[tokio::test]
async fn expand_node_pages_sources_within_token_budget() {
    let fx = Fixture::new("");
    // A tiny budget forces single-source truncation + pagination.
    let out = call(&fx, "lcm_expand", json!({"node_id": 1, "max_tokens": 2})).await;
    assert_eq!(out["source_type"], "messages");
    assert_eq!(out["pagination"]["total_sources"], 2);
    assert_eq!(out["pagination"]["has_more"], true);
    let expanded = out["expanded"].as_array().unwrap();
    assert_eq!(expanded[0]["content_truncated"], true);
    assert!(expanded[0]["next_content_offset"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn expand_rejects_mode_combinations_with_python_error_text() {
    let fx = Fixture::new("");
    let both = call(&fx, "lcm_expand", json!({"node_id": 1, "store_id": 1})).await;
    assert_eq!(
        both["error"],
        "Provide only one of node_id, externalized_ref, store_id (got store_id, node_id)"
    );
    let none = call(&fx, "lcm_expand", json!({})).await;
    assert_eq!(
        none["error"],
        "node_id, externalized_ref, or store_id is required"
    );
}

// ---- lcm_expand_query -----------------------------------------------------------------------------

#[tokio::test]
async fn expand_query_answers_over_recovered_context() {
    let fx = Fixture::new("It was a blue/green rollout.");
    let out = call(
        &fx,
        "lcm_expand_query",
        json!({"prompt": "what rollout strategy?", "query": "rollout"}),
    )
    .await;
    assert!(out["answer"].as_str().unwrap().contains("blue/green"));
    assert_eq!(out["node_ids"], json!([1]));
    assert_eq!(out["matches"].as_array().unwrap().len(), 1);
    assert!(out.get("degraded").is_none());
}

#[tokio::test]
async fn expand_query_reports_no_matches_without_synthesis() {
    let fx = Fixture::new("unused");
    let out = call(
        &fx,
        "lcm_expand_query",
        json!({"prompt": "anything?", "query": "zzz-no-such-token"}),
    )
    .await;
    assert_eq!(
        out["answer"],
        "No matching summaries found in the current session."
    );
    assert_eq!(out["node_ids"], json!([]));
}

#[tokio::test]
async fn expand_query_requires_prompt_and_selector() {
    let fx = Fixture::new("");
    let no_prompt = call(&fx, "lcm_expand_query", json!({"query": "x"})).await;
    assert_eq!(no_prompt["error"], "prompt is required");
    let no_selector = call(&fx, "lcm_expand_query", json!({"prompt": "x"})).await;
    assert_eq!(no_selector["error"], "Provide either query or node_ids");
}

/// The synthesis call carries the tool's answer budget as the output cap plus the `"compression"`
/// task label (`LCM:tools.py:609-618`); temperature stays at the provider default.
#[tokio::test]
async fn expand_query_synthesis_carries_answer_budget_and_task() {
    use daemon_core::provider::{Capabilities, Failure, ModelOutput, Request, ToolCallFormat};
    use daemon_core::Provider;

    struct CapturingAux {
        request: std::sync::Mutex<Option<Request>>,
    }
    #[async_trait::async_trait]
    impl Provider for CapturingAux {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: false,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }
        async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
            *self.request.lock().unwrap() = Some(req);
            Ok(ModelOutput {
                text: "an answer".into(),
                ..Default::default()
            })
        }
    }

    let fx = Fixture::new("unused");
    let capture = CapturingAux {
        request: std::sync::Mutex::new(None),
    };
    let cx = ToolCx {
        aux: &capture,
        ..fx.cx()
    };
    let out = dispatch(
        &cx,
        "lcm_expand_query",
        json!({"prompt": "what rollout strategy?", "query": "rollout", "max_tokens": 777}),
    )
    .await;
    let out: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(out["answer"], "an answer");
    let req = capture
        .request
        .lock()
        .unwrap()
        .take()
        .expect("synthesis called the aux provider");
    assert_eq!(req.params.max_tokens, Some(777));
    assert_eq!(req.params.temperature, None, "provider default temperature");
    assert_eq!(req.task.as_deref(), Some("compression"));
}

// ---- lcm_status -----------------------------------------------------------------------------------

#[tokio::test]
async fn status_reports_counts_config_and_dag_rollup() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_status", json!({})).await;
    assert_eq!(out["session_id"], "s1");
    assert_eq!(out["compression_count"], 2);
    assert_eq!(out["last_compression_status"], "idle");
    assert_eq!(out["model"], "test-model");
    assert_eq!(out["context_length_source"], "model_info");
    assert_eq!(out["threshold_tokens"], 350);
    assert_eq!(out["store"]["messages"], 2);
    assert_eq!(out["dag"]["total_nodes"], 1);
    assert_eq!(out["dag"]["depths"]["d0"]["count"], 1);
    assert_eq!(out["dag"]["depths"]["d0"]["source_tokens"], 40);
    // 40 source tokens over 9 summary tokens -> "4.4:1".
    assert_eq!(out["dag"]["compression_ratio"], "4.4:1");
    assert_eq!(out["session_filters"]["side_channel_active"], false);
    assert_eq!(out["ingest_protection"]["source"], "default");
    assert_eq!(out["preset_suggestion"]["read_only"], true);
    assert!(out["lifecycle"].is_null());
    assert_eq!(out["lifecycle_fragmentation"]["read_only"], true);
    assert!(out["runtime_identity"]["session_bound"].as_bool().unwrap());
}

// ---- lcm_doctor -----------------------------------------------------------------------------------

#[tokio::test]
async fn doctor_is_healthy_on_a_consistent_store() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_doctor", json!({})).await;
    assert_eq!(out["overall"], "healthy");
    let checks = out["checks"].as_array().unwrap();
    // The full ported catalog, in the Python emission order, plus `context_pressure` (emitted
    // because the fixture's `ToolCx` carries a `context_length`).
    let names: Vec<&str> = checks
        .iter()
        .map(|c| c["check"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec![
            "database_integrity",
            "schema_core_tables",
            "messages_fts_integrity",
            "nodes_fts_integrity",
            "sqlite_storage",
            "payload_storage",
            "sensitive_pattern_handling",
            "fts_index_sync",
            "orphaned_dag_nodes",
            "summary_quality",
            "config_validation",
            "source_lineage_hygiene",
            "lifecycle_fragmentation",
            "context_pressure",
        ]
    );
    // Every check on a clean, low-pressure store passes.
    assert!(
        checks.iter().all(|c| c["status"] == "pass"),
        "all checks pass on a consistent store: {checks:?}"
    );
}

#[tokio::test]
async fn doctor_ported_checks_carry_structured_detail() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_doctor", json!({})).await;
    let checks = out["checks"].as_array().unwrap();
    let find = |name: &str| checks.iter().find(|c| c["check"] == name).unwrap();

    // schema_core_tables lists the seven core objects, none missing.
    let schema = find("schema_core_tables");
    assert_eq!(schema["status"], "pass");
    assert!(schema["detail"]["missing"].as_array().unwrap().is_empty());
    assert_eq!(schema["detail"]["present"].as_array().unwrap().len(), 7);

    // sqlite_storage reports an in-memory bank with a healthy quick_check.
    let storage = find("sqlite_storage");
    assert_eq!(storage["status"], "pass");
    assert_eq!(storage["detail"]["quick_check"], "ok");
    assert_eq!(storage["detail"]["in_memory"], true);
    assert_eq!(storage["detail"]["database_exists"], false);

    // payload_storage merges the risk scan, directory stats, and ref/file integrity.
    let payload = find("payload_storage");
    assert_eq!(payload["status"], "pass");
    assert_eq!(payload["detail"]["externalized_payload_count"], 0);
    assert_eq!(payload["detail"]["externalized_payload_refs_missing"], 0);
    assert!(payload["detail"]["largest_content_rows"].is_array());

    // fts_index_sync is session-scoped and phrased like the Python detail string.
    let fts = find("fts_index_sync");
    assert_eq!(fts["status"], "pass");
    assert_eq!(fts["detail"], "2 session FTS rows, 2 session messages");

    // summary_quality covers the seeded D0 node and flags nothing degraded.
    let sq = find("summary_quality");
    assert_eq!(sq["status"], "pass");
    assert_eq!(sq["detail"]["total_nodes"], 1);
    assert_eq!(sq["detail"]["extreme_ratio_nodes"], 0);
    assert!(sq["detail"]["recommendation"]
        .as_str()
        .unwrap()
        .contains("within the diagnostic thresholds"));

    // source_lineage_hygiene is detail-only (always pass) and bank-wide (s1 + s2 = 3 messages).
    let src = find("source_lineage_hygiene");
    assert_eq!(src["status"], "pass");
    assert_eq!(src["detail"]["messages_total"], 3);
    assert_eq!(
        src["detail"]["normalization_mode"],
        "backcompat-normalization"
    );

    // lifecycle_fragmentation is read-only and, with no lifecycle rows bound, not fragmented.
    let frag = find("lifecycle_fragmentation");
    assert_eq!(frag["status"], "pass");
    assert_eq!(frag["detail"]["read_only"], true);

    // context_pressure formats percentages the way Python str()s floats (default threshold 0.35).
    let pressure = find("context_pressure");
    assert_eq!(pressure["status"], "pass");
    assert_eq!(
        pressure["detail"],
        "0.0% used, compaction triggers at 35.0%"
    );

    // config_validation is all-clear on the default config.
    assert_eq!(find("config_validation")["status"], "pass");
}

#[tokio::test]
async fn doctor_config_validation_warns_on_out_of_range_settings() {
    let fx = Fixture::new("");
    let config = LcmConfig {
        fresh_tail_count: 1,
        incremental_max_depth: 0,
        ..LcmConfig::in_memory()
    };
    let out = call_with(&fx, &config, "lcm_doctor", json!({})).await;
    let checks = out["checks"].as_array().unwrap();
    let cfg = checks
        .iter()
        .find(|c| c["check"] == "config_validation")
        .unwrap();
    assert_eq!(cfg["status"], "warn");
    let warnings = cfg["detail"].as_array().unwrap();
    assert!(warnings
        .iter()
        .any(|w| w.as_str().unwrap().contains("fresh_tail_count")));
    assert!(warnings
        .iter()
        .any(|w| w.as_str().unwrap().contains("incremental_max_depth")));
    assert_eq!(out["overall"], "warnings");
}

#[tokio::test]
async fn doctor_reports_payload_and_sensitive_checks() {
    let fx = Fixture::new("");
    // An unrecognized sensitive pattern name -> warn on sensitive_pattern_handling, with the
    // configured/active/unknown split visible in the detail.
    let config = LcmConfig {
        sensitive_patterns_enabled: true,
        sensitive_patterns: vec!["api_key".to_string(), "not_a_real_pattern".to_string()],
        ..LcmConfig::in_memory()
    };
    let out = call_with(&fx, &config, "lcm_doctor", json!({})).await;
    let checks = out["checks"].as_array().unwrap();
    let storage = checks
        .iter()
        .find(|c| c["check"] == "payload_storage")
        .unwrap();
    assert_eq!(
        storage["status"], "pass",
        "ephemeral payload storage passes"
    );
    let sens = checks
        .iter()
        .find(|c| c["check"] == "sensitive_pattern_handling")
        .unwrap();
    assert_eq!(sens["status"], "warn", "unknown pattern name warns");
    assert_eq!(sens["detail"]["active_patterns"], json!(["api_key"]));
    assert_eq!(
        sens["detail"]["unknown_patterns"],
        json!(["not_a_real_pattern"])
    );
    assert_eq!(out["overall"], "warnings");
}

// ---- externalized payloads through describe/expand ------------------------------------------------

#[tokio::test]
async fn expand_and_describe_externalized_ref_round_trip() {
    let fx = Fixture::new("");
    let dir = std::env::temp_dir().join(format!("lcm-tool-ext-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let config = LcmConfig {
        data_dir: dir.clone(),
        bank: "default".to_string(),
        ..LcmConfig::default()
    };
    let payload_dir = config.externalization_dir().unwrap();
    let body = "RECOVERED-PAYLOAD-".repeat(50);
    let reference = crate::externalize::store_payload(
        &payload_dir,
        &body,
        &crate::externalize::PayloadMeta {
            kind: "tool_result",
            field: "content",
            role: "tool",
            tool_call_id: Some("c1"),
            session_id: "s1",
        },
    )
    .unwrap();

    let expanded = call_with(
        &fx,
        &config,
        "lcm_expand",
        json!({"externalized_ref": reference}),
    )
    .await;
    assert_eq!(expanded["source_type"], "externalized_payload");
    assert_eq!(expanded["kind"], "tool_result");
    assert!(expanded["content"]
        .as_str()
        .unwrap()
        .contains("RECOVERED-PAYLOAD-"));

    let described = call_with(
        &fx,
        &config,
        "lcm_describe",
        json!({"externalized_ref": reference}),
    )
    .await;
    assert_eq!(described["kind"], "tool_result");
    assert_eq!(described["tool_call_id"], "c1");
    assert!(
        described.get("content").is_none(),
        "describe is preview-only"
    );
    assert!(described["content_preview"]
        .as_str()
        .unwrap()
        .starts_with("RECOVERED-PAYLOAD-"));

    let missing = call_with(
        &fx,
        &config,
        "lcm_expand",
        json!({"externalized_ref": "nope.json"}),
    )
    .await;
    assert_eq!(
        missing["error"],
        "Externalized payload nope.json not found in current session"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn externalized_ref_from_another_session_is_invisible() {
    let fx = Fixture::new("");
    let dir = std::env::temp_dir().join(format!("lcm-tool-ext-scope-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let config = LcmConfig {
        data_dir: dir.clone(),
        bank: "default".to_string(),
        ..LcmConfig::default()
    };
    let payload_dir = config.externalization_dir().unwrap();
    let reference = crate::externalize::store_payload(
        &payload_dir,
        "other-session payload body",
        &crate::externalize::PayloadMeta {
            kind: "tool_result",
            field: "content",
            role: "tool",
            tool_call_id: Some("c9"),
            session_id: "someone-else",
        },
    )
    .unwrap();
    let out = call_with(
        &fx,
        &config,
        "lcm_expand",
        json!({"externalized_ref": reference}),
    )
    .await;
    assert!(out["error"]
        .as_str()
        .unwrap()
        .contains("not found in current session"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn unknown_tool_is_reported() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_nope", json!({})).await;
    assert_eq!(out["status"], "unknown_tool");
}
