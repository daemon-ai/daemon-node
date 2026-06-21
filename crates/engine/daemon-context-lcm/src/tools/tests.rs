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
        ToolCx {
            store: &self.store,
            config: &CONFIG,
            tokenizer: &self.tokenizer,
            aux: &self.aux,
            session_id: "s1",
            threshold_tokens: Some(350),
            compaction_count: 2,
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

#[tokio::test]
async fn grep_returns_message_and_summary_hits_for_current_session() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_grep", json!({"query": "blue green", "sort": "recency"})).await;
    assert_eq!(out["session_scope"], "current");
    let results = out["results"].as_array().unwrap();
    assert!(results.iter().any(|r| r["type"] == "message"));
    assert!(results.iter().any(|r| r["type"] == "summary"));
    assert_eq!(out["summary_results_omitted"], false);
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
}

#[tokio::test]
async fn grep_requires_session_id_for_session_scope() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_grep", json!({"query": "x", "session_scope": "session"})).await;
    assert_eq!(out["status"], "error");
}

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
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["content_truncated"], true);
    assert_eq!(msgs[0]["next_content_offset"], 5);
    // The cursor advances; the next page returns the second row.
    let cursor = out["next_cursor"].clone();
    let page2 = call(
        &fx,
        "lcm_load_session",
        json!({"session_id": "s1", "limit": 10, "after_store_id": cursor}),
    )
    .await;
    assert_eq!(page2["has_more"], false);
    assert_eq!(page2["messages"].as_array().unwrap()[0]["role"], "assistant");
}

#[tokio::test]
async fn describe_overview_and_node_subtree() {
    let fx = Fixture::new("");
    let overview = call(&fx, "lcm_describe", json!({})).await;
    assert_eq!(overview["total_nodes"], 1);
    assert_eq!(overview["depths"][0]["depth"], 0);
    assert_eq!(overview["depths"][0]["count"], 1);

    let node = call(&fx, "lcm_describe", json!({"node_id": 1})).await;
    assert_eq!(node["node_id"], 1);
    assert_eq!(node["source_type"], "messages");
    assert_eq!(node["source_count"], 2);
    assert!(node["preview"].as_str().unwrap().contains("blue/green"));
}

#[tokio::test]
async fn expand_store_id_recovers_exact_content_cross_session() {
    let fx = Fixture::new("");
    // store_id 3 is the s2 message; expand works cross-session.
    let out = call(&fx, "lcm_expand", json!({"store_id": 3})).await;
    assert_eq!(out["type"], "message");
    assert_eq!(out["session_id"], "s2");
    assert!(out["content"].as_str().unwrap().contains("lunch"));
}

#[tokio::test]
async fn expand_node_pages_sources_within_token_budget() {
    let fx = Fixture::new("");
    // A tiny budget forces single-source truncation + pagination.
    let out = call(
        &fx,
        "lcm_expand",
        json!({"node_id": 1, "max_tokens": 2}),
    )
    .await;
    assert_eq!(out["type"], "node_expansion");
    assert_eq!(out["pagination"]["total_sources"], 2);
    assert_eq!(out["pagination"]["has_more"], true);
}

#[tokio::test]
async fn expand_query_answers_over_recovered_context() {
    let fx = Fixture::new("It was a blue/green rollout.");
    let out = call(
        &fx,
        "lcm_expand_query",
        json!({"prompt": "what rollout strategy?", "query": "rollout"}),
    )
    .await;
    assert_eq!(out["status"], "ok");
    assert!(out["answer"].as_str().unwrap().contains("blue/green"));
    assert!(out["nodes_used"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn status_reports_counts_and_config() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_status", json!({})).await;
    assert_eq!(out["compaction_count"], 2);
    assert_eq!(out["store"]["session_messages"], 2);
    assert_eq!(out["store"]["session_summaries"], 1);
    assert_eq!(out["threshold_tokens"], 350);
}

#[tokio::test]
async fn doctor_is_healthy_on_a_consistent_store() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_doctor", json!({})).await;
    assert_eq!(out["overall"], "healthy");
    let checks = out["checks"].as_array().unwrap();
    assert!(checks.iter().any(|c| c["check"] == "database_integrity" && c["status"] == "ok"));
    assert!(checks.iter().any(|c| c["check"] == "messages_fts_integrity" && c["status"] == "ok"));
}

#[tokio::test]
async fn unknown_tool_is_reported() {
    let fx = Fixture::new("");
    let out = call(&fx, "lcm_nope", json!({})).await;
    assert_eq!(out["status"], "unknown_tool");
}
