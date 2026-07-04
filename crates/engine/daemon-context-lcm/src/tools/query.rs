// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `lcm_expand_query` (§10.5, `LCM:tools.py:1226-1446`): expand matching summaries into context
//! blocks and synthesize an answer through the auxiliary provider.

use super::expand::{expand_child_nodes, expand_message_sources, get_session_node};
use super::parse::{format_sig3, py_str_or_empty, truncate_text_to_token_budget};
use super::{err, ToolCx};
use crate::escalation::strip_reasoning_blocks;
use crate::search::{self, SortMode};
use crate::store::{SourceType, SummaryNode};
use daemon_core::{Request, RequestMsg};
use serde_json::{json, Map, Value};
use std::time::Duration;

/// `_collect_context_blocks_for_node` (`LCM:tools.py:520-570`): the truncated summary block plus
/// one hydrated source block (messages or child nodes) under the remaining budget.
fn collect_context_blocks_for_node(
    cx: &ToolCx<'_>,
    node: &SummaryNode,
    max_tokens: i64,
    hydrate_externalized_content: bool,
) -> Vec<Value> {
    let (summary, summary_truncated) =
        truncate_text_to_token_budget(cx.tokenizer, &node.summary, max_tokens);
    let mut blocks = vec![json!({
        "type": "summary",
        "node_id": node.node_id,
        "depth": node.depth,
        "summary": summary,
        "summary_truncated": summary_truncated,
        "expand_hint": node.expand_hint,
        "token_count": node.token_count,
    })];
    let remaining_tokens = (max_tokens - cx.tokenizer.count_text(&summary) as i64).max(0);

    match node.source_type {
        SourceType::Messages => {
            let (messages, pagination) = expand_message_sources(
                cx,
                node,
                remaining_tokens,
                0,
                None,
                0,
                hydrate_externalized_content,
            );
            let has_more = pagination
                .get("has_more")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !messages.is_empty() || has_more {
                blocks.push(json!({
                    "type": "messages",
                    "node_id": node.node_id,
                    "messages": messages,
                    "pagination": pagination,
                }));
            }
        }
        SourceType::Nodes => {
            let (children, pagination) =
                expand_child_nodes(cx, node, Some(remaining_tokens), 0, None);
            let has_more = pagination
                .get("has_more")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !children.is_empty() || has_more {
                blocks.push(json!({
                    "type": "child_nodes",
                    "node_id": node.node_id,
                    "children": children,
                    "pagination": pagination,
                }));
            }
        }
    }
    blocks
}

/// `_context_content_token_count` (`LCM:tools.py:573-586`): what the blocks actually cost.
fn context_content_token_count(cx: &ToolCx<'_>, blocks: &[Value]) -> i64 {
    let count = |v: Option<&Value>| -> i64 {
        cx.tokenizer
            .count_text(v.and_then(Value::as_str).unwrap_or("")) as i64
    };
    let mut total = 0i64;
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("summary") => total += count(block.get("summary")),
            Some("messages") => {
                for message in block
                    .get("messages")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                {
                    total += count(message.get("content"));
                    total += count(message.get("transcript_content"));
                }
            }
            Some("child_nodes") => {
                for child in block
                    .get("children")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                {
                    total += count(child.get("summary"));
                }
            }
            _ => {}
        }
    }
    total
}

/// `_synthesize_expansion_answer` (`LCM:tools.py:589-624`), over the injected aux provider. The
/// Python model-routing layer has no daemon counterpart — the engine's aux provider *is* the
/// route. Reasoning blocks are stripped like every other aux response.
async fn synthesize_expansion_answer(
    cx: &ToolCx<'_>,
    prompt: &str,
    context_blocks: &[Value],
    timeout: Duration,
) -> Result<String, SynthesisError> {
    let system_prompt = "You answer questions using expanded LCM retrieval context. \
                         Be concise, factual, and grounded in the provided context. \
                         If the context is insufficient, say so plainly.";
    let context_json =
        serde_json::to_string_pretty(&context_blocks).unwrap_or_else(|_| "[]".to_string());
    let user_prompt = format!("QUESTION:\n{prompt}\n\nEXPANDED CONTEXT:\n{context_json}");
    let request = Request {
        system: system_prompt.to_string(),
        messages: vec![RequestMsg {
            role: "user".into(),
            content: user_prompt,
            ..Default::default()
        }],
        ..Default::default()
    };
    match tokio::time::timeout(timeout, cx.aux.chat(request)).await {
        Err(_) => Err(SynthesisError::Timeout),
        Ok(Err(e)) => Err(SynthesisError::Provider(e.to_string())),
        Ok(Ok(out)) => Ok(strip_reasoning_blocks(&out.text).trim().to_string()),
    }
}

enum SynthesisError {
    Timeout,
    Provider(String),
}

pub(super) async fn expand_query(cx: &ToolCx<'_>, args: &Value) -> String {
    let prompt = py_str_or_empty(args.get("prompt")).trim().to_string();
    if prompt.is_empty() {
        return err("prompt is required");
    }

    // `_parse_int_arg` (`LCM:tools.py:1236`): int() coercion (booleans included) or the error.
    let parse_int_arg = |name: &str, default: i64| -> Result<i64, String> {
        match args.get(name) {
            None => Ok(default),
            Some(v) => match v {
                Value::Bool(b) => Ok(i64::from(*b)),
                _ => {
                    super::parse::coerce_int(v).ok_or_else(|| format!("{name} must be an integer"))
                }
            },
        }
    };

    let max_tokens = match parse_int_arg("max_tokens", 2000) {
        Ok(n) => n.max(1),
        Err(e) => return err(&e),
    };
    let context_default = max_tokens.max(cx.config.expansion_context_tokens.max(1) as i64);
    let context_max_tokens = match parse_int_arg("context_max_tokens", context_default) {
        Ok(n) => n.max(1),
        Err(e) => return err(&e),
    };
    let max_results = match parse_int_arg("max_results", 5) {
        Ok(n) => n,
        Err(e) => return err(&e),
    };

    let query = py_str_or_empty(args.get("query")).trim().to_string();
    let raw_node_ids = args.get("node_ids").filter(|v| super::parse::py_truthy(v));

    let mut nodes: Vec<SummaryNode> = Vec::new();
    if let Some(raw_node_ids) = raw_node_ids {
        // Python iterates whatever it got; anything non-integer inside raises the same error.
        let Some(items) = raw_node_ids.as_array() else {
            return err("node_ids must contain only integers");
        };
        for raw in items {
            let parsed = match raw {
                Value::Bool(b) => Some(i64::from(*b)),
                _ => super::parse::coerce_int(raw),
            };
            let Some(node_id) = parsed else {
                return err("node_ids must contain only integers");
            };
            if let Some(node) = get_session_node(cx, node_id) {
                nodes.push(node);
            }
        }
    } else if !query.is_empty() {
        nodes = search::search_nodes(
            cx.store,
            &query,
            Some(cx.session_id),
            SortMode::Recency,
            None,
            max_results.max(0) as usize,
        )
        .unwrap_or_default()
        .into_iter()
        .map(|r| r.node)
        .collect();
    } else {
        return err("Provide either query or node_ids");
    }

    if nodes.is_empty() {
        return json!({
            "prompt": prompt,
            "query": query,
            "answer": "No matching summaries found in the current session.",
            "node_ids": [],
            "matches": [],
        })
        .to_string();
    }

    let selected_nodes: Vec<&SummaryNode> =
        nodes.iter().take(max_results.max(0) as usize).collect();
    let mut context_blocks: Vec<Value> = Vec::new();
    let mut context_budget_used = 0i64;
    for node in &selected_nodes {
        let remaining_context_tokens = (context_max_tokens - context_budget_used).max(0);
        let node_blocks = collect_context_blocks_for_node(cx, node, remaining_context_tokens, true);
        context_budget_used += context_content_token_count(cx, &node_blocks);
        context_blocks.extend(node_blocks);
    }

    let context_pagination = build_context_pagination(&context_blocks);
    let context_truncated = context_pagination.iter().any(|item| {
        item.get("summary_truncated")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || item
                .get("pagination")
                .and_then(|p| p.get("has_more"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
    });

    let matches: Vec<Value> = selected_nodes
        .iter()
        .map(|node| {
            json!({
                "node_id": node.node_id,
                "depth": node.depth,
                "summary": node.summary.chars().take(300).collect::<String>(),
                "expand_hint": node.expand_hint,
            })
        })
        .collect();
    let node_ids: Vec<i64> = selected_nodes.iter().map(|n| n.node_id).collect();

    let model = if !cx.config.expansion_model.is_empty() {
        cx.config.expansion_model.clone()
    } else if !cx.config.summary_model.is_empty() {
        cx.config.summary_model.clone()
    } else {
        String::new()
    };
    let timeout = Duration::from_millis(cx.config.expansion_timeout_ms);
    let timeout_seconds = cx.config.expansion_timeout_ms as f64 / 1000.0;

    let degraded_payload = |reason: String, include_timeout: bool| -> String {
        let mut payload = Map::new();
        payload.insert("prompt".into(), json!(prompt));
        payload.insert("query".into(), json!(query));
        payload.insert("error".into(), json!(reason));
        payload.insert("degraded".into(), json!(true));
        payload.insert("model".into(), json!(model));
        payload.insert("max_tokens".into(), json!(max_tokens));
        payload.insert("context_max_tokens".into(), json!(context_max_tokens));
        payload.insert("context_truncated".into(), json!(context_truncated));
        payload.insert("context_pagination".into(), json!(context_pagination));
        payload.insert("node_ids".into(), json!(node_ids));
        payload.insert("matches".into(), json!(matches));
        if include_timeout {
            payload.insert("timeout_seconds".into(), json!(timeout_seconds));
        }
        Value::Object(payload).to_string()
    };

    let answer = match synthesize_expansion_answer(cx, &prompt, &context_blocks, timeout).await {
        Ok(answer) => answer,
        Err(SynthesisError::Timeout) => {
            tracing::warn!(
                timeout = timeout_seconds,
                "lcm: expand_query synthesis timed out"
            );
            return degraded_payload(
                format!(
                    "lcm_expand_query synthesis timed out after {}s",
                    format_sig3(timeout_seconds)
                ),
                true,
            );
        }
        // The Python handler lets provider exceptions escape to the plugin harness; the daemon
        // engine degrades in-band instead (there is no harness to catch a panic).
        Err(SynthesisError::Provider(e)) => {
            tracing::warn!(error = %e, "lcm: expand_query synthesis failed");
            return degraded_payload(format!("lcm_expand_query synthesis failed: {e}"), false);
        }
    };
    if answer.is_empty() {
        tracing::warn!("lcm: expand_query synthesis returned an empty answer");
        return degraded_payload(
            "lcm_expand_query synthesis returned an empty answer".to_string(),
            false,
        );
    }

    json!({
        "prompt": prompt,
        "query": query,
        "answer": answer,
        "model": model,
        "max_tokens": max_tokens,
        "context_max_tokens": context_max_tokens,
        "context_truncated": context_truncated,
        "context_pagination": context_pagination,
        "node_ids": node_ids,
        "matches": matches,
    })
    .to_string()
}

/// The `context_pagination` assembly (`LCM:tools.py:1299-1374`): every truncated summary, child
/// summary, and unfinished source block, each with ready-to-send `lcm_expand` arguments.
fn build_context_pagination(context_blocks: &[Value]) -> Vec<Value> {
    let mut context_pagination: Vec<Value> = Vec::new();
    for block in context_blocks {
        let Some(block_obj) = block.as_object() else {
            continue;
        };
        let block_type = block_obj.get("type").and_then(Value::as_str);
        let node_id = block_obj.get("node_id").cloned().unwrap_or(Value::Null);

        if block_type == Some("summary")
            && block_obj
                .get("summary_truncated")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        {
            context_pagination.push(json!({
                "node_id": node_id,
                "type": "summary",
                "summary_truncated": true,
                "expand_args": {"node_id": node_id},
            }));
            continue;
        }

        if block_type == Some("child_nodes") {
            for child in block_obj
                .get("children")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                if child
                    .get("summary_truncated")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    let child_node_id = child.get("node_id").cloned().unwrap_or(Value::Null);
                    context_pagination.push(json!({
                        "node_id": node_id,
                        "type": "child_summary",
                        "child_node_id": child_node_id,
                        "source_index": child.get("source_index").cloned().unwrap_or(Value::Null),
                        "summary_truncated": true,
                        "expand_args": {"node_id": child_node_id},
                    }));
                }
            }
        }

        let Some(pagination) = block_obj.get("pagination") else {
            continue;
        };
        if !pagination
            .get("has_more")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            continue;
        }

        let mut item = Map::new();
        item.insert("node_id".into(), node_id.clone());
        item.insert("type".into(), json!(block_type));
        item.insert("pagination".into(), pagination.clone());
        let next_source_offset = pagination
            .get("next_source_offset")
            .cloned()
            .filter(|v| !v.is_null())
            .unwrap_or(json!(0));
        let next_content_offset = pagination
            .get("next_content_offset")
            .cloned()
            .filter(|v| !v.is_null())
            .unwrap_or(json!(0));
        if block_type == Some("messages") {
            let truncated_message = block_obj
                .get("messages")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .find(|m| {
                    m.get("content_truncated")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                });
            if let Some(message) = truncated_message {
                item.insert(
                    "source_index".into(),
                    message.get("source_index").cloned().unwrap_or(Value::Null),
                );
                item.insert(
                    "content_source".into(),
                    message
                        .get("content_source")
                        .cloned()
                        .unwrap_or(Value::Null),
                );
                let externalized = message.get("externalized").cloned().unwrap_or(json!({}));
                let externalized_ref = externalized.get("ref").cloned().filter(|v| !v.is_null());
                if let Some(reference) = &externalized_ref {
                    item.insert("externalized_ref".into(), reference.clone());
                    item.insert(
                        "tool_call_id".into(),
                        externalized
                            .get("tool_call_id")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                }
                let hydrated = message.get("content_source").and_then(Value::as_str)
                    == Some("externalized_payload");
                if let (true, Some(reference)) = (hydrated, externalized_ref) {
                    item.insert(
                        "expand_args".into(),
                        json!({
                            "externalized_ref": reference,
                            "content_offset": next_content_offset,
                        }),
                    );
                } else {
                    item.insert(
                        "expand_args".into(),
                        json!({
                            "node_id": node_id,
                            "source_offset": next_source_offset,
                            "content_offset": next_content_offset,
                        }),
                    );
                }
            } else {
                item.insert(
                    "expand_args".into(),
                    json!({
                        "node_id": node_id,
                        "source_offset": next_source_offset,
                        "content_offset": next_content_offset,
                    }),
                );
            }
        } else if block_type == Some("child_nodes") {
            item.insert(
                "expand_args".into(),
                json!({
                    "node_id": node_id,
                    "source_offset": next_source_offset,
                }),
            );
        }
        context_pagination.push(Value::Object(item));
    }
    context_pagination
}
