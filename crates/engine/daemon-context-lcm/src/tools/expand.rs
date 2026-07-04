// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `lcm_describe` (§10.3, `LCM:tools.py:974-1033`) and `lcm_expand` (§10.4,
//! `LCM:tools.py:1036-1223`), plus the source-expansion helpers `lcm_expand_query` shares
//! (`_expand_message_sources` / `_expand_child_nodes`, `LCM:tools.py:302-517`).

use super::parse::{
    coerce_int, full_content_slice, parse_non_negative_int, parse_positive_int, py_display,
    py_str_or_empty, py_truthy, slice_content_for_response, truncate_text_to_token_budget,
    ContentSlice,
};
use super::{err, ToolCx};
use crate::externalize;
use crate::extraction::sanitize_pre_compaction_content;
use crate::store::{MessageRow, SourceType, SummaryNode};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// `_get_session_node` (`LCM:tools.py:77`): a node visible to the current session only.
pub(super) fn get_session_node(cx: &ToolCx<'_>, node_id: i64) -> Option<SummaryNode> {
    cx.store
        .get_node(node_id)
        .ok()
        .flatten()
        .filter(|n| n.session_id == cx.session_id)
}

/// `_get_externalized_payload` (`LCM:tools.py:84-97`): load a payload by ref, gated on session —
/// a payload claiming a session outside `allowed` is invisible; an unclaimed one always loads.
pub(super) fn get_externalized_payload(
    cx: &ToolCx<'_>,
    reference: &str,
    allowed_session_ids: &[&str],
) -> Option<Value> {
    let dir = cx.config.externalization_dir()?;
    let payload = externalize::load_payload(&dir, reference)?;
    let payload_session = payload
        .get("session_id")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    if !payload_session.is_empty() && !allowed_session_ids.contains(&payload_session) {
        return None;
    }
    Some(payload)
}

/// A JSON node-id argument the way SQLite affinity would accept it from Python: an integer, an
/// integral float, or a numeric string.
pub(super) fn parse_node_id_like(value: &Value) -> Option<i64> {
    match value {
        Value::Number(_) | Value::String(_) => coerce_int(value).filter(|_| {
            // Reject non-integral floats: SQLite `node_id = 3.7` matches nothing.
            value.as_f64().is_none_or(|f| f.fract() == 0.0)
        }),
        _ => None,
    }
}

/// `_is_compact_externalized_marker` (`LCM:tools.py:258`): a short placeholder row whose full text
/// should be returned verbatim rather than token-sliced.
fn is_compact_externalized_marker(content: &str, reference: Option<&str>) -> bool {
    if reference.is_none() || content.is_empty() {
        return false;
    }
    if content.chars().count() > 512 {
        return false;
    }
    content.starts_with("[Externalized tool output:")
        || content.starts_with("[GC'd externalized tool output:")
        || content.starts_with("[Externalized payload:")
        || content.starts_with("[GC'd externalized payload:")
        || content.contains("[Externalized LCM ingest payload:")
}

/// `_restore_ingest_placeholder_for_lookup` (`LCM:tools.py:238`): only when the row's own ref is
/// an ingest payload does restoring make sense; `None` when nothing changed.
fn restore_ingest_placeholder_for_lookup(
    cx: &ToolCx<'_>,
    content: &str,
    reference: Option<&str>,
    ref_payload: Option<&Value>,
    session_id: &str,
) -> Option<String> {
    if content.is_empty() || reference.is_none() {
        return None;
    }
    let payload = ref_payload?;
    if payload.get("kind").and_then(|k| k.as_str()) != Some("ingest_payload") {
        return None;
    }
    let dir = cx.config.externalization_dir()?;
    let restored = externalize::restore_ingest_placeholders(&dir, content, session_id);
    (restored != content).then_some(restored)
}

/// `find_externalized_payload_for_message` (`LCM:externalize.py:225`) narrowed to the tool-result
/// digest lookup `_expand_message_sources` performs: a content+tool_call_id+session match, as a
/// content-free summary.
fn find_externalized_payload_for_message(
    cx: &ToolCx<'_>,
    content: &str,
    tool_call_id: &str,
    session_id: &str,
) -> Option<Value> {
    if content.is_empty() {
        return None;
    }
    let dir = cx.config.externalization_dir()?;
    externalize::find_payload_for_content(
        &dir,
        content,
        Some("tool_result"),
        tool_call_id,
        session_id,
    )
    .map(|(reference, record)| externalize::payload_summary(&reference, &record))
}

/// Inputs to [`pagination_payload`] (`_pagination_payload`, `LCM:tools.py:272-299`).
pub(super) struct Pagination {
    pub total_sources: usize,
    pub source_offset: usize,
    pub content_offset: usize,
    pub source_limit: usize,
    pub returned_sources: usize,
    pub next_source_offset: Option<usize>,
    pub next_content_offset: usize,
    pub has_more: bool,
}

/// `_pagination_payload`: normalize the cursor fields when nothing remains.
pub(super) fn pagination_payload(mut p: Pagination) -> Value {
    if !p.has_more {
        p.next_source_offset = None;
        p.next_content_offset = 0;
    }
    let remaining_sources = match (p.has_more, p.next_source_offset) {
        (true, Some(next)) => p.total_sources.saturating_sub(next),
        _ => 0,
    };
    json!({
        "source_offset": p.source_offset,
        "content_offset": p.content_offset,
        "source_limit": p.source_limit,
        "returned_sources": p.returned_sources,
        "total_sources": p.total_sources,
        "next_source_offset": p.next_source_offset,
        "next_content_offset": p.next_content_offset,
        "has_more": p.has_more,
        "remaining_sources": remaining_sources,
    })
}

/// `_expand_message_sources` (`LCM:tools.py:302-444`): expand a D0 node's raw rows under a shared
/// token budget, hydrating externalized tool payloads and surfacing recovery metadata.
pub(super) fn expand_message_sources(
    cx: &ToolCx<'_>,
    node: &SummaryNode,
    max_tokens: i64,
    source_offset: usize,
    source_limit: Option<usize>,
    content_offset: usize,
    hydrate_externalized_content: bool,
) -> (Vec<Value>, Value) {
    let total_sources = node.source_ids.len();
    let source_offset = source_offset.min(total_sources);
    let remaining_source_count = total_sources - source_offset;
    let source_limit = source_limit
        .map(|l| l.min(remaining_source_count))
        .unwrap_or(remaining_source_count);
    let page_ids: Vec<i64> = node.source_ids[source_offset..source_offset + source_limit].to_vec();
    let stored_by_id: HashMap<i64, MessageRow> = cx
        .store
        .get_messages(&page_ids)
        .unwrap_or_default()
        .into_iter()
        .map(|r| (r.store_id, r))
        .collect();

    let mut messages: Vec<Value> = Vec::new();
    let mut budget_used: i64 = 0;
    let mut next_source_offset: Option<usize> = Some(source_offset);
    let mut next_content_offset = content_offset;
    let mut has_more = source_offset < total_sources;
    let mut broke = false;

    for (relative_index, store_id) in page_ids.iter().enumerate() {
        let source_index = source_offset + relative_index;
        let remaining_tokens = max_tokens - budget_used;
        if remaining_tokens <= 0 {
            next_source_offset = Some(source_index);
            next_content_offset = 0;
            has_more = true;
            broke = true;
            break;
        }
        let Some(stored) = stored_by_id.get(store_id) else {
            next_source_offset = Some(source_index + 1);
            next_content_offset = 0;
            has_more = source_index + 1 < total_sources;
            continue;
        };
        let transcript_content = stored.content.clone().unwrap_or_default();
        let mut content = transcript_content.clone();
        let mut content_source = "message";
        let ingest_refs = externalize::extract_ingest_refs(&transcript_content);
        let reference = ingest_refs
            .first()
            .cloned()
            .or_else(|| externalize::extract_ref(&transcript_content));
        let ref_payload = reference.as_deref().and_then(|r| {
            get_externalized_payload(cx, r, &[cx.session_id, stored.session_id.as_str()])
        });
        let externalized = ref_payload
            .clone()
            .filter(|p| p.get("kind").and_then(|k| k.as_str()) != Some("ingest_payload"));
        if hydrate_externalized_content {
            if let Some(ext) = &externalized {
                content = ext
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();
                content_source = "externalized_payload";
            }
        }
        let effective_content_offset = if source_index == source_offset {
            content_offset
        } else {
            0
        };
        let sliced: ContentSlice = if !hydrate_externalized_content
            && is_compact_externalized_marker(&content, reference.as_deref())
        {
            full_content_slice(&content, effective_content_offset)
        } else {
            slice_content_for_response(
                cx.tokenizer,
                &content,
                remaining_tokens,
                effective_content_offset,
            )
        };

        let mut expanded = Map::new();
        expanded.insert("store_id".into(), json!(stored.store_id));
        expanded.insert("source_index".into(), json!(source_index));
        expanded.insert("session_id".into(), json!(stored.session_id));
        expanded.insert("source".into(), json!(stored.source));
        expanded.insert(
            "from_current_session".into(),
            json!(stored.session_id == cx.session_id),
        );
        expanded.insert("role".into(), json!(stored.role));
        expanded.insert("content".into(), json!(sliced.content));
        expanded.insert("content_chars".into(), json!(sliced.content_chars));
        expanded.insert("content_offset".into(), json!(sliced.content_offset));
        expanded.insert(
            "content_returned_chars".into(),
            json!(sliced.content_returned_chars),
        );
        expanded.insert("content_truncated".into(), json!(sliced.content_truncated));
        expanded.insert(
            "next_content_offset".into(),
            json!(sliced.next_content_offset),
        );
        expanded.insert("content_source".into(), json!(content_source));
        if content_source == "externalized_payload" {
            expanded.insert("transcript_content".into(), json!(transcript_content));
        }
        if stored.role == "tool" {
            if let Some(ext) = &externalized {
                let mut summary = ext.clone();
                if let Value::Object(ref mut map) = summary {
                    map.remove("content");
                }
                expanded.insert("externalized".into(), summary);
            }
            if !expanded.contains_key("externalized") {
                // The digest lookup tries progressively restored/sanitized identities: the
                // sanitizer may have rewritten the row before externalization, and ingest
                // placeholders hide the original body (`LCM:tools.py:390-418`).
                let mut lookup_candidates: Vec<String> = vec![transcript_content.clone()];
                if let Some(restored) = restore_ingest_placeholder_for_lookup(
                    cx,
                    &transcript_content,
                    reference.as_deref(),
                    ref_payload.as_ref(),
                    &stored.session_id,
                ) {
                    lookup_candidates.insert(0, restored.clone());
                    let sanitized_restored = sanitize_pre_compaction_content(&restored);
                    if sanitized_restored != restored {
                        lookup_candidates.insert(0, sanitized_restored);
                    }
                }
                let sanitized_content = sanitize_pre_compaction_content(&transcript_content);
                if sanitized_content != transcript_content {
                    lookup_candidates.insert(0, sanitized_content);
                }
                for candidate in &lookup_candidates {
                    if let Some(summary) = find_externalized_payload_for_message(
                        cx,
                        candidate,
                        stored.tool_call_id.as_deref().unwrap_or(""),
                        &stored.session_id,
                    ) {
                        expanded.insert("externalized".into(), summary);
                        break;
                    }
                }
            }
        }
        messages.push(Value::Object(expanded));
        budget_used += cx.tokenizer.count_text(&sliced.content) as i64;
        if sliced.has_more {
            next_source_offset = Some(source_index);
            next_content_offset = sliced.next_content_offset;
            has_more = true;
            broke = true;
            break;
        }
        next_source_offset = Some(source_index + 1);
        next_content_offset = 0;
        has_more = source_index + 1 < total_sources;
    }
    if !broke {
        // Python's for-else: the whole page was consumed without a budget break.
        has_more = (source_offset + source_limit) < total_sources;
        next_source_offset = has_more.then_some(source_offset + source_limit);
        next_content_offset = 0;
    }

    let returned_sources = messages.len();
    let pagination = pagination_payload(Pagination {
        total_sources,
        source_offset,
        content_offset,
        source_limit,
        returned_sources,
        next_source_offset,
        next_content_offset,
        has_more,
    });
    (messages, pagination)
}

/// `_expand_child_nodes` (`LCM:tools.py:447-517`): expand a condensation node's children as
/// (optionally token-budgeted) summary stubs.
pub(super) fn expand_child_nodes(
    cx: &ToolCx<'_>,
    node: &SummaryNode,
    max_tokens: Option<i64>,
    source_offset: usize,
    source_limit: Option<usize>,
) -> (Vec<Value>, Value) {
    let total_sources = node.source_ids.len();
    let source_offset = source_offset.min(total_sources);
    let remaining_source_count = total_sources - source_offset;
    let source_limit = source_limit
        .map(|l| l.min(remaining_source_count))
        .unwrap_or(remaining_source_count);
    let selected: Vec<(usize, SummaryNode)> = node.source_ids
        [source_offset..source_offset + source_limit]
        .iter()
        .enumerate()
        .filter_map(|(relative_index, child_id)| {
            get_session_node(cx, *child_id).map(|child| (source_offset + relative_index, child))
        })
        .collect();

    let mut expanded: Vec<Value> = Vec::new();
    let mut budget_used: i64 = 0;
    let mut next_source_offset: Option<usize> = None;
    let mut has_more = (source_offset + source_limit) < total_sources;
    for (source_index, child) in &selected {
        let mut summary = child.summary.clone();
        let mut summary_truncated = false;
        if let Some(max_tokens) = max_tokens {
            let remaining_tokens = max_tokens - budget_used;
            if remaining_tokens <= 0 {
                next_source_offset = Some(*source_index);
                has_more = true;
                break;
            }
            let (truncated, was_truncated) =
                truncate_text_to_token_budget(cx.tokenizer, &summary, remaining_tokens);
            summary = truncated;
            summary_truncated = was_truncated;
        }
        let full_len = child.summary.chars().count();
        let rendered_summary: String = if max_tokens.is_none() {
            summary.chars().take(1000).collect()
        } else {
            summary.clone()
        };
        expanded.push(json!({
            "node_id": child.node_id,
            "source_index": source_index,
            "depth": child.depth,
            "summary": rendered_summary,
            "summary_truncated": summary_truncated || (max_tokens.is_none() && full_len > 1000),
            "token_count": child.token_count,
            "source_token_count": child.source_token_count,
            "expand_hint": child.expand_hint,
        }));
        budget_used += cx.tokenizer.count_text(&summary) as i64;
        if summary_truncated {
            next_source_offset = Some(source_index + 1);
            has_more = source_index + 1 < total_sources;
            break;
        }
        next_source_offset = Some(source_index + 1);
    }
    if has_more && next_source_offset.is_none() {
        next_source_offset = Some(source_offset + source_limit);
    }

    let pagination = pagination_payload(Pagination {
        total_sources,
        source_offset,
        content_offset: 0,
        source_limit,
        returned_sources: expanded.len(),
        next_source_offset,
        next_content_offset: 0,
        has_more,
    });
    (expanded, pagination)
}

// ---- 10.3 lcm_describe --------------------------------------------------------------------------

pub(super) fn describe(cx: &ToolCx<'_>, args: &Value) -> String {
    let externalized_ref = py_str_or_empty(args.get("externalized_ref"))
        .trim()
        .to_string();
    if !externalized_ref.is_empty() {
        let Some(payload) = get_externalized_payload(cx, &externalized_ref, &[cx.session_id])
        else {
            return err(&format!(
                "Externalized payload {externalized_ref} not found in current session"
            ));
        };
        let content = payload
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let preview: String = content.chars().take(500).collect();
        return json!({
            "externalized_ref": externalized_ref,
            "kind": payload.get("kind").cloned().unwrap_or_else(|| json!("tool_result")),
            "tool_call_id": payload.get("tool_call_id").cloned().unwrap_or_else(|| json!("")),
            "role": payload.get("role").cloned().unwrap_or_else(|| json!("")),
            "session_id": payload.get("session_id").cloned().unwrap_or_else(|| json!("")),
            "field_path": payload.get("field_path").cloned().unwrap_or_else(|| json!("")),
            "content_chars": payload.get("content_chars").cloned().unwrap_or_else(|| json!(0)),
            "content_bytes": payload.get("content_bytes").cloned().unwrap_or_else(|| json!(0)),
            "created_at": payload.get("created_at").cloned().unwrap_or(Value::Null),
            "content_preview": preview,
        })
        .to_string();
    }

    if let Some(raw_node_id) = args.get("node_id").filter(|v| !v.is_null()) {
        let node = parse_node_id_like(raw_node_id).and_then(|id| get_session_node(cx, id));
        let Some(node) = node else {
            return err(&format!(
                "Node {} not found in current session",
                py_display(raw_node_id)
            ));
        };
        return describe_subtree(cx, &node).to_string();
    }

    // Session DAG overview (`LCM:tools.py:1010-1033`): counts + <=20 node stubs per depth.
    let all_nodes = cx
        .store
        .get_session_nodes(cx.session_id, None, i64::MAX)
        .unwrap_or_default();
    let mut depths: std::collections::BTreeMap<i64, Vec<&SummaryNode>> =
        std::collections::BTreeMap::new();
    for node in &all_nodes {
        depths.entry(node.depth).or_default().push(node);
    }
    let mut depths_obj = Map::new();
    for (depth, nodes) in &depths {
        depths_obj.insert(
            format!("d{depth}"),
            json!({
                "count": nodes.len(),
                "total_tokens": nodes.iter().map(|n| n.token_count).sum::<i64>(),
                "total_source_tokens": nodes.iter().map(|n| n.source_token_count).sum::<i64>(),
                "nodes": nodes.iter().take(20).map(|n| json!({
                    "node_id": n.node_id,
                    "token_count": n.token_count,
                    "expand_hint": n.expand_hint,
                })).collect::<Vec<_>>(),
            }),
        );
    }
    json!({
        "session_id": cx.session_id,
        "store_message_count": cx.store.message_count(cx.session_id).unwrap_or(0),
        "depths": depths_obj,
    })
    .to_string()
}

/// `describe_subtree` (`LCM:dag.py:555-583`): node metadata + child stubs, no content load.
fn describe_subtree(cx: &ToolCx<'_>, node: &SummaryNode) -> Value {
    let children: Vec<Value> = if node.source_type == SourceType::Nodes {
        node.source_ids
            .iter()
            .filter_map(|id| cx.store.get_node(*id).ok().flatten())
            .map(|child| {
                json!({
                    "node_id": child.node_id,
                    "depth": child.depth,
                    "token_count": child.token_count,
                    "source_token_count": child.source_token_count,
                    "expand_hint": child.expand_hint,
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    json!({
        "node_id": node.node_id,
        "depth": node.depth,
        "token_count": node.token_count,
        "source_token_count": node.source_token_count,
        "source_type": node.source_type.as_str(),
        "num_sources": node.source_ids.len(),
        "earliest_at": node.earliest_at,
        "latest_at": node.latest_at,
        "expand_hint": node.expand_hint,
        "children": children,
    })
}

// ---- 10.4 lcm_expand ----------------------------------------------------------------------------

pub(super) fn expand(cx: &ToolCx<'_>, args: &Value) -> String {
    let externalized_ref = py_str_or_empty(args.get("externalized_ref"))
        .trim()
        .to_string();
    let raw_store_id_arg = args.get("store_id").filter(|v| !v.is_null());
    let raw_node_id_arg = args.get("node_id").filter(|v| !v.is_null());

    let mut modes_provided: Vec<&str> = Vec::new();
    if !externalized_ref.is_empty() {
        modes_provided.push("externalized_ref");
    }
    if raw_store_id_arg.is_some() {
        modes_provided.push("store_id");
    }
    if raw_node_id_arg.is_some() {
        modes_provided.push("node_id");
    }
    if modes_provided.len() > 1 {
        return err(&format!(
            "Provide only one of node_id, externalized_ref, store_id (got {})",
            modes_provided.join(", ")
        ));
    }
    if modes_provided.is_empty() {
        return err("node_id, externalized_ref, or store_id is required");
    }

    let max_tokens = parse_positive_int(args.get("max_tokens"), 4000);
    let source_offset = parse_non_negative_int(args.get("source_offset"), 0) as usize;
    let source_limit = args
        .get("source_limit")
        .filter(|v| !v.is_null())
        .map(|v| parse_positive_int(Some(v), 0) as usize);
    let content_offset = parse_non_negative_int(args.get("content_offset"), 0) as usize;

    if !externalized_ref.is_empty() {
        let Some(payload) = get_externalized_payload(cx, &externalized_ref, &[cx.session_id])
        else {
            return err(&format!(
                "Externalized payload {externalized_ref} not found in current session"
            ));
        };
        let content = payload
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let sliced = slice_content_for_response(cx.tokenizer, content, max_tokens, content_offset);
        return json!({
            "externalized_ref": externalized_ref,
            "source_type": "externalized_payload",
            "kind": payload.get("kind").cloned().unwrap_or_else(|| json!("tool_result")),
            "tool_call_id": payload.get("tool_call_id").cloned().unwrap_or_else(|| json!("")),
            "role": payload.get("role").cloned().unwrap_or_else(|| json!("")),
            "session_id": payload.get("session_id").cloned().unwrap_or_else(|| json!("")),
            "field_path": payload.get("field_path").cloned().unwrap_or_else(|| json!("")),
            "content_chars": payload.get("content_chars").cloned()
                .unwrap_or_else(|| json!(content.chars().count())),
            "content_bytes": payload.get("content_bytes").cloned().unwrap_or_else(|| json!(0)),
            "content": sliced.content,
            "content_offset": sliced.content_offset,
            "content_returned_chars": sliced.content_returned_chars,
            "content_truncated": sliced.content_truncated,
            "next_content_offset": sliced.next_content_offset,
            "has_more": sliced.has_more,
        })
        .to_string();
    }

    if let Some(raw_store_id) = raw_store_id_arg {
        return expand_store_id(cx, raw_store_id, max_tokens, content_offset);
    }

    let raw_node_id = raw_node_id_arg.expect("node_id mode was validated above");
    let node = parse_node_id_like(raw_node_id).and_then(|id| get_session_node(cx, id));
    let Some(node) = node else {
        return err(&format!(
            "Node {} not found in current session",
            py_display(raw_node_id)
        ));
    };

    match node.source_type {
        SourceType::Messages => {
            let (messages, pagination) = expand_message_sources(
                cx,
                &node,
                max_tokens,
                source_offset,
                source_limit,
                content_offset,
                false,
            );
            json!({
                "node_id": raw_node_id,
                "depth": node.depth,
                "source_type": "messages",
                "expanded": messages,
                "pagination": pagination,
            })
            .to_string()
        }
        SourceType::Nodes => {
            let (children, pagination) =
                expand_child_nodes(cx, &node, Some(max_tokens), source_offset, source_limit);
            json!({
                "node_id": raw_node_id,
                "depth": node.depth,
                "source_type": "nodes",
                "expanded": children,
                "pagination": pagination,
            })
            .to_string()
        }
    }
}

/// The `store_id` mode of `lcm_expand` (`LCM:tools.py:1108-1178`) — the only cross-session
/// recovery path.
fn expand_store_id(
    cx: &ToolCx<'_>,
    raw_store_id: &Value,
    max_tokens: i64,
    content_offset: usize,
) -> String {
    let Some(store_id) = coerce_int(raw_store_id) else {
        return err("store_id must be an integer");
    };
    let stored = match cx.store.get_message(store_id) {
        Ok(Some(row)) => row,
        Ok(None) => return err(&format!("Message store_id {store_id} not found")),
        Err(e) => return err(&e.to_string()),
    };
    let transcript_content = stored.content.clone().unwrap_or_default();
    let sliced = slice_content_for_response(
        cx.tokenizer,
        &transcript_content,
        max_tokens,
        content_offset,
    );
    let from_current_session = !cx.session_id.is_empty() && stored.session_id == cx.session_id;

    let mut result = Map::new();
    result.insert("store_id".into(), json!(store_id));
    result.insert("source_type".into(), json!("raw_message"));
    result.insert("session_id".into(), json!(stored.session_id));
    result.insert("source".into(), json!(stored.source));
    result.insert("role".into(), json!(stored.role));
    result.insert("timestamp".into(), json!(stored.timestamp));
    result.insert(
        "tool_call_id".into(),
        json!(stored.tool_call_id.as_deref().unwrap_or("")),
    );
    result.insert("from_current_session".into(), json!(from_current_session));
    result.insert("content".into(), json!(sliced.content));
    result.insert("content_chars".into(), json!(sliced.content_chars));
    result.insert("content_offset".into(), json!(sliced.content_offset));
    result.insert(
        "content_returned_chars".into(),
        json!(sliced.content_returned_chars),
    );
    result.insert("content_truncated".into(), json!(sliced.content_truncated));
    result.insert(
        "next_content_offset".into(),
        json!(sliced.next_content_offset),
    );
    result.insert("has_more".into(), json!(sliced.has_more));

    // Surface externalized-payload metadata when the row references one; content is not hydrated
    // (mirroring `_expand_message_sources`' default). Metadata stays session-scoped — a
    // cross-session row surfaces only the ref string (`LCM:tools.py:1137-1177`).
    let mut ref_values: Vec<String> = vec![transcript_content];
    if let Some(raw_tool_calls) = stored.tool_calls.as_deref().filter(|s| !s.is_empty()) {
        // serde_json's default (BTree) map ordering reproduces Python's `sort_keys=True` dump.
        match serde_json::from_str::<Value>(raw_tool_calls) {
            Ok(parsed) if py_truthy(&parsed) => ref_values.push(parsed.to_string()),
            Ok(_) => {}
            Err(_) => ref_values.push(raw_tool_calls.to_string()),
        }
    }
    let mut refs: Vec<String> = Vec::new();
    for value in &ref_values {
        for found in externalize::extract_ingest_refs(value) {
            if !refs.contains(&found) {
                refs.push(found);
            }
        }
        if let Some(legacy) = externalize::extract_ref(value) {
            if !refs.contains(&legacy) {
                refs.push(legacy);
            }
        }
    }
    if !refs.is_empty() {
        result.insert("externalized_refs".into(), json!(refs));
        result.insert("externalized_ref".into(), json!(refs[0]));
        if from_current_session {
            let mut payload_summaries: Vec<Value> = Vec::new();
            for reference in &refs {
                let Some(payload) = get_externalized_payload(cx, reference, &[cx.session_id])
                else {
                    continue;
                };
                let mut summary = payload;
                if let Value::Object(ref mut map) = summary {
                    map.remove("content");
                }
                payload_summaries.push(summary);
            }
            if !payload_summaries.is_empty() {
                result.insert("externalized_payloads".into(), json!(payload_summaries));
                result.insert("externalized".into(), payload_summaries[0].clone());
            }
        } else {
            result.insert(
                "externalized_note".into(),
                json!(
                    "Externalized payload metadata is session-scoped; \
                     cross-session ref is surfaced for traceability only and cannot be expanded in this version."
                ),
            );
        }
    }
    Value::Object(result).to_string()
}
