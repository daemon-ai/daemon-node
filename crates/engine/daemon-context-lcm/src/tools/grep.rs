// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `lcm_grep` (§10.1, `LCM:tools.py:771-971`) and `lcm_load_session` (§10.2,
//! `LCM:tools.py:682-768`).

use super::parse::{
    parse_grep_role, parse_int_value, parse_load_session_roles, parse_optional_float,
    parse_optional_timestamp, parse_strict_int, py_display, py_str_or_empty, py_truthy,
    slice_loaded_content,
};
use super::{err, ToolCx};
use crate::search::{self, SortMode, AGE_DECAY_RATE};
use crate::store::MessageFilter;
use serde_json::{json, Map, Value};
use std::cmp::Ordering;

/// `_LCM_GREP_HARD_LIMIT_CAP` (`LCM:tools.py:194`).
const GREP_HARD_LIMIT_CAP: i64 = 200;
/// `_LCM_LOAD_SESSION_DEFAULT_LIMIT` / `_HARD_LIMIT_CAP` / content-chars bounds
/// (`LCM:tools.py:195-198`).
const LOAD_SESSION_DEFAULT_LIMIT: i64 = 100;
const LOAD_SESSION_HARD_LIMIT_CAP: i64 = 200;
const LOAD_SESSION_DEFAULT_MAX_CONTENT_CHARS: i64 = 4000;
const LOAD_SESSION_HARD_MAX_CONTENT_CHARS: i64 = 20_000;

/// One merged grep hit: the response object plus the transient sort metadata Python carries in
/// `_sort_*` keys (stripped before serialization by construction here).
struct GrepHit {
    value: Map<String, Value>,
    ts: f64,
    rank: f64,
    directness: f64,
    role_bias: f64,
    is_message: bool,
    hybrid_summary_override: f64,
}

/// `_combined_result_sort_key` (`LCM:tools.py:41-70`): the 6-slot lexicographic key per sort mode.
fn combined_sort_key(hit: &GrepHit, sort: SortMode, now: f64) -> [f64; 6] {
    let type_bias = if hit.is_message { 0.0 } else { 1.0 };
    let effective_directness = if hit.is_message {
        hit.directness
    } else {
        hit.directness * 0.8
    };
    match sort {
        SortMode::Relevance => [
            hit.rank,
            -effective_directness,
            hit.role_bias,
            -hit.ts,
            type_bias,
            0.0,
        ],
        SortMode::Hybrid => {
            let age_hours = ((now - hit.ts) / 3600.0).max(0.0);
            let blended = hit.rank / (1.0 + age_hours * AGE_DECAY_RATE);
            [
                -hit.hybrid_summary_override,
                blended,
                -effective_directness,
                hit.role_bias,
                -hit.ts,
                type_bias,
            ]
        }
        SortMode::Recency => {
            if hit.is_message {
                [
                    -hit.ts,
                    type_bias,
                    hit.role_bias,
                    hit.rank,
                    0.0,
                    f64::INFINITY,
                ]
            } else {
                [-hit.ts, type_bias, 0.0, hit.rank, 0.0, hit.role_bias]
            }
        }
    }
}

/// Lexicographic total order over sort keys (`tuple <` in Python; NaN-free by construction).
fn cmp_keys(a: &[f64; 6], b: &[f64; 6]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match x.partial_cmp(y) {
            Some(Ordering::Equal) | None => continue,
            Some(order) => return order,
        }
    }
    Ordering::Equal
}

/// The role bias slot (`LCM:tools.py:47-55`): user 0, assistant 1, tool 2, anything else
/// (including summaries, which carry no role) 1.
fn role_bias(role: Option<&str>) -> f64 {
    match role {
        Some("user") => 0.0,
        Some("assistant") => 1.0,
        Some("tool") => 2.0,
        _ => 1.0,
    }
}

pub(super) fn grep(cx: &ToolCx<'_>, args: &Value) -> String {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if query.is_empty() {
        return err("No query provided");
    }

    let requested_limit = parse_int_value(args.get("limit"), 10);
    if requested_limit <= 0 {
        return err("limit must be a positive integer");
    }
    let limit = requested_limit.min(GREP_HARD_LIMIT_CAP) as usize;
    let sort = SortMode::parse(&py_str_or_empty(args.get("sort")));
    // `source_limit` (`LCM:tools.py:797`): over-fetch each source so the merged sort has slack.
    let source_limit = (limit * 4).max(limit).max(20);

    let requested_session_scope = match args.get("session_scope") {
        None => "current".to_string(),
        Some(v) => py_display(v).to_lowercase(),
    };
    let explicit_session_id = args
        .get("session_id")
        .filter(|v| !v.is_null())
        .map(|v| py_display(v).trim().to_string())
        .unwrap_or_default();
    let source = {
        let s = py_str_or_empty(args.get("source")).trim().to_string();
        (!s.is_empty()).then_some(s)
    };
    let role = match parse_grep_role(args.get("role")) {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    let time_from = match parse_optional_timestamp(args.get("time_from"), "time_from") {
        Ok(t) => t,
        Err(e) => return err(&e),
    };
    let time_to = match parse_optional_timestamp(args.get("time_to"), "time_to") {
        Ok(t) => t,
        Err(e) => return err(&e),
    };
    if let (Some(from), Some(to)) = (time_from, time_to) {
        if to < from {
            return err("time_to must be greater than or equal to time_from");
        }
    }
    // A raw-message filter (role/time bounds) suppresses summary hits — summaries have neither a
    // role nor an exact timestamp (`LCM:tools.py:816`).
    let raw_message_filter_active = role.is_some() || time_from.is_some() || time_to.is_some();

    let (search_session_id, session_scope): (Option<&str>, &str) =
        match requested_session_scope.as_str() {
            "current" => {
                if !explicit_session_id.is_empty() {
                    return err("session_id is only valid with session_scope=session");
                }
                (Some(cx.session_id), "current")
            }
            "all" => {
                if !explicit_session_id.is_empty() {
                    return err("session_id is not used with session_scope=all");
                }
                (None, "all")
            }
            "session" => {
                if explicit_session_id.is_empty() {
                    return err("session_scope=session requires session_id");
                }
                (Some(explicit_session_id.as_str()), "session")
            }
            other => {
                // Preserve historical behavior for unknown scopes: route through the
                // current-session path and report (`LCM:tools.py:845-855`).
                tracing::warn!(scope = other, "lcm: ignoring unsupported session_scope");
                (Some(cx.session_id), "current")
            }
        };

    let current_session_id = cx.session_id;
    let has_current_session = !current_session_id.is_empty();
    let mut hits: Vec<GrepHit> = Vec::new();

    let filter = MessageFilter {
        session: search_session_id,
        role: role.as_deref(),
        source: source.as_deref(),
        time_from,
        time_to,
    };
    match search::search_messages(cx.store, &query, sort, &filter, source_limit) {
        Ok(msg_hits) => {
            for hit in msg_hits {
                let timestamp = hit.row.timestamp;
                let mut value = Map::new();
                value.insert("type".into(), json!("message"));
                value.insert("depth".into(), json!("raw"));
                value.insert("store_id".into(), json!(hit.row.store_id));
                value.insert("session_id".into(), json!(hit.row.session_id));
                value.insert("source".into(), json!(hit.row.source));
                value.insert("role".into(), json!(hit.row.role));
                value.insert("timestamp".into(), json!(timestamp));
                value.insert("snippet".into(), json!(hit.snippet));
                value.insert(
                    "from_current_session".into(),
                    json!(has_current_session && hit.row.session_id == current_session_id),
                );
                hits.push(GrepHit {
                    role_bias: role_bias(Some(&hit.row.role)),
                    value,
                    ts: timestamp,
                    rank: hit.rank,
                    directness: hit.directness,
                    is_message: true,
                    hybrid_summary_override: 0.0,
                });
            }
        }
        Err(e) => tracing::warn!(error = %e, "lcm: message search failed"),
    }

    // Summary-node search is intentionally current-session only: cross-session DAG expansion is
    // deferred, and raw hits stay expandable across sessions via `lcm_expand(store_id=…)`
    // (`LCM:tools.py:893-897`).
    if session_scope == "current" && !raw_message_filter_active {
        match search::search_nodes(
            cx.store,
            &query,
            search_session_id,
            sort,
            source.as_deref(),
            source_limit,
        ) {
            Ok(node_hits) => {
                for hit in node_hits {
                    let node = &hit.node;
                    let ts = node
                        .latest_at
                        .filter(|v| *v != 0.0)
                        .unwrap_or(node.created_at);
                    let snippet: String = node.summary.chars().take(300).collect();
                    let mut value = Map::new();
                    value.insert("type".into(), json!("summary"));
                    value.insert("depth".into(), json!(format!("d{}", node.depth)));
                    value.insert("node_id".into(), json!(node.node_id));
                    value.insert("session_id".into(), json!(node.session_id));
                    value.insert("snippet".into(), json!(snippet));
                    value.insert("token_count".into(), json!(node.token_count));
                    value.insert("expand_hint".into(), json!(node.expand_hint));
                    value.insert("earliest_at".into(), json!(node.earliest_at));
                    value.insert("latest_at".into(), json!(node.latest_at));
                    value.insert("from_current_session".into(), json!(true));
                    hits.push(GrepHit {
                        value,
                        ts,
                        rank: hit.rank,
                        directness: hit.directness,
                        role_bias: role_bias(None),
                        is_message: false,
                        hybrid_summary_override: 0.0,
                    });
                }
            }
            Err(e) => tracing::warn!(error = %e, "lcm: node search failed"),
        }
    }

    if sort == SortMode::Hybrid {
        // A summary that is dramatically more direct than every raw hit jumps the recency-decay
        // blend (`LCM:tools.py:928-935`).
        let max_message_directness = hits
            .iter()
            .filter(|h| h.is_message)
            .map(|h| h.directness)
            .fold(0.0f64, f64::max);
        for hit in &mut hits {
            if !hit.is_message && hit.directness >= max_message_directness + 8.0 {
                hit.hybrid_summary_override = 1.0;
            }
        }
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    hits.sort_by(|a, b| {
        cmp_keys(
            &combined_sort_key(a, sort, now),
            &combined_sort_key(b, sort, now),
        )
    });

    let total_results = hits.len();
    let results: Vec<Value> = hits
        .into_iter()
        .take(limit)
        .map(|h| Value::Object(h.value))
        .collect();

    let mut response = Map::new();
    response.insert("query".into(), json!(query));
    response.insert("sort".into(), json!(sort.as_str()));
    response.insert("session_scope".into(), json!(session_scope));
    response.insert("source".into(), json!(source));
    response.insert("limit".into(), json!(limit));
    response.insert("total_results".into(), json!(total_results));
    response.insert("results".into(), json!(results));
    if let Some(role) = role {
        response.insert("role".into(), json!(role));
    }
    if let Some(from) = time_from {
        response.insert("time_from".into(), json!(from));
    }
    if let Some(to) = time_to {
        response.insert("time_to".into(), json!(to));
    }
    if raw_message_filter_active {
        response.insert("summary_results_omitted".into(), json!(true));
    }
    if session_scope == "session" {
        response.insert("session_id".into(), json!(explicit_session_id));
    }
    if requested_limit > GREP_HARD_LIMIT_CAP {
        response.insert("limit_clamped_from".into(), json!(requested_limit));
    }
    if !matches!(
        requested_session_scope.as_str(),
        "current" | "all" | "session"
    ) {
        response.insert(
            "ignored_session_scope".into(),
            json!(requested_session_scope),
        );
        response.insert(
            "scope_note".into(),
            json!("Unsupported session_scope; stayed on current. Valid values: current, all, session."),
        );
    }
    Value::Object(response).to_string()
}

pub(super) fn load_session(cx: &ToolCx<'_>, args: &Value) -> String {
    let session_id = py_str_or_empty(args.get("session_id")).trim().to_string();
    if session_id.is_empty() {
        return err("session_id is required");
    }

    let requested_limit = match args.get("limit") {
        None => LOAD_SESSION_DEFAULT_LIMIT,
        Some(v) => match parse_strict_int(v, "limit") {
            Ok(n) => n,
            Err(e) => return err(&e),
        },
    };
    if requested_limit <= 0 {
        return err("limit must be a positive integer");
    }
    let limit = requested_limit.min(LOAD_SESSION_HARD_LIMIT_CAP);

    let requested_max_content_chars = match args.get("max_content_chars") {
        None => LOAD_SESSION_DEFAULT_MAX_CONTENT_CHARS,
        Some(v) => match parse_strict_int(v, "max_content_chars") {
            Ok(n) => n,
            Err(e) => return err(&e),
        },
    };
    if requested_max_content_chars <= 0 {
        return err("max_content_chars must be a positive integer");
    }
    let max_content_chars = requested_max_content_chars.min(LOAD_SESSION_HARD_MAX_CONTENT_CHARS);

    let after_store_id = match args.get("after_store_id") {
        None => 0,
        Some(v) => match parse_strict_int(v, "after_store_id") {
            Ok(n) => n,
            Err(e) => return err(&e),
        },
    };
    if after_store_id < 0 {
        return err("after_store_id must be a non-negative integer");
    }

    let roles = match parse_load_session_roles(args.get("roles")) {
        Ok(r) => r,
        Err(e) => return err(&e),
    };
    let time_from = match parse_optional_float(args.get("time_from"), "time_from") {
        Ok(t) => t,
        Err(e) => return err(&e),
    };
    let time_to = match parse_optional_float(args.get("time_to"), "time_to") {
        Ok(t) => t,
        Err(e) => return err(&e),
    };
    if let (Some(from), Some(to)) = (time_from, time_to) {
        if to < from {
            return err("time_to must be greater than or equal to time_from");
        }
    }

    let total_messages =
        match cx
            .store
            .count_session_load_messages(&session_id, &roles, time_from, time_to)
        {
            Ok(n) => n,
            Err(e) => return err(&e.to_string()),
        };
    let rows = match cx.store.load_session_page(
        &session_id,
        after_store_id,
        limit + 1,
        &roles,
        time_from,
        time_to,
    ) {
        Ok(r) => r,
        Err(e) => return err(&e.to_string()),
    };
    let has_more = rows.len() as i64 > limit;
    let page_rows = &rows[..rows.len().min(limit as usize)];
    let next_cursor = if has_more {
        page_rows.last().map(|r| r.store_id)
    } else {
        None
    };

    let messages: Vec<Value> = page_rows
        .iter()
        .map(|row| {
            let content = row.content.as_deref().unwrap_or("");
            let sliced = slice_loaded_content(content, max_content_chars as usize);
            let mut item = Map::new();
            item.insert("store_id".into(), json!(row.store_id));
            item.insert("session_id".into(), json!(row.session_id));
            item.insert("source".into(), json!(row.source));
            item.insert("role".into(), json!(row.role));
            item.insert("timestamp".into(), json!(row.timestamp));
            item.insert("content".into(), json!(sliced.content));
            item.insert("content_chars".into(), json!(sliced.content_chars));
            item.insert(
                "content_returned_chars".into(),
                json!(sliced.content_returned_chars),
            );
            item.insert("content_truncated".into(), json!(sliced.content_truncated));
            item.insert(
                "next_content_offset".into(),
                json!(sliced.next_content_offset),
            );
            item.insert(
                "from_current_session".into(),
                json!(!cx.session_id.is_empty() && row.session_id == cx.session_id),
            );
            if let Some(id) = row.tool_call_id.as_deref().filter(|s| !s.is_empty()) {
                item.insert("tool_call_id".into(), json!(id));
            }
            // The Python row carries `tool_calls` parsed from its JSON column (falling back to the
            // raw string), and serializes it only when truthy (`LCM:store.py:996-1000`).
            if let Some(raw) = row.tool_calls.as_deref().filter(|s| !s.is_empty()) {
                let parsed = serde_json::from_str::<Value>(raw)
                    .unwrap_or_else(|_| Value::String(raw.to_string()));
                if py_truthy(&parsed) {
                    item.insert("tool_calls".into(), parsed);
                }
            }
            if let Some(name) = row.tool_name.as_deref().filter(|s| !s.is_empty()) {
                item.insert("tool_name".into(), json!(name));
            }
            Value::Object(item)
        })
        .collect();

    let mut response = Map::new();
    response.insert("session_id".into(), json!(session_id));
    response.insert("limit".into(), json!(limit));
    response.insert("max_content_chars".into(), json!(max_content_chars));
    response.insert("after_store_id".into(), json!(after_store_id));
    response.insert("total_messages".into(), json!(total_messages));
    response.insert("returned_messages".into(), json!(messages.len()));
    response.insert("messages".into(), json!(messages));
    response.insert("next_cursor".into(), json!(next_cursor));
    response.insert("has_more".into(), json!(has_more));
    if !roles.is_empty() {
        response.insert("roles".into(), json!(roles));
    }
    if let Some(from) = time_from {
        response.insert("time_from".into(), json!(from));
    }
    if let Some(to) = time_to {
        response.insert("time_to".into(), json!(to));
    }
    if requested_limit > LOAD_SESSION_HARD_LIMIT_CAP {
        response.insert("limit_clamped_from".into(), json!(requested_limit));
    }
    if requested_max_content_chars > LOAD_SESSION_HARD_MAX_CONTENT_CHARS {
        response.insert(
            "max_content_chars_clamped_from".into(),
            json!(requested_max_content_chars),
        );
    }
    Value::Object(response).to_string()
}
