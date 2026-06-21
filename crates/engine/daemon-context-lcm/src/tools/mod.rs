//! The seven `lcm_*` drill-down tools (§10).
//!
//! Ported from `LCM:tools.py`. Each handler takes parsed JSON args and returns a JSON string. They
//! are `ContextEngine`-owned (not in the main `ToolRegistry`): the host registers thin adapters that
//! resolve the calling session's [`LcmContextEngine`](crate::LcmContextEngine) and delegate to
//! [`LcmContextEngine::call_tool`], which builds a [`ToolCx`] and calls [`dispatch`].
//!
//! Scope divergence from the Python plugin (intentional, §14): tools read the **durable store** (the
//! full per-turn transcript ingested in `before_turn`) rather than the live `Conversation`, so a
//! `store_id`/`node_id` recovers exact content regardless of what is currently in-context.

pub mod schemas;

use crate::config::LcmConfig;
use crate::search::{self, SortMode};
use crate::store::{MessageFilter, Store};
use crate::tokens::Tokenizer;
use daemon_core::tools::ToolDef;
use daemon_core::{Provider, Request, RequestMsg};
use serde_json::{json, Value};
use std::time::Duration;

/// The stable names of the seven tools (the order `tool_defs` / `ContextEngine::tools` report).
pub const TOOL_NAMES: [&str; 7] = [
    "lcm_grep",
    "lcm_load_session",
    "lcm_describe",
    "lcm_expand",
    "lcm_expand_query",
    "lcm_status",
    "lcm_doctor",
];

/// The §12 [`ToolDef`]s for all seven tools (session-independent — enumerate once).
pub fn tool_defs() -> Vec<ToolDef> {
    let schemas = [
        schemas::LCM_GREP,
        schemas::LCM_LOAD_SESSION,
        schemas::LCM_DESCRIBE,
        schemas::LCM_EXPAND,
        schemas::LCM_EXPAND_QUERY,
        schemas::LCM_STATUS,
        schemas::LCM_DOCTOR,
    ];
    TOOL_NAMES
        .iter()
        .zip(schemas)
        .map(|(name, schema)| ToolDef {
            name: (*name).to_string(),
            schema: schema.to_string(),
        })
        .collect()
}

/// Everything a tool handler needs, assembled per-call by [`LcmContextEngine::call_tool`].
pub(crate) struct ToolCx<'a> {
    /// The bank store (shared; row-scoped by session).
    pub store: &'a Store,
    /// The engine config (thresholds, fresh-tail count, paths).
    pub config: &'a LcmConfig,
    /// The model-aware token counter.
    pub tokenizer: &'a Tokenizer,
    /// The auxiliary provider (for `lcm_expand_query`).
    pub aux: &'a dyn Provider,
    /// The foreground session id (the §14.1 identity invariant).
    pub session_id: &'a str,
    /// The model-window-derived compaction threshold, if known (status/doctor).
    pub threshold_tokens: Option<usize>,
    /// How many compactions have run this incarnation (status).
    pub compaction_count: u64,
}

/// Dispatch one `lcm_*` tool by name, returning a JSON string (§10.7).
pub(crate) async fn dispatch(cx: &ToolCx<'_>, name: &str, args: Value) -> String {
    match name {
        "lcm_grep" => grep(cx, &args),
        "lcm_load_session" => load_session(cx, &args),
        "lcm_describe" => describe(cx, &args),
        "lcm_expand" => expand(cx, &args),
        "lcm_expand_query" => expand_query(cx, &args).await,
        "lcm_status" => status(cx),
        "lcm_doctor" => doctor(cx),
        other => json!({"status": "unknown_tool", "tool": other}).to_string(),
    }
}

// ---- 10.1 lcm_grep ------------------------------------------------------------------------------

fn grep(cx: &ToolCx<'_>, args: &Value) -> String {
    let query = args.get("query").and_then(Value::as_str).unwrap_or("").trim();
    if query.is_empty() {
        return err("query is required");
    }
    let limit = arg_u64(args, "limit", 10).clamp(1, 200) as usize;
    let sort = SortMode::parse(args.get("sort").and_then(Value::as_str).unwrap_or("recency"));
    let scope = args
        .get("session_scope")
        .and_then(Value::as_str)
        .unwrap_or("current");
    let role = args.get("role").and_then(Value::as_str);
    let source = args.get("source").and_then(Value::as_str);
    let time_from = args.get("time_from").and_then(Value::as_f64);
    let time_to = args.get("time_to").and_then(Value::as_f64);
    let explicit_session = args.get("session_id").and_then(Value::as_str);

    // A raw filter (role/time) suppresses summary hits (they have no role/exact timestamp) — §10.1.
    let raw_filter_set = role.is_some() || time_from.is_some() || time_to.is_some();

    // `current` searches the foreground session (summaries included); `all`/`session` are raw-only.
    let (session_filter, scope_label, summaries_allowed) = match scope {
        "all" => (None, "all", false),
        "session" => match explicit_session {
            Some(s) => (Some(s), "session", false),
            None => return err("session_id is required when session_scope=session"),
        },
        _ => (Some(cx.session_id), "current", true),
    };

    let filter = MessageFilter {
        session: session_filter,
        role,
        source,
        time_from,
        time_to,
    };
    let messages =
        search::search_messages(cx.store, query, sort, &filter, limit).unwrap_or_default();
    let mut results: Vec<Value> = messages
        .iter()
        .map(|m| {
            json!({
                "type": "message",
                "depth": "raw",
                "store_id": m.row.store_id,
                "session_id": m.row.session_id,
                "role": m.row.role,
                "source": normalize_source(&m.row.source),
                "timestamp": m.row.timestamp,
                "snippet": m.snippet,
            })
        })
        .collect();

    let summary_omitted = summaries_allowed && raw_filter_set;
    if summaries_allowed && !raw_filter_set {
        let nodes = search::search_nodes(cx.store, query, cx.session_id, limit).unwrap_or_default();
        for n in &nodes {
            results.push(json!({
                "type": "summary",
                "depth": format!("d{}", n.node.depth),
                "node_id": n.node.node_id,
                "session_id": n.node.session_id,
                "snippet": truncate_chars(&n.node.summary, 200),
                "expand_hint": n.node.expand_hint,
            }));
        }
    }

    json!({
        "query": query,
        "sort": sort.as_str(),
        "session_scope": scope_label,
        "source": source,
        "limit": limit,
        "total_results": results.len(),
        "results": results,
        "summary_results_omitted": summary_omitted,
    })
    .to_string()
}

// ---- 10.2 lcm_load_session ----------------------------------------------------------------------

fn load_session(cx: &ToolCx<'_>, args: &Value) -> String {
    let session = args
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or(cx.session_id);
    let limit = arg_u64(args, "limit", 100).clamp(1, 200) as i64;
    let max_chars = arg_u64(args, "max_content_chars", 4000).clamp(1, 20000) as usize;
    let after = args.get("after_store_id").and_then(Value::as_i64);

    // Fetch one extra row to detect `has_more` without a second query (§10.2).
    let mut rows = match cx.store.load_session_page(session, after, limit + 1) {
        Ok(r) => r,
        Err(e) => return err(&e.to_string()),
    };
    let has_more = rows.len() as i64 > limit;
    if has_more {
        rows.truncate(limit as usize);
    }
    let next_cursor = rows.last().map(|r| r.store_id);

    let messages: Vec<Value> = rows
        .iter()
        .map(|r| {
            let full = r.content.as_deref().unwrap_or("");
            let (content, truncated) = truncate_chars_flagged(full, max_chars);
            json!({
                "store_id": r.store_id,
                "role": r.role,
                "source": normalize_source(&r.source),
                "timestamp": r.timestamp,
                "content": content,
                "content_truncated": truncated,
                "next_content_offset": if truncated { Some(max_chars) } else { None },
            })
        })
        .collect();

    json!({
        "session_id": session,
        "limit": limit,
        "returned": messages.len(),
        "has_more": has_more,
        "next_cursor": next_cursor,
        "messages": messages,
    })
    .to_string()
}

// ---- 10.3 lcm_describe --------------------------------------------------------------------------

fn describe(cx: &ToolCx<'_>, args: &Value) -> String {
    if let Some(node_id) = args.get("node_id").and_then(Value::as_i64) {
        return describe_node(cx, node_id);
    }
    // No-arg: a per-depth session DAG overview (counts + <=20 node stubs/depth) — §10.3.
    let mut depths = Vec::new();
    let mut depth = 0i64;
    loop {
        let count = cx.store.count_at_depth(cx.session_id, depth).unwrap_or(0);
        if count == 0 {
            // Allow a single gap (defensive), then stop.
            if depth > 0 && cx.store.count_at_depth(cx.session_id, depth + 1).unwrap_or(0) == 0 {
                break;
            }
            if depth == 0 {
                break;
            }
        }
        let stubs = cx
            .store
            .get_session_nodes(cx.session_id, Some(depth), 20)
            .unwrap_or_default();
        depths.push(json!({
            "depth": depth,
            "count": count,
            "nodes": stubs.iter().map(node_stub).collect::<Vec<_>>(),
        }));
        depth += 1;
        if depth > 16 {
            break;
        }
    }
    json!({
        "session_id": cx.session_id,
        "total_nodes": cx.store.summary_count(cx.session_id).unwrap_or(0),
        "depths": depths,
    })
    .to_string()
}

fn describe_node(cx: &ToolCx<'_>, node_id: i64) -> String {
    let node = match cx.store.get_node(node_id) {
        Ok(Some(n)) => n,
        Ok(None) => return err(&format!("node {node_id} not found")),
        Err(e) => return err(&e.to_string()),
    };
    // Subtree stub: the node's metadata + a stub for each direct child (no content load) — §5.4.
    let children: Vec<Value> = if node.source_type == crate::SourceType::Nodes {
        node.source_ids
            .iter()
            .filter_map(|id| cx.store.get_node(*id).ok().flatten())
            .map(|c| node_stub(&c))
            .collect()
    } else {
        Vec::new()
    };
    json!({
        "node_id": node.node_id,
        "depth": node.depth,
        "source_type": node.source_type.as_str(),
        "source_count": node.source_ids.len(),
        "token_count": node.token_count,
        "source_token_count": node.source_token_count,
        "earliest_at": node.earliest_at,
        "latest_at": node.latest_at,
        "expand_hint": node.expand_hint,
        "preview": truncate_chars(&node.summary, 500),
        "children": children,
    })
    .to_string()
}

// ---- 10.4 lcm_expand ----------------------------------------------------------------------------

fn expand(cx: &ToolCx<'_>, args: &Value) -> String {
    let max_tokens = arg_u64(args, "max_tokens", 4000).max(1) as usize;
    let content_offset = arg_u64(args, "content_offset", 0) as usize;

    if let Some(store_id) = args.get("store_id").and_then(Value::as_i64) {
        return expand_store_id(cx, store_id, max_tokens, content_offset);
    }
    if let Some(node_id) = args.get("node_id").and_then(Value::as_i64) {
        return expand_node(cx, node_id, args, max_tokens, content_offset);
    }
    err("exactly one of node_id or store_id is required")
}

fn expand_store_id(
    cx: &ToolCx<'_>,
    store_id: i64,
    max_tokens: usize,
    content_offset: usize,
) -> String {
    let row = match cx.store.get_message(store_id) {
        Ok(Some(r)) => r,
        Ok(None) => return err(&format!("store_id {store_id} not found")),
        Err(e) => return err(&e.to_string()),
    };
    let full = row.content.clone().unwrap_or_default();
    let sliced = slice_chars_from(&full, content_offset);
    let (content, next_content_offset) = truncate_to_token_budget(cx.tokenizer, &sliced, max_tokens);
    let consumed = content.chars().count();
    json!({
        "type": "message",
        "store_id": row.store_id,
        "session_id": row.session_id,
        "role": row.role,
        "source": normalize_source(&row.source),
        "timestamp": row.timestamp,
        "content": content,
        "pagination": {
            "content_offset": content_offset,
            "next_content_offset": next_content_offset.map(|n| content_offset + n),
            "has_more": next_content_offset.is_some(),
            "returned_chars": consumed,
        },
    })
    .to_string()
}

fn expand_node(
    cx: &ToolCx<'_>,
    node_id: i64,
    args: &Value,
    max_tokens: usize,
    content_offset: usize,
) -> String {
    let node = match cx.store.get_node(node_id) {
        Ok(Some(n)) => n,
        Ok(None) => return err(&format!("node {node_id} not found")),
        Err(e) => return err(&e.to_string()),
    };
    // Current-session only for node mode (§10.4 / §14.1).
    if node.session_id != cx.session_id {
        return err("node belongs to another session; use store_id for cross-session recovery");
    }

    let total_sources = node.source_ids.len();
    let source_offset = arg_u64(args, "source_offset", 0) as usize;
    let source_limit = args
        .get("source_limit")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(total_sources);
    let page_ids: Vec<i64> = node
        .source_ids
        .iter()
        .skip(source_offset)
        .take(source_limit.max(1))
        .copied()
        .collect();

    // For a leaf (messages) node we recover real rows; for a condensation we recover child summaries.
    let blocks: Vec<(String, String)> = if node.source_type == crate::SourceType::Messages {
        cx.store
            .get_messages(&page_ids)
            .unwrap_or_default()
            .into_iter()
            .map(|r| {
                (
                    r.role.clone(),
                    r.content.clone().unwrap_or_default(),
                )
            })
            .collect()
    } else {
        page_ids
            .iter()
            .filter_map(|id| cx.store.get_node(*id).ok().flatten())
            .map(|c| (format!("d{}", c.depth), c.summary))
            .collect()
    };

    // Share the token budget across sources; paginate within the first source via content_offset.
    let mut remaining = max_tokens;
    let mut rendered: Vec<Value> = Vec::new();
    let mut next_content_offset: Option<usize> = None;
    let mut consumed_sources = 0usize;
    for (i, (label, body)) in blocks.iter().enumerate() {
        if remaining == 0 {
            break;
        }
        let body = if i == 0 && content_offset > 0 {
            slice_chars_from(body, content_offset)
        } else {
            body.clone()
        };
        let (content, more) = truncate_to_token_budget(cx.tokenizer, &body, remaining);
        let used = cx.tokenizer.count_text(&content);
        remaining = remaining.saturating_sub(used);
        if let Some(more_at) = more {
            // This source overflowed the shared budget — record where to resume and stop.
            next_content_offset = Some(if i == 0 { content_offset + more_at } else { more_at });
            rendered.push(json!({"role": label, "content": content}));
            consumed_sources += 1;
            break;
        }
        rendered.push(json!({"role": label, "content": content}));
        consumed_sources += 1;
    }

    let returned_sources = consumed_sources;
    let next_source_offset = source_offset + returned_sources;
    let has_more = next_content_offset.is_some() || next_source_offset < total_sources;
    json!({
        "type": "node_expansion",
        "node_id": node.node_id,
        "depth": node.depth,
        "source_type": node.source_type.as_str(),
        "sources": rendered,
        "pagination": {
            "source_offset": source_offset,
            "content_offset": content_offset,
            "returned_sources": returned_sources,
            "total_sources": total_sources,
            "next_source_offset": if next_source_offset < total_sources { Some(next_source_offset) } else { None },
            "next_content_offset": next_content_offset,
            "has_more": has_more,
            "remaining_sources": total_sources.saturating_sub(next_source_offset),
        },
    })
    .to_string()
}

// ---- 10.5 lcm_expand_query ----------------------------------------------------------------------

async fn expand_query(cx: &ToolCx<'_>, args: &Value) -> String {
    let prompt = args.get("prompt").and_then(Value::as_str).unwrap_or("").trim();
    if prompt.is_empty() {
        return err("prompt is required");
    }
    let max_results = arg_u64(args, "max_results", 5).max(1) as usize;
    let answer_budget = arg_u64(args, "max_tokens", 2000).max(1) as usize;
    let context_budget = arg_u64(args, "context_max_tokens", 32000).max(1) as usize;

    // Candidate selection: explicit node_ids, else a search over this session's summaries.
    let candidates: Vec<crate::SummaryNode> =
        if let Some(ids) = args.get("node_ids").and_then(Value::as_array) {
            ids.iter()
                .filter_map(Value::as_i64)
                .filter_map(|id| cx.store.get_node(id).ok().flatten())
                .filter(|n| n.session_id == cx.session_id)
                .take(max_results)
                .collect()
        } else {
            let q = args.get("query").and_then(Value::as_str).unwrap_or(prompt);
            search::search_nodes(cx.store, q, cx.session_id, max_results)
                .unwrap_or_default()
                .into_iter()
                .map(|r| r.node)
                .collect()
        };

    // Build a bounded context from the candidate summaries.
    let mut context = String::new();
    let mut used = 0usize;
    let mut included = 0usize;
    for node in &candidates {
        let block = format!("[d{} node {}]\n{}\n\n", node.depth, node.node_id, node.summary);
        let cost = cx.tokenizer.count_text(&block);
        if used + cost > context_budget && included > 0 {
            break;
        }
        context.push_str(&block);
        used += cost;
        included += 1;
    }

    if context.is_empty() {
        return json!({
            "status": "degraded",
            "reason": "no_matching_context",
            "answer": "",
            "context_pagination": {"hint": "use lcm_grep then lcm_expand(node_id=…) to recover detail"},
        })
        .to_string();
    }

    let request = Request {
        system: format!(
            "Answer the user's question using ONLY the recovered context below. Be concise \
             (~{answer_budget} tokens). If the answer isn't present, say so.\n\nCONTEXT:\n{context}"
        ),
        messages: vec![RequestMsg {
            role: "user".into(),
            content: prompt.to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };
    match tokio::time::timeout(
        Duration::from_millis(cx.config.summary_timeout_ms),
        cx.aux.chat(request),
    )
    .await
    {
        Ok(Ok(out)) if !out.text.trim().is_empty() => json!({
            "status": "ok",
            "answer": out.text,
            "nodes_used": included,
            "context_tokens": used,
        })
        .to_string(),
        _ => json!({
            "status": "degraded",
            "reason": "aux_unavailable_or_empty",
            "answer": "",
            "nodes_used": included,
            "context_pagination": {"hint": "expand the listed nodes manually via lcm_expand(node_id=…)"},
        })
        .to_string(),
    }
}

// ---- 10.6 lcm_status / lcm_doctor ---------------------------------------------------------------

fn status(cx: &ToolCx<'_>) -> String {
    let counts = cx.store.table_counts().unwrap_or_default();
    let frontier = cx.store.get_frontier(cx.session_id).unwrap_or(0);
    json!({
        "session_id": cx.session_id,
        "compaction_count": cx.compaction_count,
        "threshold_tokens": cx.threshold_tokens,
        "frontier_store_id": frontier,
        "store": {
            "messages": counts.messages,
            "summary_nodes": counts.nodes,
            "session_messages": cx.store.message_count(cx.session_id).unwrap_or(0),
            "session_summaries": cx.store.summary_count(cx.session_id).unwrap_or(0),
        },
        "config": {
            "bank": cx.config.bank,
            "context_threshold": cx.config.context_threshold,
            "fresh_tail_count": cx.config.fresh_tail_count,
            "condensation_fanin": cx.config.condensation_fanin,
            "incremental_max_depth": cx.config.incremental_max_depth,
        },
    })
    .to_string()
}

fn doctor(cx: &ToolCx<'_>) -> String {
    let mut checks = Vec::new();
    let mut worst = Health::Healthy;

    // database_integrity
    let integrity = cx.store.integrity_check().unwrap_or_else(|e| e.to_string());
    let ok = integrity == "ok";
    push_check(&mut checks, &mut worst, "database_integrity", ok, Health::Unhealthy, &integrity);

    // fts_index_sync (both shadows must match their base table)
    let counts = cx.store.table_counts().unwrap_or_default();
    let msg_sync = counts.messages == counts.messages_fts;
    let node_sync = counts.nodes == counts.nodes_fts;
    push_check(
        &mut checks,
        &mut worst,
        "messages_fts_integrity",
        msg_sync,
        Health::Warnings,
        &format!("messages={} fts={}", counts.messages, counts.messages_fts),
    );
    push_check(
        &mut checks,
        &mut worst,
        "nodes_fts_integrity",
        node_sync,
        Health::Warnings,
        &format!("nodes={} fts={}", counts.nodes, counts.nodes_fts),
    );

    // orphaned_dag_nodes
    let orphans = cx.store.orphaned_node_count().unwrap_or(0);
    push_check(
        &mut checks,
        &mut worst,
        "orphaned_dag_nodes",
        orphans == 0,
        Health::Warnings,
        &format!("{orphans} orphaned child references"),
    );

    json!({
        "overall": worst.as_str(),
        "runtime_identity": {"session_id": cx.session_id, "bank": cx.config.bank},
        "checks": checks,
        "skipped": ["ingest_protection", "sensitive_pattern_handling", "payload_storage"],
    })
    .to_string()
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Health {
    Healthy,
    Warnings,
    Unhealthy,
}

impl Health {
    fn as_str(self) -> &'static str {
        match self {
            Health::Healthy => "healthy",
            Health::Warnings => "warnings",
            Health::Unhealthy => "unhealthy",
        }
    }
}

fn push_check(
    checks: &mut Vec<Value>,
    worst: &mut Health,
    name: &str,
    ok: bool,
    fail_level: Health,
    detail: &str,
) {
    let status = if ok {
        "ok"
    } else {
        *worst = (*worst).max(fail_level);
        if fail_level == Health::Unhealthy {
            "fail"
        } else {
            "warn"
        }
    };
    checks.push(json!({"check": name, "status": status, "detail": detail}));
}

// ---- helpers ------------------------------------------------------------------------------------

fn err(message: &str) -> String {
    json!({"status": "error", "error": message}).to_string()
}

fn arg_u64(args: &Value, key: &str, default: u64) -> u64 {
    args.get(key).and_then(Value::as_u64).unwrap_or(default)
}

fn normalize_source(source: &str) -> &str {
    if source.is_empty() {
        "unknown"
    } else {
        source
    }
}

fn node_stub(node: &crate::SummaryNode) -> Value {
    json!({
        "node_id": node.node_id,
        "depth": node.depth,
        "token_count": node.token_count,
        "source_count": node.source_ids.len(),
        "expand_hint": node.expand_hint,
    })
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    truncate_chars_flagged(s, max_chars).0
}

/// Truncate to `max_chars` characters, returning `(text, was_truncated)`.
fn truncate_chars_flagged(s: &str, max_chars: usize) -> (String, bool) {
    if s.chars().count() <= max_chars {
        (s.to_string(), false)
    } else {
        (s.chars().take(max_chars).collect(), true)
    }
}

/// Return the substring starting at character offset `offset`.
fn slice_chars_from(s: &str, offset: usize) -> String {
    s.chars().skip(offset).collect()
}

/// Largest character prefix of `text` whose token count fits `max_tokens` (§14.6 binary search).
/// Returns `(prefix, next_char_offset)`; `next_char_offset` is `Some` only when truncation occurred.
fn truncate_to_token_budget(
    tok: &Tokenizer,
    text: &str,
    max_tokens: usize,
) -> (String, Option<usize>) {
    if tok.count_text(text) <= max_tokens {
        return (text.to_string(), None);
    }
    let chars: Vec<char> = text.chars().collect();
    let (mut lo, mut hi) = (0usize, chars.len());
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        let candidate: String = chars[..mid].iter().collect();
        if tok.count_text(&candidate) <= max_tokens {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let prefix: String = chars[..lo].iter().collect();
    (prefix, Some(lo))
}

#[cfg(test)]
mod tests;
