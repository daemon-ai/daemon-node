// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-semantic-search` — the `semantic_search` chat tool (`daemon_core::Tool`): embedding
//! based retrieval over the workspace, backed by an injected [`WorkspaceIndex`] (Cursor
//! `SemanticSearch` parity). **No LLM calls in the tool** — it embeds the query through the shared
//! embedder and returns ranked file chunks (`path:line-line` + text).
//!
//! ## Session containment (mandatory)
//!
//! The index is rooted at the node's `workspace_root`, which — in an isolated-session deployment —
//! is the *parent* of every session's sandbox (`<root>/<session_id>`). The tool therefore computes
//! the calling session's cwd relative to that root and uses it as an implicit directory filter, so a
//! session can only ever see chunks under its own subtree. Model-supplied `target_directories`
//! narrow *within* that subtree; a `target_directory` that would escape it yields no results. In a
//! bound-repo deployment the session cwd equals the index root, so the filter is a no-op.

#![forbid(unsafe_code)]

use std::sync::Arc;

use async_trait::async_trait;
use daemon_core::{Tool, ToolCall, ToolConcurrency, ToolOutcome, TurnCx};
use daemon_workspace_index::{IndexHit, WorkspaceIndex};
use serde::Deserialize;

/// The default (and maximum) number of results returned.
const DEFAULT_RESULTS: i64 = 15;
/// The minimum clamp for `num_results`.
const MIN_RESULTS: i64 = 1;
/// The maximum clamp for `num_results`.
const MAX_RESULTS: i64 = 15;
/// The fallback result byte budget when the turn does not specify one.
const DEFAULT_BUDGET: usize = 8192;

/// The JSON-Schema advertised for the `semantic_search` tool.
const SEMANTIC_SEARCH_SCHEMA: &str = r#"{
  "type": "object",
  "description": "Semantic (embedding-based) search over the current workspace's code and text files. Finds relevant code by MEANING, not exact text - reach for this when you know what you want conceptually ('where is auth handled', 'the retry backoff logic') but not the exact symbol or string. For exact strings/symbols prefer a grep/search tool instead. Results are ranked file chunks, each with a `path:start-end` line span and the chunk text. The search is scoped to your session's workspace; pass target_directories to narrow to specific subdirectories.",
  "properties": {
    "query": {
      "type": "string",
      "description": "What to search for, in natural language. Describe the concept, behavior, or role of the code you want (e.g. 'jwt token validation', 'where retries are scheduled'). A complete phrase works better than a single keyword."
    },
    "target_directories": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Optional workspace-relative directory prefixes to restrict the search to (e.g. [\"src/auth\"]). Omit to search the whole workspace. A directory that escapes your session workspace is ignored."
    },
    "num_results": {
      "type": "integer",
      "description": "Max results to return (default 15, clamped to [1,15]).",
      "default": 15
    }
  },
  "required": ["query"]
}"#;

/// The `semantic_search` tool over an injected [`WorkspaceIndex`].
pub struct SemanticSearchTool {
    index: Arc<WorkspaceIndex>,
}

impl SemanticSearchTool {
    /// A semantic-search tool over `index`.
    pub fn new(index: Arc<WorkspaceIndex>) -> Self {
        Self { index }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Args {
    query: Option<String>,
    target_directories: Option<Vec<String>>,
    num_results: Option<i64>,
}

#[async_trait]
impl Tool for SemanticSearchTool {
    fn name(&self) -> &str {
        "semantic_search"
    }

    fn schema(&self) -> &str {
        SEMANTIC_SEARCH_SCHEMA
    }

    /// Read-only over the index: safe to run alongside other parallel calls in a batch.
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let call_id = call.call_id.clone();
        let args: Args = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call_id,
                    false,
                    format!("semantic_search: invalid arguments: {e}"),
                )
            }
        };

        // Building state: never an error — tell the model to retry shortly.
        if !self.index.ready() {
            return ToolOutcome::text(
                call_id,
                true,
                "semantic index is still building; retry shortly",
            );
        }

        let query = args.query.as_deref().map(str::trim).unwrap_or_default();
        if query.is_empty() {
            return ToolOutcome::text(
                call_id,
                true,
                "semantic_search: provide a `query` describing what to find.",
            );
        }
        let k = clamp(
            args.num_results.unwrap_or(DEFAULT_RESULTS),
            MIN_RESULTS,
            MAX_RESULTS,
        ) as usize;

        // MANDATORY session-cwd containment: the session may only see chunks under its own subtree.
        let base =
            match self.base_filter(cx) {
                Some(base) => base,
                None => return ToolOutcome::text(
                    call_id,
                    true,
                    "semantic_search: the session workspace is outside the index root; no results.",
                ),
            };
        let targets = args.target_directories.unwrap_or_default();
        let dir_filters: Vec<String> = if targets.is_empty() {
            vec![base]
        } else {
            let resolved: Vec<String> = targets
                .iter()
                .filter_map(|t| resolve_within(&base, t))
                .collect();
            if resolved.is_empty() {
                // Every requested directory escaped the session subtree — nothing to search.
                return ToolOutcome::text(
                    call_id,
                    true,
                    "semantic_search: no results (requested directories are outside your workspace).",
                );
            }
            resolved
        };

        let hits = self.index.query(query, k, &dir_filters).await;
        ToolOutcome::text(call_id, true, present(query, &hits, cx.tool_result_budget))
    }
}

impl SemanticSearchTool {
    /// The implicit directory filter: the calling session's cwd relative to the index root
    /// (`""` when cwd == root). `None` when the cwd is not under the index root (containment cannot
    /// be expressed, so the caller returns no results).
    fn base_filter(&self, cx: &TurnCx<'_>) -> Option<String> {
        let root = self.index.workspace_root();
        let cwd = cx.exec.cwd();
        let rel = cwd.strip_prefix(root).ok()?;
        Some(
            rel.components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/"),
        )
    }
}

/// Join `target` onto `base` and normalize, returning `None` if it would climb above `base` (an
/// escape attempt). Both are `/`-separated, root-relative directory prefixes.
fn resolve_within(base: &str, target: &str) -> Option<String> {
    let mut segs: Vec<String> = base
        .split('/')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let base_len = segs.len();
    for part in target.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => {
                if segs.len() > base_len {
                    segs.pop();
                } else {
                    return None;
                }
            }
            s => segs.push(s.to_string()),
        }
    }
    Some(segs.join("/"))
}

/// Render ranked hits (Cursor parity): full chunk text for the top results (until the result byte
/// budget is spent), then one-line `path:start-end (score)` summaries for the remainder.
fn present(query: &str, hits: &[IndexHit], budget: usize) -> String {
    if hits.is_empty() {
        return format!("No semantic matches found for \"{query}\".");
    }
    let budget = if budget == 0 { DEFAULT_BUDGET } else { budget };
    let mut out = format!("{} result(s) for \"{query}\":\n\n", hits.len());
    for hit in hits {
        let header = format!("{}:{}-{}", hit.path, hit.start_line, hit.end_line);
        if out.len() < budget {
            out.push_str(&header);
            out.push('\n');
            out.push_str(&hit.snippet);
            if !hit.snippet.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
        } else {
            out.push_str(&format!("{header} ({:.2})\n", hit.score));
        }
    }
    out
}

/// Clamp an i64 into `[lo, hi]`.
fn clamp(value: i64, lo: i64, hi: i64) -> i64 {
    value.max(lo).min(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_within_narrows_and_rejects_escapes() {
        assert_eq!(resolve_within("", "src").as_deref(), Some("src"));
        assert_eq!(resolve_within("", ".").as_deref(), Some(""));
        assert_eq!(
            resolve_within("ses1", "src/auth").as_deref(),
            Some("ses1/src/auth")
        );
        assert_eq!(
            resolve_within("ses1", "sub/../other").as_deref(),
            Some("ses1/other")
        );
        // Climbing above the base is rejected.
        assert_eq!(resolve_within("ses1", ".."), None);
        assert_eq!(resolve_within("", ".."), None);
        assert_eq!(resolve_within("ses1", "../ses2"), None);
    }

    #[test]
    fn present_formats_headers_and_empty() {
        assert!(present("q", &[], 0).contains("No semantic matches"));
        let hits = vec![IndexHit {
            path: "src/a.rs".into(),
            start_line: 3,
            end_line: 7,
            score: 0.9,
            snippet: "fn a() {}".into(),
        }];
        let out = present("q", &hits, 0);
        assert!(out.contains("src/a.rs:3-7"));
        assert!(out.contains("fn a() {}"));
    }

    #[test]
    fn present_switches_to_compact_when_budget_exhausted() {
        let hits: Vec<IndexHit> = (0..3)
            .map(|i| IndexHit {
                path: format!("f{i}.rs"),
                start_line: 1,
                end_line: 2,
                score: 0.5,
                snippet: "x".repeat(50),
            })
            .collect();
        // A tiny budget forces the compact one-liner form for later hits.
        let out = present("q", &hits, 20);
        assert!(out.contains("(0.50)"), "compact score line expected: {out}");
    }
}
