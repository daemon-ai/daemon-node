// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! JSON-schema strings for the seven `lcm_*` drill-down tools (§10.7).
//!
//! Ported from `LCM:schemas.py`. Each is the `ToolDef::schema` advertised to the model. Kept as
//! string constants (not built at runtime) so they are cheap to enumerate and easy to diff.

/// `lcm_grep` — search the transcript + summaries (§10.1).
pub const LCM_GREP: &str = r#"{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "FTS5 query; quotes for phrases (AND by default; prefer 1-3 distinctive terms)"},
    "limit": {"type": "integer", "default": 10, "description": "Max results (hard cap 200; clamps report limit_clamped_from)"},
    "sort": {"type": "string", "enum": ["recency", "relevance", "hybrid"], "default": "recency"},
    "session_scope": {"type": "string", "enum": ["current", "all", "session"], "default": "current", "description": "Broader scopes return raw-message hits only"},
    "session_id": {"type": "string", "description": "Required iff session_scope=session; invalid otherwise"},
    "role": {"type": "string", "enum": ["system", "user", "assistant", "tool", "unknown"], "description": "Raw-message role filter; suppresses summary hits"},
    "source": {"type": "string", "description": "Source/platform filter (raw rows + summaries via lineage); 'unknown' for unattributed content"},
    "time_from": {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "Inclusive lower bound: Unix seconds or timezone-aware ISO 8601 (naive rejected); suppresses summary hits"},
    "time_to": {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "Inclusive upper bound: Unix seconds or timezone-aware ISO 8601 (naive rejected); suppresses summary hits"}
  },
  "required": ["query"]
}"#;

/// `lcm_load_session` — ordered raw-message page for one explicit session (§10.2).
pub const LCM_LOAD_SESSION: &str = r#"{
  "type": "object",
  "properties": {
    "session_id": {"type": "string", "description": "Explicit LCM session id to load (required; no current/all fallback)"},
    "limit": {"type": "integer", "default": 100, "description": "Max raw messages (hard cap 200; clamps report limit_clamped_from)"},
    "max_content_chars": {"type": "integer", "default": 4000, "description": "Per-message content cap (hard cap 20000); recover full rows via lcm_expand(store_id=...)"},
    "after_store_id": {"type": "integer", "default": 0, "description": "Exclusive cursor; pass the previous page's next_cursor"},
    "roles": {"type": "array", "items": {"type": "string"}, "description": "Optional role filter, e.g. ['user', 'assistant']"},
    "time_from": {"type": "number", "description": "Inclusive minimum timestamp (Unix seconds)"},
    "time_to": {"type": "number", "description": "Inclusive maximum timestamp (Unix seconds)"}
  },
  "required": ["session_id"]
}"#;

/// `lcm_describe` — DAG/metadata overview, no content load (§10.3).
pub const LCM_DESCRIBE: &str = r#"{
  "type": "object",
  "properties": {
    "node_id": {"type": "integer", "description": "Describe one node's subtree; omit for a session overview"},
    "externalized_ref": {"type": "string", "description": "Describe an externalized payload's metadata (no content)"}
  }
}"#;

/// `lcm_expand` — recover exact content for a node, message, or externalized payload (§10.4).
pub const LCM_EXPAND: &str = r#"{
  "type": "object",
  "properties": {
    "node_id": {"type": "integer", "description": "Expand a summary node (current session)"},
    "store_id": {"type": "integer", "description": "Recover a raw message (cross-session)"},
    "externalized_ref": {"type": "string", "description": "Recover an externalized payload's bytes by its ref"},
    "max_tokens": {"type": "integer", "default": 4000, "minimum": 1},
    "source_offset": {"type": "integer", "default": 0, "description": "Node-mode source pagination"},
    "source_limit": {"type": "integer", "description": "Max sources to return this page"},
    "content_offset": {"type": "integer", "default": 0, "description": "Char offset within a single source"}
  }
}"#;

/// `lcm_expand_query` — NL Q&A over expanded context via the aux provider (§10.5).
pub const LCM_EXPAND_QUERY: &str = r#"{
  "type": "object",
  "properties": {
    "prompt": {"type": "string", "description": "The question to answer over recovered context"},
    "query": {"type": "string", "description": "Search query to find candidate summaries before expansion"},
    "node_ids": {"type": "array", "items": {"type": "integer"}, "description": "Explicit summary node IDs to expand instead of searching"},
    "max_results": {"type": "integer", "default": 5, "description": "Max candidate summaries to expand when using query"},
    "max_tokens": {"type": "integer", "default": 2000, "description": "Answer token budget returned to the main agent"},
    "context_max_tokens": {"type": "integer", "default": 32000, "description": "Expanded-context budget for the auxiliary model (default max(max_tokens, expansion_context_tokens))"}
  },
  "required": ["prompt"]
}"#;

/// `lcm_status` — compaction/store/config diagnostics, no params (§10.6).
pub const LCM_STATUS: &str = r#"{"type": "object", "properties": {}}"#;

/// `lcm_doctor` — health checks, no params (§10.6).
pub const LCM_DOCTOR: &str = r#"{"type": "object", "properties": {}}"#;
