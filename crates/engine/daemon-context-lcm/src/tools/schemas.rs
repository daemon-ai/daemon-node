//! JSON-schema strings for the seven `lcm_*` drill-down tools (§10.7).
//!
//! Ported from `LCM:schemas.py`. Each is the `ToolDef::schema` advertised to the model. Kept as
//! string constants (not built at runtime) so they are cheap to enumerate and easy to diff.

/// `lcm_grep` — search the transcript + summaries (§10.1).
pub const LCM_GREP: &str = r#"{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "FTS5 query; quotes for phrases"},
    "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 200},
    "sort": {"type": "string", "enum": ["recency", "relevance", "hybrid"], "default": "recency"},
    "session_scope": {"type": "string", "enum": ["current", "all", "session"], "default": "current"},
    "session_id": {"type": "string", "description": "Required iff session_scope=session"},
    "role": {"type": "string", "enum": ["user", "assistant", "tool", "system"]},
    "source": {"type": "string"},
    "time_from": {"type": "number", "description": "Unix seconds (inclusive lower bound)"},
    "time_to": {"type": "number", "description": "Unix seconds (inclusive upper bound)"}
  },
  "required": ["query"]
}"#;

/// `lcm_load_session` — ordered raw-message page for a session (§10.2).
pub const LCM_LOAD_SESSION: &str = r#"{
  "type": "object",
  "properties": {
    "session_id": {"type": "string", "description": "Defaults to the current session"},
    "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 200},
    "max_content_chars": {"type": "integer", "default": 4000, "minimum": 1, "maximum": 20000},
    "after_store_id": {"type": "integer", "description": "Exclusive cursor; next_cursor from a prior page"}
  }
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
    "query": {"type": "string", "description": "Candidate selection query"},
    "node_ids": {"type": "array", "items": {"type": "integer"}},
    "max_results": {"type": "integer", "default": 5},
    "max_tokens": {"type": "integer", "default": 2000, "description": "Answer token budget"},
    "context_max_tokens": {"type": "integer", "default": 32000}
  },
  "required": ["prompt"]
}"#;

/// `lcm_status` — compaction/store/config diagnostics, no params (§10.6).
pub const LCM_STATUS: &str = r#"{"type": "object", "properties": {}}"#;

/// `lcm_doctor` — health checks, no params (§10.6).
pub const LCM_DOCTOR: &str = r#"{"type": "object", "properties": {}}"#;
