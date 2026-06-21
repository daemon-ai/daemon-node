//! The LCM summary-store schema (skeleton).
//!
//! A faithful subset of the hermes-lcm SQLite schema: the summary DAG node table plus an FTS5 index
//! over summary text for drill-down search. The deep port adds the message-ingest tables, lifecycle
//! columns, and externalization blobs per `daemon-context-lcm-port-spec.md` §4.

/// The DDL applied at store open (idempotent — `IF NOT EXISTS` throughout).
pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS summary_nodes (
    node_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL,
    depth              INTEGER NOT NULL DEFAULT 0,
    summary            TEXT NOT NULL,
    token_count        INTEGER NOT NULL DEFAULT 0,
    source_token_count INTEGER NOT NULL DEFAULT 0,
    created_at         REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_summary_nodes_session ON summary_nodes(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS summary_fts USING fts5(
    summary,
    content='summary_nodes',
    content_rowid='node_id'
);

CREATE TRIGGER IF NOT EXISTS summary_nodes_ai AFTER INSERT ON summary_nodes BEGIN
    INSERT INTO summary_fts(rowid, summary) VALUES (new.node_id, new.summary);
END;
"#;
