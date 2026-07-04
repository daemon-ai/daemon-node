// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The LCM store schema (v4 subset — `daemon-context-lcm-port-spec.md` §4).
//!
//! A faithful subset of the hermes-lcm SQLite schema covering milestones M1-M4: the lossless
//! `messages` transcript (FTS5 external-content + sync triggers), the summary-DAG `summary_nodes`
//! table with `source_ids` lineage (no edges table — §5.2) + its FTS index, and the
//! `lcm_lifecycle_state` / `metadata` / `lcm_migration_state` bookkeeping tables. Protection blobs
//! (M5), routing/preset state (M7), and the legacy-import migration ladder are out of scope here.

/// The schema version a fresh DB is created at (`SCHEMA_VERSION = 4`, §4.1).
pub const SCHEMA_VERSION: i64 = 4;

/// The DDL applied at store open (idempotent — `IF NOT EXISTS` throughout). A greenfield DB is
/// created directly at v4 (no historical ladder; legacy `lcm.db` import is out of scope — §4.6).
pub const SCHEMA: &str = r#"
-- §4.2 lossless message transcript (the `store_id` lineage D0 nodes reference)
CREATE TABLE IF NOT EXISTS messages (
    store_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    source          TEXT    DEFAULT '',
    role            TEXT    NOT NULL,
    content         TEXT,
    tool_call_id    TEXT,
    tool_calls      TEXT,
    tool_name       TEXT,
    timestamp       REAL    NOT NULL,
    token_estimate  INTEGER DEFAULT 0,
    pinned          INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_msg_session        ON messages(session_id, store_id);
CREATE INDEX IF NOT EXISTS idx_msg_session_ts     ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_msg_source_session ON messages(source, session_id, store_id);

-- §4.3 the summary DAG; lineage lives in `source_ids` (JSON), interpreted by `source_type`
CREATE TABLE IF NOT EXISTS summary_nodes (
    node_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT    NOT NULL,
    depth              INTEGER NOT NULL DEFAULT 0,
    summary            TEXT    NOT NULL,
    token_count        INTEGER NOT NULL DEFAULT 0,
    source_token_count INTEGER NOT NULL DEFAULT 0,
    source_ids         TEXT    NOT NULL DEFAULT '[]',
    source_type        TEXT    NOT NULL DEFAULT 'messages',
    created_at         REAL    NOT NULL,
    earliest_at        REAL,
    latest_at          REAL,
    expand_hint        TEXT    DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_nodes_session_depth  ON summary_nodes(session_id, depth, created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_session_latest ON summary_nodes(session_id, latest_at, created_at);

-- §4.5 per-conversation compaction frontier + debt (v2 adds the maintenance/rollover/reset
-- timestamps + the finalized-session index — see MIGRATION_V2)
CREATE TABLE IF NOT EXISTS lcm_lifecycle_state (
    conversation_id                  TEXT PRIMARY KEY,
    current_session_id               TEXT,
    last_finalized_session_id        TEXT,
    current_frontier_store_id        INTEGER NOT NULL DEFAULT 0,
    last_finalized_frontier_store_id INTEGER NOT NULL DEFAULT 0,
    debt_kind                        TEXT,
    debt_size_estimate               INTEGER NOT NULL DEFAULT 0,
    current_bound_at                 REAL,
    last_finalized_at                REAL,
    debt_updated_at                  REAL,
    updated_at                       REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_current_session
    ON lcm_lifecycle_state(current_session_id);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS lcm_migration_state (
    step_name    TEXT PRIMARY KEY,
    completed_at REAL
);

-- §4.4 FTS5 external-content indexes + sync triggers
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='store_id'
);
CREATE TRIGGER IF NOT EXISTS msg_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.store_id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS msg_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.store_id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS msg_fts_update AFTER UPDATE OF content ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.store_id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.store_id, new.content);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    summary,
    content='summary_nodes',
    content_rowid='node_id'
);
CREATE TRIGGER IF NOT EXISTS nodes_fts_insert AFTER INSERT ON summary_nodes BEGIN
    INSERT INTO nodes_fts(rowid, summary) VALUES (new.node_id, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS nodes_fts_delete AFTER DELETE ON summary_nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, summary) VALUES ('delete', old.node_id, old.summary);
END;
"#;

/// Migration v2 — the session-lifecycle timestamps + finalized-session index
/// (`LCM:db_bootstrap.py:155-175` `ensure_lifecycle_state_columns` /
/// `idx_lcm_lifecycle_last_finalized_session`): stamps for maintenance attempts, `/new` rollovers,
/// and resets, plus the reverse lookup by finalized session.
pub const MIGRATION_V2: &str = r#"
ALTER TABLE lcm_lifecycle_state ADD COLUMN last_maintenance_attempt_at REAL;
ALTER TABLE lcm_lifecycle_state ADD COLUMN last_rollover_at REAL;
ALTER TABLE lcm_lifecycle_state ADD COLUMN last_reset_at REAL;
CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_last_finalized_session
    ON lcm_lifecycle_state(last_finalized_session_id);
"#;

/// One external-content FTS5 index + its sync triggers, as a repairable unit
/// (`ExternalContentFtsSpec`, `LCM:db_bootstrap.py`). `create_sql`/`trigger_sqls` must stay
/// byte-identical to the corresponding [`SCHEMA`] DDL (asserted by a test) so a startup rebuild
/// leaves `sqlite_master` matching the golden schema.
pub struct FtsSpec {
    /// The FTS5 virtual-table name.
    pub table: &'static str,
    /// The single indexed column.
    pub indexed_column: &'static str,
    /// The external-content base table.
    pub content_table: &'static str,
    /// The base table's rowid column (`content_rowid=`).
    pub content_rowid: &'static str,
    /// The `CREATE VIRTUAL TABLE` DDL (identical to the [`SCHEMA`] copy).
    pub create_sql: &'static str,
    /// The sync-trigger DDL (identical to the [`SCHEMA`] copies).
    pub trigger_sqls: &'static [&'static str],
}

/// The `messages` transcript FTS index.
pub const MESSAGES_FTS: FtsSpec = FtsSpec {
    table: "messages_fts",
    indexed_column: "content",
    content_table: "messages",
    content_rowid: "store_id",
    create_sql: r#"CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='store_id'
);"#,
    trigger_sqls: &[
        r#"CREATE TRIGGER IF NOT EXISTS msg_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.store_id, new.content);
END;"#,
        r#"CREATE TRIGGER IF NOT EXISTS msg_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.store_id, old.content);
END;"#,
        r#"CREATE TRIGGER IF NOT EXISTS msg_fts_update AFTER UPDATE OF content ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.store_id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.store_id, new.content);
END;"#,
    ],
};

/// The `summary_nodes` DAG FTS index.
pub const NODES_FTS: FtsSpec = FtsSpec {
    table: "nodes_fts",
    indexed_column: "summary",
    content_table: "summary_nodes",
    content_rowid: "node_id",
    create_sql: r#"CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    summary,
    content='summary_nodes',
    content_rowid='node_id'
);"#,
    trigger_sqls: &[
        r#"CREATE TRIGGER IF NOT EXISTS nodes_fts_insert AFTER INSERT ON summary_nodes BEGIN
    INSERT INTO nodes_fts(rowid, summary) VALUES (new.node_id, new.summary);
END;"#,
        r#"CREATE TRIGGER IF NOT EXISTS nodes_fts_delete AFTER DELETE ON summary_nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, summary) VALUES ('delete', old.node_id, old.summary);
END;"#,
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    /// A rebuild must recreate exactly the DDL the base schema declares, or a repaired database
    /// would drift from the golden `sqlite_master` dump.
    #[test]
    fn fts_specs_match_schema_ddl() {
        for spec in [&MESSAGES_FTS, &NODES_FTS] {
            assert!(
                SCHEMA.contains(spec.create_sql),
                "{}: create_sql drifted from SCHEMA",
                spec.table
            );
            for trigger in spec.trigger_sqls {
                assert!(
                    SCHEMA.contains(trigger),
                    "{}: trigger DDL drifted from SCHEMA",
                    spec.table
                );
            }
        }
    }
}
