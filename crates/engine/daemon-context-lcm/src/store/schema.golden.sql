CREATE INDEX idx_lcm_lifecycle_current_session
    ON lcm_lifecycle_state(current_session_id);
CREATE INDEX idx_msg_session        ON messages(session_id, store_id);
CREATE INDEX idx_msg_session_ts     ON messages(session_id, timestamp);
CREATE INDEX idx_msg_source_session ON messages(source, session_id, store_id);
CREATE INDEX idx_nodes_session_depth  ON summary_nodes(session_id, depth, created_at);
CREATE INDEX idx_nodes_session_latest ON summary_nodes(session_id, latest_at, created_at);
CREATE TABLE lcm_lifecycle_state (
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
CREATE TABLE lcm_migration_state (
    step_name    TEXT PRIMARY KEY,
    completed_at REAL
);
CREATE TABLE messages (
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
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='store_id'
);
CREATE TABLE 'messages_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID;
CREATE TABLE 'messages_fts_data'(id INTEGER PRIMARY KEY, block BLOB);
CREATE TABLE 'messages_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
CREATE TABLE 'messages_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE VIRTUAL TABLE nodes_fts USING fts5(
    summary,
    content='summary_nodes',
    content_rowid='node_id'
);
CREATE TABLE 'nodes_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID;
CREATE TABLE 'nodes_fts_data'(id INTEGER PRIMARY KEY, block BLOB);
CREATE TABLE 'nodes_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
CREATE TABLE 'nodes_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE summary_nodes (
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
CREATE TRIGGER msg_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.store_id, old.content);
END;
CREATE TRIGGER msg_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.store_id, new.content);
END;
CREATE TRIGGER msg_fts_update AFTER UPDATE OF content ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.store_id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.store_id, new.content);
END;
CREATE TRIGGER nodes_fts_delete AFTER DELETE ON summary_nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, summary) VALUES ('delete', old.node_id, old.summary);
END;
CREATE TRIGGER nodes_fts_insert AFTER INSERT ON summary_nodes BEGIN
    INSERT INTO nodes_fts(rowid, summary) VALUES (new.node_id, new.summary);
END;
