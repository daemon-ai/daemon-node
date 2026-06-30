// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The BEAM SQLite schema — port of `beam.py` `init_beam` (L485-L1026) and the knowledge-layer
//! `init_*` functions.
//!
//! Emitted as one static DDL batch (current columns folded in; no historical migration needed for a
//! fresh store). sqlite-vec `vec0` virtual tables are created separately under the `vec-ext` feature
//! because they require the registered extension. FTS5 comes from the bundled SQLite build.

/// The core + knowledge schema, run by the `PRAGMA user_version` ladder ([`super::MIGRATIONS`]).
/// Connection pragmas (WAL etc.) live in [`super::Store::init`], not here: the ladder runs in a
/// transaction and `journal_mode` cannot change inside one.
pub const SCHEMA: &str = r#"
-- ── BEAM tiers ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS working_memory (
    id                       TEXT PRIMARY KEY,
    content                  TEXT NOT NULL,
    source                   TEXT,
    timestamp                TEXT,
    session_id               TEXT DEFAULT 'default',
    importance               REAL DEFAULT 0.5,
    metadata_json            TEXT,
    veracity                 TEXT DEFAULT 'unknown',
    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    memory_type              TEXT DEFAULT 'unknown',
    consolidated_at          TEXT,
    consolidation_claimed_at TEXT,
    recall_count             INTEGER DEFAULT 0,
    last_recalled            TIMESTAMP,
    pinned                   INTEGER DEFAULT 0,
    valid_until              TIMESTAMP,
    superseded_by            TEXT,
    scope                    TEXT DEFAULT 'global',
    author_id                TEXT,
    author_type              TEXT,
    channel_id               TEXT,
    trust_tier               TEXT DEFAULT 'STATED',
    validator                TEXT,
    validated_at             TIMESTAMP,
    validation_count         INTEGER DEFAULT 0,
    event_date               TEXT,
    event_date_precision     TEXT DEFAULT 'unknown',
    temporal_tags            TEXT DEFAULT '[]',
    corrected_by             INTEGER
);
CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_wm_timestamp ON working_memory(timestamp);
CREATE INDEX IF NOT EXISTS idx_wm_unconsolidated
    ON working_memory(session_id, timestamp) WHERE consolidated_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_wm_context_session
    ON working_memory(session_id, importance DESC, timestamp DESC) WHERE superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS episodic_memory (
    rowid                INTEGER PRIMARY KEY AUTOINCREMENT,
    id                   TEXT UNIQUE NOT NULL,
    content              TEXT NOT NULL,
    source               TEXT,
    timestamp            TEXT,
    session_id           TEXT DEFAULT 'default',
    importance           REAL DEFAULT 0.5,
    metadata_json        TEXT,
    summary_of           TEXT DEFAULT '',
    veracity             TEXT DEFAULT 'unknown',
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tier                 INTEGER DEFAULT 1,
    degraded_at          TEXT,
    memory_type          TEXT DEFAULT 'unknown',
    binary_vector        BLOB,
    recall_count         INTEGER DEFAULT 0,
    last_recalled        TIMESTAMP,
    valid_until          TIMESTAMP,
    superseded_by        TEXT,
    scope                TEXT DEFAULT 'global',
    author_id            TEXT,
    author_type          TEXT,
    channel_id           TEXT,
    trust_tier           TEXT DEFAULT 'STATED',
    validator            TEXT,
    validated_at         TIMESTAMP,
    validation_count     INTEGER DEFAULT 0,
    event_date           TEXT,
    event_date_precision TEXT DEFAULT 'unknown',
    temporal_tags        TEXT DEFAULT '[]',
    corrected_by         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_em_session ON episodic_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_em_tier ON episodic_memory(tier);

CREATE TABLE IF NOT EXISTS scratchpad (
    id         TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    session_id TEXT DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sp_session ON scratchpad(session_id);

-- ── Vectors (float32 fallback store) + events + validations + log ──────────────
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id      TEXT PRIMARY KEY,
    embedding_json TEXT NOT NULL,
    model          TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_events (
    event_id         TEXT PRIMARY KEY,
    memory_id        TEXT,
    operation        TEXT CHECK(operation IN ('CREATE','UPDATE','DELETE','CONSOLIDATE')),
    timestamp        TEXT,
    device_id        TEXT,
    payload          TEXT,
    parent_event_ids TEXT DEFAULT '[]',
    importance       REAL DEFAULT 0.5,
    expiry           TEXT,
    event_hash       TEXT,
    synced_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_ev_timestamp ON memory_events(timestamp);

CREATE TABLE IF NOT EXISTS memory_validations (
    validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     TEXT,
    validator     TEXT,
    validated_at  TEXT,
    action        TEXT,
    new_content   TEXT,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS consolidation_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT,
    items_consolidated INTEGER,
    summary_preview    TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Structured facts + MEMORIA tables ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS facts (
    fact_id       TEXT PRIMARY KEY,
    session_id    TEXT,
    subject       TEXT,
    predicate     TEXT,
    object        TEXT,
    timestamp     TEXT,
    source_msg_id TEXT,
    confidence    REAL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memoria_facts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT,
    message_idx      INTEGER,
    fact_type        TEXT,
    key              TEXT,
    value            TEXT,
    context_snippet  TEXT,
    importance       REAL,
    timestamp        TEXT,
    source_memory_id TEXT
);
CREATE TABLE IF NOT EXISTS memoria_timelines (
    event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT,
    date             TEXT,
    message_idx      INTEGER,
    description      TEXT,
    source           TEXT,
    source_memory_id TEXT
);
CREATE TABLE IF NOT EXISTS memoria_instructions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT,
    message_idx      INTEGER,
    instruction      TEXT,
    active           INTEGER DEFAULT 1,
    topic            TEXT,
    context_snippet  TEXT,
    source_memory_id TEXT
);
CREATE TABLE IF NOT EXISTS memoria_preferences (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT,
    message_idx      INTEGER,
    preference       TEXT,
    topic            TEXT,
    evolution        TEXT,
    context_snippet  TEXT,
    source_memory_id TEXT
);
CREATE TABLE IF NOT EXISTS memoria_kg (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT,
    subject          TEXT,
    predicate        TEXT,
    object           TEXT,
    message_idx      INTEGER,
    confidence       REAL,
    source_memory_id TEXT
);

-- Provider audit log (bank-co-located), port of `hermes_memory_provider/audit.py` L19-L40.
-- Fire-and-forget mutation trail; failures never break a memory operation.
CREATE TABLE IF NOT EXISTS audit_log (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL NOT NULL,
    action        TEXT NOT NULL,
    memory_id     TEXT,
    bank          TEXT,
    scope         TEXT,
    profile       TEXT,
    session_id    TEXT,
    source_tool   TEXT,
    tokens_used   INTEGER,
    reason        TEXT,
    metadata_json TEXT
);

-- ── FTS5 (bundled SQLite) + sync triggers ──────────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS fts_episodes
    USING fts5(content, content='episodic_memory', content_rowid='rowid');
CREATE VIRTUAL TABLE IF NOT EXISTS fts_working
    USING fts5(id UNINDEXED, content);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts
    USING fts5(subject, predicate, object, content='facts');

CREATE TRIGGER IF NOT EXISTS em_ai AFTER INSERT ON episodic_memory BEGIN
    INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS em_ad AFTER DELETE ON episodic_memory BEGIN
    INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS em_au AFTER UPDATE ON episodic_memory BEGIN
    INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS wm_ai AFTER INSERT ON working_memory BEGIN
    INSERT INTO fts_working(id, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS wm_ad AFTER DELETE ON working_memory BEGIN
    DELETE FROM fts_working WHERE id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS wm_au AFTER UPDATE OF content ON working_memory BEGIN
    DELETE FROM fts_working WHERE id = old.id;
    INSERT INTO fts_working(id, content) VALUES (new.id, new.content);
END;

-- ── Knowledge layer ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS triples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    valid_from  TEXT NOT NULL,
    valid_until TEXT,
    source      TEXT,
    confidence  REAL DEFAULT 1.0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);

CREATE TABLE IF NOT EXISTS annotations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id  TEXT NOT NULL,
    kind       TEXT NOT NULL,
    value      TEXT NOT NULL,
    source     TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_annot_unique ON annotations(memory_id, kind, value);

CREATE TABLE IF NOT EXISTS canonical_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id    TEXT NOT NULL,
    category    TEXT NOT NULL,
    name        TEXT NOT NULL,
    body        TEXT NOT NULL,
    source      TEXT,
    confidence  REAL DEFAULT 1.0,
    version     INTEGER NOT NULL DEFAULT 1,
    valid_from  TEXT NOT NULL,
    valid_until TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_current
    ON canonical_facts(owner_id, category, name) WHERE valid_until IS NULL;

CREATE TABLE IF NOT EXISTS gists (
    id                TEXT PRIMARY KEY,
    text              TEXT NOT NULL,
    timestamp         TEXT,
    participants_json TEXT,
    location          TEXT,
    emotion           TEXT,
    time_scope        TEXT,
    memory_id         TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    target     TEXT NOT NULL,
    edge_type  TEXT NOT NULL,
    weight     REAL DEFAULT 1.0,
    timestamp  TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS consolidated_facts (
    id            TEXT PRIMARY KEY,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    confidence    REAL DEFAULT 0.5,
    mention_count INTEGER DEFAULT 1,
    first_seen    TEXT,
    last_seen     TEXT,
    sources_json  TEXT,
    veracity      TEXT DEFAULT 'unknown',
    superseded_by TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conflicts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_a_id     TEXT NOT NULL,
    fact_b_id     TEXT NOT NULL,
    conflict_type TEXT,
    resolution    TEXT,
    resolved_at   TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"#;
