CREATE INDEX idx_annot_kind_value ON annotations(kind, value);
CREATE INDEX idx_annot_memory_kind ON annotations(memory_id, kind);
CREATE UNIQUE INDEX idx_annot_unique ON annotations(memory_id, kind, value);
CREATE UNIQUE INDEX idx_canonical_current
    ON canonical_facts(owner_id, category, name) WHERE valid_until IS NULL;
CREATE INDEX idx_canonical_owner_category ON canonical_facts(owner_id, category);
CREATE INDEX idx_canonical_slot ON canonical_facts(owner_id, category, name);
CREATE INDEX idx_cf_object ON consolidated_facts(object);
CREATE INDEX idx_cf_predicate ON consolidated_facts(predicate);
CREATE INDEX idx_cf_subject ON consolidated_facts(subject);
CREATE INDEX idx_edges_source ON graph_edges(source);
CREATE INDEX idx_edges_target ON graph_edges(target);
CREATE INDEX idx_edges_type ON graph_edges(edge_type);
CREATE INDEX idx_em_author ON episodic_memory(author_id);
CREATE INDEX idx_em_channel ON episodic_memory(channel_id);
CREATE INDEX idx_em_event_date ON episodic_memory(event_date);
CREATE INDEX idx_em_scope_imp
    ON episodic_memory(scope, importance) WHERE superseded_by IS NULL;
CREATE INDEX idx_em_session ON episodic_memory(session_id);
CREATE INDEX idx_em_source ON episodic_memory(source);
CREATE INDEX idx_em_tier ON episodic_memory(tier);
CREATE INDEX idx_em_timestamp ON episodic_memory(timestamp);
CREATE INDEX idx_facts_key ON memoria_facts(key);
CREATE INDEX idx_facts_object ON facts(object);
CREATE INDEX idx_facts_predicate ON facts(predicate);
CREATE INDEX idx_facts_session ON memoria_facts(session_id);
CREATE INDEX idx_facts_source ON facts(source_msg_id);
CREATE INDEX idx_facts_subject ON facts(subject);
CREATE INDEX idx_facts_type ON memoria_facts(fact_type);
CREATE INDEX idx_instr_active ON memoria_instructions(active);
CREATE INDEX idx_instr_session ON memoria_instructions(session_id);
CREATE INDEX idx_kg_predicate ON memoria_kg(predicate);
CREATE INDEX idx_kg_session ON memoria_kg(session_id);
CREATE INDEX idx_kg_subject ON memoria_kg(subject);
CREATE INDEX idx_me_device_id ON memory_events(device_id);
CREATE INDEX idx_me_memory_id ON memory_events(memory_id);
CREATE INDEX idx_me_timestamp ON memory_events(timestamp);
CREATE INDEX idx_mem_emb_type ON memory_embeddings(memory_id, model);
CREATE INDEX idx_pref_session ON memoria_preferences(session_id);
CREATE INDEX idx_sp_session ON scratchpad(session_id);
CREATE INDEX idx_timelines_date ON memoria_timelines(date);
CREATE INDEX idx_timelines_session ON memoria_timelines(session_id);
CREATE INDEX idx_triples_object ON triples(object);
CREATE INDEX idx_triples_predicate ON triples(predicate);
CREATE INDEX idx_triples_subject ON triples(subject);
CREATE INDEX idx_triples_valid_from ON triples(valid_from);
CREATE INDEX idx_validations_memory ON memory_validations(memory_id);
CREATE INDEX idx_validations_validator ON memory_validations(validator);
CREATE INDEX idx_wm_author ON working_memory(author_id);
CREATE INDEX idx_wm_channel ON working_memory(channel_id);
CREATE INDEX idx_wm_consolidation_claims
    ON working_memory(consolidation_claimed_at) WHERE consolidation_claimed_at IS NOT NULL;
CREATE INDEX idx_wm_context_global
    ON working_memory(scope, importance DESC, timestamp DESC) WHERE superseded_by IS NULL;
CREATE INDEX idx_wm_context_session
    ON working_memory(session_id, importance DESC, timestamp DESC) WHERE superseded_by IS NULL;
CREATE INDEX idx_wm_event_date ON working_memory(event_date);
CREATE INDEX idx_wm_session ON working_memory(session_id);
CREATE INDEX idx_wm_session_recall
    ON working_memory(session_id, last_recalled) WHERE valid_until IS NULL;
CREATE INDEX idx_wm_source ON working_memory(source);
CREATE INDEX idx_wm_timestamp ON working_memory(timestamp);
CREATE INDEX idx_wm_unconsolidated
    ON working_memory(session_id, timestamp) WHERE consolidated_at IS NULL;
CREATE INDEX idx_wm_validated_at ON working_memory(validated_at);
CREATE INDEX idx_wm_validator ON working_memory(validator);
CREATE TABLE annotations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id  TEXT NOT NULL,
    kind       TEXT NOT NULL,
    value      TEXT NOT NULL,
    source     TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE audit_log (
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
CREATE TABLE canonical_facts (
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
CREATE TABLE conflicts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_a_id     TEXT NOT NULL,
    fact_b_id     TEXT NOT NULL,
    conflict_type TEXT,
    resolution    TEXT,
    resolved_at   TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE consolidated_facts (
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
CREATE TABLE consolidation_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT,
    items_consolidated INTEGER,
    summary_preview    TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE episodic_memory (
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
CREATE TABLE facts (
    fact_id       TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    timestamp     TEXT,
    source_msg_id TEXT,
    confidence    REAL DEFAULT 1.0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE VIRTUAL TABLE fts_episodes
    USING fts5(content, content='episodic_memory', content_rowid='rowid');
CREATE TABLE 'fts_episodes_config'(k PRIMARY KEY, v) WITHOUT ROWID;
CREATE TABLE 'fts_episodes_data'(id INTEGER PRIMARY KEY, block BLOB);
CREATE TABLE 'fts_episodes_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
CREATE TABLE 'fts_episodes_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
CREATE VIRTUAL TABLE fts_facts
    USING fts5(subject, predicate, object, content='facts');
CREATE TABLE 'fts_facts_config'(k PRIMARY KEY, v) WITHOUT ROWID;
CREATE TABLE 'fts_facts_data'(id INTEGER PRIMARY KEY, block BLOB);
CREATE TABLE 'fts_facts_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
CREATE TABLE 'fts_facts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
CREATE VIRTUAL TABLE fts_working
    USING fts5(id UNINDEXED, content);
CREATE TABLE 'fts_working_config'(k PRIMARY KEY, v) WITHOUT ROWID;
CREATE TABLE 'fts_working_content'(id INTEGER PRIMARY KEY, c0, c1);
CREATE TABLE 'fts_working_data'(id INTEGER PRIMARY KEY, block BLOB);
CREATE TABLE 'fts_working_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
CREATE TABLE 'fts_working_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
CREATE TABLE gists (
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
CREATE TABLE graph_edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    target     TEXT NOT NULL,
    edge_type  TEXT NOT NULL,
    weight     REAL DEFAULT 1.0,
    timestamp  TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE memoria_facts (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT DEFAULT 'default',
    message_idx        INTEGER,
    fact_type          TEXT,
    key                TEXT,
    value              TEXT,
    context_snippet    TEXT,
    importance         REAL DEFAULT 0.5,
    timestamp          TEXT,
    version_id         INTEGER DEFAULT 0,
    previous_value     TEXT,
    updated_msg_idx    INTEGER,
    valid_from_msg_idx INTEGER,
    valid_to_msg_idx   INTEGER,
    source_memory_id   TEXT
);
CREATE TABLE memoria_instructions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT DEFAULT 'default',
    message_idx      INTEGER,
    instruction      TEXT,
    active           INTEGER DEFAULT 1,
    topic            TEXT,
    context_snippet  TEXT,
    source_memory_id TEXT
);
CREATE TABLE memoria_kg (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT DEFAULT 'default',
    subject          TEXT,
    predicate        TEXT,
    object           TEXT,
    message_idx      INTEGER,
    confidence       REAL DEFAULT 0.7,
    source_memory_id TEXT
);
CREATE TABLE memoria_preferences (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT DEFAULT 'default',
    message_idx      INTEGER,
    preference       TEXT,
    topic            TEXT,
    evolution        TEXT,
    context_snippet  TEXT,
    source_memory_id TEXT
);
CREATE TABLE memoria_timelines (
    event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT DEFAULT 'default',
    date             TEXT,
    message_idx      INTEGER,
    description      TEXT,
    source           TEXT,
    source_memory_id TEXT
);
CREATE TABLE memory_embeddings (
    memory_id      TEXT PRIMARY KEY,
    embedding_json TEXT NOT NULL,
    model          TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE memory_events (
    event_id         TEXT PRIMARY KEY,
    memory_id        TEXT NOT NULL,
    operation        TEXT NOT NULL CHECK(operation IN ('CREATE','UPDATE','DELETE','CONSOLIDATE')),
    timestamp        TEXT NOT NULL,
    device_id        TEXT NOT NULL,
    payload          TEXT,
    parent_event_ids TEXT DEFAULT '[]',
    importance       REAL DEFAULT 0.5,
    expiry           TEXT,
    event_hash       TEXT,
    synced_at        TEXT
);
CREATE TABLE memory_validations (
    validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     TEXT NOT NULL,
    validator     TEXT NOT NULL,
    validated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action        TEXT NOT NULL,
    new_content   TEXT,
    note          TEXT
);
CREATE TABLE scratchpad (
    id         TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    session_id TEXT DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE sync_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE triples (
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
CREATE TABLE working_memory (
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
CREATE TRIGGER em_ad AFTER DELETE ON episodic_memory BEGIN
    INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER em_ai AFTER INSERT ON episodic_memory BEGIN
    INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER em_au AFTER UPDATE ON episodic_memory BEGIN
    INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO fts_facts(fts_facts, rowid, subject, predicate, object)
    VALUES ('delete', old.rowid, old.subject, old.predicate, old.object);
END;
CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO fts_facts(rowid, subject, predicate, object)
    VALUES (new.rowid, new.subject, new.predicate, new.object);
END;
CREATE TRIGGER trim_validations_to_3
AFTER INSERT ON memory_validations
BEGIN
    DELETE FROM memory_validations
    WHERE memory_id = NEW.memory_id
      AND validation_id NOT IN (
          SELECT validation_id FROM memory_validations
          WHERE memory_id = NEW.memory_id
          ORDER BY validation_id DESC
          LIMIT 3
      );
END;
CREATE TRIGGER wm_ad AFTER DELETE ON working_memory BEGIN
    DELETE FROM fts_working WHERE id = old.id;
END;
CREATE TRIGGER wm_ai AFTER INSERT ON working_memory BEGIN
    INSERT INTO fts_working(id, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER wm_au AFTER UPDATE OF content ON working_memory BEGIN
    DELETE FROM fts_working WHERE id = old.id;
    INSERT INTO fts_working(id, content) VALUES (new.id, new.content);
END;
