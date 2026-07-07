CREATE INDEX completion_notices_parent ON completion_notices (parent_session);
CREATE INDEX cron_jobs_due ON cron_jobs (paused, next_fire_unix);
CREATE INDEX cron_runs_job ON cron_runs (job_id, rowseq);
CREATE INDEX journal_seals_stream ON journal_seals (stream, id);
CREATE INDEX pending_session_input_session ON pending_session_input (session_id, rowseq);
CREATE INDEX room_members_room ON room_members (room_id);
CREATE TABLE acp_catalog (
    name  TEXT PRIMARY KEY,
    entry BLOB NOT NULL
);
CREATE TABLE background_edges (
    rowseq         INTEGER PRIMARY KEY AUTOINCREMENT,
    child          TEXT NOT NULL UNIQUE,
    parent_session TEXT NOT NULL,
    work_label     TEXT NOT NULL,
    origin         TEXT NOT NULL DEFAULT 'background'
);
CREATE TABLE chat_routes (
    key        TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    profile    TEXT,
    descriptor BLOB NOT NULL
);
CREATE TABLE completion_inbox (
    rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    job_id     TEXT NOT NULL,
    payload    BLOB NOT NULL,
    UNIQUE(session_id, epoch, job_id)
);
CREATE TABLE completion_notice_outbox (
rowseq         INTEGER PRIMARY KEY AUTOINCREMENT,
parent_session TEXT NOT NULL,
child          TEXT NOT NULL,
payload        BLOB NOT NULL
, call_id TEXT);
CREATE TABLE completion_notices (
child          TEXT PRIMARY KEY,
parent_session TEXT NOT NULL,
notified       INTEGER NOT NULL DEFAULT 0
, call_id TEXT);
CREATE TABLE cron_jobs (
    id             TEXT PRIMARY KEY,
    schedule       TEXT NOT NULL,
    spec           BLOB NOT NULL,
    next_fire_unix INTEGER,
    paused         INTEGER NOT NULL DEFAULT 0,
    last_run_unix  INTEGER,
    last_ok        INTEGER,
    last_detail    TEXT,
    fire_count     INTEGER NOT NULL DEFAULT 0,
    created_unix   INTEGER NOT NULL
, owner TEXT);
CREATE TABLE cron_runs (
    rowseq        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT NOT NULL,
    started_unix  INTEGER NOT NULL,
    finished_unix INTEGER,
    ok            INTEGER NOT NULL,
    detail        TEXT,
    session       TEXT,
    manual        INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE cron_suggestions (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL DEFAULT '',
    spec         BLOB NOT NULL,
    dedup_key    TEXT NOT NULL UNIQUE,
    status       TEXT NOT NULL DEFAULT 'pending',
    created_unix INTEGER NOT NULL
);
CREATE TABLE delegations (
    rowseq         INTEGER PRIMARY KEY AUTOINCREMENT,
    child          TEXT NOT NULL UNIQUE,
    parent_session TEXT NOT NULL,
    parent_epoch   INTEGER NOT NULL,
    job_id         TEXT NOT NULL,
    payload        BLOB NOT NULL
);
CREATE TABLE detached_seq (
parent_session TEXT PRIMARY KEY,
n              INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE enqueued_jobs (
    job_id TEXT PRIMARY KEY
);
CREATE TABLE job_outbox (
    rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL,
    session_id TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    payload    BLOB NOT NULL,
    lifetime   TEXT
, child TEXT);
CREATE TABLE journal_entries (
    cursor       INTEGER PRIMARY KEY AUTOINCREMENT,
    stream       TEXT NOT NULL,
    segment      INTEGER NOT NULL,
    seq          INTEGER NOT NULL,
    bytes        BLOB NOT NULL,
    content_hash BLOB NOT NULL,
    UNIQUE (stream, segment, seq)
);
CREATE TABLE journal_roots (
    stream    TEXT NOT NULL,
    segment   INTEGER NOT NULL,
    root      BLOB NOT NULL,
    signature BLOB NOT NULL,
    PRIMARY KEY (stream, segment)
);
CREATE TABLE journal_seals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stream         TEXT NOT NULL,
    seal_cursor    INTEGER NOT NULL,
    retained_turns INTEGER NOT NULL,
    epoch          INTEGER NOT NULL,
    recorded_unix  INTEGER NOT NULL
);
CREATE TABLE pending_approvals (
    rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    job_id     TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    prompt     TEXT NOT NULL,
    path       TEXT,
    decision   INTEGER, fingerprint TEXT,
    UNIQUE(session_id, job_id)
);
CREATE TABLE pending_session_input (
rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,
session_id TEXT NOT NULL,
input      BLOB NOT NULL
);
CREATE TABLE room_members (
    room_id    TEXT NOT NULL,
    member     TEXT NOT NULL,
    profile    TEXT,
    session_id TEXT NOT NULL,
    PRIMARY KEY (room_id, member)
);
CREATE TABLE rooms (
    id         TEXT PRIMARY KEY,
    name       TEXT,
    policy     TEXT NOT NULL,
    descriptor BLOB NOT NULL
);
CREATE VIRTUAL TABLE session_fts USING fts5 (
    session_id UNINDEXED,
    title,
    body,
    tokenize = 'unicode61'
);
CREATE TABLE 'session_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID;
CREATE TABLE 'session_fts_content'(id INTEGER PRIMARY KEY, c0, c1, c2);
CREATE TABLE 'session_fts_data'(id INTEGER PRIMARY KEY, block BLOB);
CREATE TABLE 'session_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
CREATE TABLE 'session_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
CREATE TABLE session_meta (
    session_id       TEXT PRIMARY KEY,
    bound_profile    TEXT,
    overlay          BLOB,
    title            TEXT,
    last_activity_ms INTEGER,
    role             TEXT,
    parent           TEXT,
    pinned           INTEGER NOT NULL DEFAULT 0,
    archived         INTEGER NOT NULL DEFAULT 0,
    scheduled_job    TEXT,
    activation_epoch INTEGER NOT NULL DEFAULT 0
, owner TEXT, terminal_ms INTEGER);
CREATE TABLE session_record (
    session_id  TEXT PRIMARY KEY,
    partition   INTEGER NOT NULL,
    epoch       INTEGER NOT NULL,
    status_kind TEXT NOT NULL,
    status_job  TEXT,
    snapshot    BLOB NOT NULL,
    fence       INTEGER NOT NULL
);
CREATE TABLE session_usage (
    session_id          TEXT PRIMARY KEY,
    input_tokens        INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    api_calls           INTEGER NOT NULL,
    cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens  INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens    INTEGER NOT NULL DEFAULT 0,
    cost_micros         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE tool_overrides (
tool    TEXT PRIMARY KEY,
enabled INTEGER NOT NULL
);
CREATE TABLE wake_outbox (
    rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL
);
