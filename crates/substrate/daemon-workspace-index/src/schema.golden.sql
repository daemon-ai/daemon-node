CREATE INDEX chunks_path ON chunks(path);
CREATE TABLE chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    path       TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE TABLE files (
    path         TEXT PRIMARY KEY,
    content_hash BLOB NOT NULL,
    mtime_ms     INTEGER NOT NULL,
    size         INTEGER NOT NULL,
    indexed_ms   INTEGER NOT NULL
);
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE sqlite_sequence(name,seq);
