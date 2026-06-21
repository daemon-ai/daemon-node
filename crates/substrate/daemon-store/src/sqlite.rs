//! SQLite-backed [`SessionStore`] (host-spec §4), behind the `sqlite` feature.
//!
//! A durable, single-file backend that mirrors [`InMemoryStore`](crate::InMemoryStore)'s semantics
//! exactly — the same atomic `checkpoint_and_enqueue`, `(session, epoch, job)` idempotency, monotonic
//! fencing, and append-only trace journal — so it is a drop-in alternative the impl-agnostic
//! acceptance harness passes against unchanged. Snapshots and trace entries are stored as opaque
//! BLOBs, keeping this crate free of the engine/crypto stacks (layout §3 DAG).
//!
//! Concurrency: a single `Mutex<Connection>` serializes all access; the database runs in WAL mode.
//! The multi-statement operations ([`SqliteStore::checkpoint_and_enqueue`],
//! [`SqliteStore::record_completion_and_wake`]) commit their durable mutations in one SQLite
//! transaction / before any post-commit fault fires, so a crash boundary leaves consistent state.

use crate::{
    Activation, Checkpoint, CommittedRoot, FaultPoint, JobCommand, JobCompletion, JournalEntry,
    JournalPage, SessionStatus, SessionStore, StoreError, StoreStats, TraceEntry, TraceSegment,
};
use async_trait::async_trait;
use daemon_common::{
    ContentHash, DaemonError, Epoch, FenceToken, JobId, JournalStreamId, MerkleRoot, PartitionId,
    SessionId, SnapshotBlob, UsageDelta,
};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::Mutex;

/// The durable schema = the host-spec §4 tables (session record, completion inbox, job/wake
/// outboxes, the enqueued-job dedupe set) plus the verifiable trace journal (entries + sealed
/// roots). The activation lease is the monotonic `fence` column on `session_record`.
const SCHEMA: &str = r#"
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS session_record (
    session_id  TEXT PRIMARY KEY,
    partition   INTEGER NOT NULL,
    epoch       INTEGER NOT NULL,
    status_kind TEXT NOT NULL,
    status_job  TEXT,
    snapshot    BLOB NOT NULL,
    fence       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS completion_inbox (
    rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    job_id     TEXT NOT NULL,
    payload    BLOB NOT NULL,
    UNIQUE(session_id, epoch, job_id)
);

CREATE TABLE IF NOT EXISTS job_outbox (
    rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL,
    session_id TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    payload    BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS enqueued_jobs (
    job_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS wake_outbox (
    rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS delegations (
    rowseq         INTEGER PRIMARY KEY AUTOINCREMENT,
    child          TEXT NOT NULL UNIQUE,
    parent_session TEXT NOT NULL,
    parent_epoch   INTEGER NOT NULL,
    job_id         TEXT NOT NULL,
    payload        BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS session_usage (
    session_id    TEXT PRIMARY KEY,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    api_calls     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS journal_entries (
    cursor       INTEGER PRIMARY KEY AUTOINCREMENT,
    stream       TEXT NOT NULL,
    segment      INTEGER NOT NULL,
    seq          INTEGER NOT NULL,
    bytes        BLOB NOT NULL,
    content_hash BLOB NOT NULL,
    UNIQUE (stream, segment, seq)
);

CREATE TABLE IF NOT EXISTS journal_roots (
    stream    TEXT NOT NULL,
    segment   INTEGER NOT NULL,
    root      BLOB NOT NULL,
    signature BLOB NOT NULL,
    PRIMARY KEY (stream, segment)
);
"#;

/// A durable, SQLite-backed [`SessionStore`].
pub struct SqliteStore {
    conn: Mutex<Connection>,
    /// A one-shot armed crash boundary (acceptance test #2), mirroring `InMemoryStore`.
    fault: Mutex<Option<FaultPoint>>,
}

fn sql_err(e: rusqlite::Error) -> StoreError {
    StoreError::Common(DaemonError::Other(format!("sqlite: {e}")))
}

fn status_from_row(kind: &str, job: Option<String>) -> SessionStatus {
    match kind {
        "active" => SessionStatus::Active,
        "suspended" => SessionStatus::Suspended {
            job_id: JobId::new(job.unwrap_or_default()),
        },
        "ready" => SessionStatus::Ready,
        _ => SessionStatus::Completed,
    }
}

impl SqliteStore {
    /// Open (creating if absent) a SQLite-backed store at `path`, applying the durable schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let conn = Connection::open(path).map_err(sql_err)?;
        conn.execute_batch(SCHEMA).map_err(sql_err)?;
        Ok(Self {
            conn: Mutex::new(conn),
            fault: Mutex::new(None),
        })
    }

    /// Open an ephemeral in-memory SQLite store (tests; the single connection keeps it alive).
    pub fn open_in_memory() -> Result<Self, StoreError> {
        let conn = Connection::open_in_memory().map_err(sql_err)?;
        conn.execute_batch(SCHEMA).map_err(sql_err)?;
        Ok(Self {
            conn: Mutex::new(conn),
            fault: Mutex::new(None),
        })
    }

    /// Arm the store to fail at a given durable boundary (acceptance test #2). `None` disarms.
    pub fn set_fault(&self, fault: Option<FaultPoint>) {
        *self.fault.lock().unwrap() = fault;
    }

    /// Fire (and clear) an armed fault at `point`, if any.
    fn take_fault(&self, point: FaultPoint) -> Result<(), StoreError> {
        let mut f = self.fault.lock().unwrap();
        if *f == Some(point) {
            *f = None;
            return Err(StoreError::Fault(point));
        }
        Ok(())
    }

    /// Read the committed (sealed) root of a `(stream, segment)`, if any.
    fn committed_root(
        conn: &Connection,
        stream: &JournalStreamId,
        segment: u64,
    ) -> Option<CommittedRoot> {
        conn.query_row(
            "SELECT root, signature FROM journal_roots WHERE stream = ?1 AND segment = ?2",
            params![stream.as_str(), segment as i64],
            |row| {
                let root_bytes: Vec<u8> = row.get(0)?;
                let mut root = [0u8; 32];
                root.copy_from_slice(&root_bytes);
                Ok(CommittedRoot {
                    root: MerkleRoot::new(root),
                    signature: row.get::<_, Vec<u8>>(1)?,
                })
            },
        )
        .optional()
        .ok()
        .flatten()
    }

    /// Read a session's current fence, or `NotFound`.
    fn fence_of(conn: &Connection, id: &SessionId) -> Result<FenceToken, StoreError> {
        conn.query_row(
            "SELECT fence FROM session_record WHERE session_id = ?1",
            params![id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(sql_err)?
        .map(|f| FenceToken(f as u64))
        .ok_or_else(|| StoreError::NotFound(id.clone()))
    }
}

#[async_trait]
impl SessionStore for SqliteStore {
    async fn create_session(
        &self,
        id: SessionId,
        partition: PartitionId,
        snapshot: SnapshotBlob,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO session_record \
             (session_id, partition, epoch, status_kind, status_job, snapshot, fence) \
             VALUES (?1, ?2, 0, 'ready', NULL, ?3, 0)",
            params![id.as_str(), partition.0 as i64, snapshot.as_bytes()],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn acquire_activation_lease(&self, id: &SessionId) -> Result<FenceToken, StoreError> {
        let conn = self.conn.lock().unwrap();
        let current = Self::fence_of(&conn, id)?;
        let next = current.next();
        conn.execute(
            "UPDATE session_record SET fence = ?2, status_kind = 'active' WHERE session_id = ?1",
            params![id.as_str(), next.0 as i64],
        )
        .map_err(sql_err)?;
        Ok(next)
    }

    async fn load_for_activation(
        &self,
        id: &SessionId,
        fence: FenceToken,
    ) -> Result<Activation, StoreError> {
        let conn = self.conn.lock().unwrap();
        let snapshot = conn
            .query_row(
                "SELECT snapshot FROM session_record WHERE session_id = ?1",
                params![id.as_str()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(sql_err)?
            .map(SnapshotBlob::new)
            .ok_or_else(|| StoreError::NotFound(id.clone()))?;

        let mut stmt = conn
            .prepare(
                "SELECT epoch, job_id, payload FROM completion_inbox \
                 WHERE session_id = ?1 ORDER BY rowseq",
            )
            .map_err(sql_err)?;
        let unapplied = stmt
            .query_map(params![id.as_str()], |row| {
                Ok(JobCompletion {
                    session_id: id.clone(),
                    epoch: Epoch(row.get::<_, i64>(0)? as u64),
                    job_id: JobId::new(row.get::<_, String>(1)?),
                    payload: row.get::<_, Vec<u8>>(2)?,
                })
            })
            .map_err(sql_err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(sql_err)?;

        Ok(Activation {
            snapshot,
            unapplied,
            fence,
        })
    }

    async fn checkpoint_and_enqueue(
        &self,
        checkpoint: Checkpoint,
        job: JobCommand,
        fence: FenceToken,
    ) -> Result<(), StoreError> {
        // Fence first: a stale incarnation must not commit (lifecycle §5).
        {
            let conn = self.conn.lock().unwrap();
            let current = Self::fence_of(&conn, &checkpoint.session_id)?;
            if fence < current {
                return Err(StoreError::Fenced {
                    have: fence.0,
                    current: current.0,
                });
            }
        }
        // Boundary: abort the whole transaction before anything is written.
        self.take_fault(FaultPoint::BeforeSnapshot)?;

        // Atomic commit: snapshot, epoch, status, and job-outbox enqueue land together.
        {
            let mut conn = self.conn.lock().unwrap();
            let tx = conn.transaction().map_err(sql_err)?;
            tx.execute(
                "UPDATE session_record \
                 SET snapshot = ?2, epoch = ?3, status_kind = 'suspended', status_job = ?4 \
                 WHERE session_id = ?1",
                params![
                    checkpoint.session_id.as_str(),
                    checkpoint.snapshot.as_bytes(),
                    checkpoint.epoch.0 as i64,
                    job.job_id.as_str(),
                ],
            )
            .map_err(sql_err)?;
            // Dedupe re-enqueues from idempotent re-activation (mirrors `enqueued_jobs`).
            let fresh = tx
                .execute(
                    "INSERT OR IGNORE INTO enqueued_jobs (job_id) VALUES (?1)",
                    params![job.job_id.as_str()],
                )
                .map_err(sql_err)?;
            if fresh > 0 {
                tx.execute(
                    "INSERT INTO job_outbox (job_id, session_id, epoch, payload) \
                     VALUES (?1, ?2, ?3, ?4)",
                    params![
                        job.job_id.as_str(),
                        job.session_id.as_str(),
                        job.epoch.0 as i64,
                        job.payload,
                    ],
                )
                .map_err(sql_err)?;
            }
            tx.commit().map_err(sql_err)?;
        }

        // Post-commit crash boundaries: durable state is already complete and consistent.
        self.take_fault(FaultPoint::AfterSnapshot)?;
        self.take_fault(FaultPoint::AfterJobOutbox)?;
        Ok(())
    }

    async fn mark_completed(
        &self,
        checkpoint: Checkpoint,
        fence: FenceToken,
    ) -> Result<(), StoreError> {
        let mut guard = self.conn.lock().unwrap();
        let current = Self::fence_of(&guard, &checkpoint.session_id)?;
        if fence < current {
            return Err(StoreError::Fenced {
                have: fence.0,
                current: current.0,
            });
        }
        let tx = guard.transaction().map_err(sql_err)?;
        tx.execute(
            "UPDATE session_record \
             SET snapshot = ?2, epoch = ?3, status_kind = 'completed', status_job = NULL \
             WHERE session_id = ?1",
            params![
                checkpoint.session_id.as_str(),
                checkpoint.snapshot.as_bytes(),
                checkpoint.epoch.0 as i64,
            ],
        )
        .map_err(sql_err)?;
        // If this session was delegated by a parent, fulfill that parent's job and wake it in the
        // same transaction — the binding is durable, so a crash cannot orphan the parent at any depth.
        let parent: Option<(String, i64, String)> = tx
            .query_row(
                "SELECT parent_session, parent_epoch, job_id FROM delegations WHERE child = ?1",
                params![checkpoint.session_id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .map_err(sql_err)?;
        if let Some((parent_session, parent_epoch, job_id)) = parent {
            let payload = format!("child:{}", checkpoint.session_id).into_bytes();
            let fresh = tx
                .execute(
                    "INSERT OR IGNORE INTO completion_inbox (session_id, epoch, job_id, payload) \
                     VALUES (?1, ?2, ?3, ?4)",
                    params![parent_session, parent_epoch, job_id, payload],
                )
                .map_err(sql_err)?;
            if fresh > 0 {
                tx.execute(
                    "UPDATE session_record SET status_kind = 'ready' WHERE session_id = ?1",
                    params![parent_session],
                )
                .map_err(sql_err)?;
                tx.execute(
                    "INSERT INTO wake_outbox (session_id) VALUES (?1)",
                    params![parent_session],
                )
                .map_err(sql_err)?;
            }
        }
        tx.commit().map_err(sql_err)?;
        Ok(())
    }

    async fn bind_delegation(&self, child: SessionId, job: JobCommand) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO delegations \
             (child, parent_session, parent_epoch, job_id, payload) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                child.as_str(),
                job.session_id.as_str(),
                job.epoch.0 as i64,
                job.job_id.as_str(),
                job.payload,
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn children_of(&self, parent: &SessionId) -> Vec<SessionId> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn
            .prepare("SELECT child FROM delegations WHERE parent_session = ?1 ORDER BY rowseq")
        {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        let rows = stmt
            .query_map(params![parent.as_str()], |row| {
                Ok(SessionId::new(row.get::<_, String>(0)?))
            })
            .and_then(|r| r.collect::<Result<Vec<_>, _>>());
        rows.unwrap_or_default()
    }

    async fn enqueue_wake(&self, id: SessionId) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO wake_outbox (session_id) VALUES (?1)",
            params![id.as_str()],
        );
    }

    async fn delegation_work(&self, child: &SessionId) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT payload FROM delegations WHERE child = ?1",
            params![child.as_str()],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .optional()
        .ok()
        .flatten()
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn record_usage(&self, id: &SessionId, delta: UsageDelta) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO session_usage (session_id, input_tokens, output_tokens, api_calls) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(session_id) DO UPDATE SET \
               input_tokens = input_tokens + excluded.input_tokens, \
               output_tokens = output_tokens + excluded.output_tokens, \
               api_calls = api_calls + excluded.api_calls",
            params![
                id.as_str(),
                delta.input_tokens as i64,
                delta.output_tokens as i64,
                delta.api_calls as i64,
            ],
        );
    }

    async fn usage_of(&self, id: &SessionId) -> UsageDelta {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT input_tokens, output_tokens, api_calls FROM session_usage WHERE session_id = ?1",
            params![id.as_str()],
            |row| {
                Ok(UsageDelta {
                    input_tokens: row.get::<_, i64>(0)? as u64,
                    output_tokens: row.get::<_, i64>(1)? as u64,
                    api_calls: row.get::<_, i64>(2)? as u32,
                })
            },
        )
        .optional()
        .ok()
        .flatten()
        .unwrap_or_default()
    }

    async fn record_completion_and_wake(&self, c: &JobCompletion) -> Result<(), StoreError> {
        // Commit the completion + Ready status durably; only then consider the wake.
        {
            let conn = self.conn.lock().unwrap();
            let exists: bool = conn
                .query_row(
                    "SELECT 1 FROM session_record WHERE session_id = ?1",
                    params![c.session_id.as_str()],
                    |_| Ok(()),
                )
                .optional()
                .map_err(sql_err)?
                .is_some();
            if !exists {
                return Err(StoreError::NotFound(c.session_id.clone()));
            }
            // Idempotent per (session, epoch, job): a redelivered completion is a no-op.
            let fresh = conn
                .execute(
                    "INSERT OR IGNORE INTO completion_inbox (session_id, epoch, job_id, payload) \
                     VALUES (?1, ?2, ?3, ?4)",
                    params![
                        c.session_id.as_str(),
                        c.epoch.0 as i64,
                        c.job_id.as_str(),
                        c.payload,
                    ],
                )
                .map_err(sql_err)?;
            if fresh == 0 {
                return Ok(());
            }
            conn.execute(
                "UPDATE session_record SET status_kind = 'ready' WHERE session_id = ?1",
                params![c.session_id.as_str()],
            )
            .map_err(sql_err)?;
        }
        // Boundary: completion durable + session Ready; crash before publishing the wake. The
        // recovery scan must still re-activate the Ready session (invariant #7).
        self.take_fault(FaultPoint::BeforeWakePublish)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO wake_outbox (session_id) VALUES (?1)",
            params![c.session_id.as_str()],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn scan_resumable(&self, partition: PartitionId) -> Result<Vec<SessionId>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT session_id FROM session_record \
                 WHERE partition = ?1 AND status_kind IN ('ready', 'active')",
            )
            .map_err(sql_err)?;
        let ids = stmt
            .query_map(params![partition.0 as i64], |row| {
                Ok(SessionId::new(row.get::<_, String>(0)?))
            })
            .map_err(sql_err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(sql_err)?;
        Ok(ids)
    }

    async fn dequeue_job(&self) -> Option<JobCommand> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT rowseq, job_id, session_id, epoch, payload FROM job_outbox \
                 ORDER BY rowseq LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        JobCommand {
                            job_id: JobId::new(row.get::<_, String>(1)?),
                            session_id: SessionId::new(row.get::<_, String>(2)?),
                            epoch: Epoch(row.get::<_, i64>(3)? as u64),
                            payload: row.get::<_, Vec<u8>>(4)?,
                        },
                    ))
                },
            )
            .optional()
            .ok()??;
        let (rowseq, job) = row;
        conn.execute("DELETE FROM job_outbox WHERE rowseq = ?1", params![rowseq])
            .ok()?;
        Some(job)
    }

    async fn dequeue_wake(&self) -> Option<SessionId> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT rowseq, session_id FROM wake_outbox ORDER BY rowseq LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        SessionId::new(row.get::<_, String>(1)?),
                    ))
                },
            )
            .optional()
            .ok()??;
        let (rowseq, id) = row;
        conn.execute("DELETE FROM wake_outbox WHERE rowseq = ?1", params![rowseq])
            .ok()?;
        Some(id)
    }

    async fn status(&self, id: &SessionId) -> Option<SessionStatus> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT status_kind, status_job FROM session_record WHERE session_id = ?1",
            params![id.as_str()],
            |row| {
                Ok(status_from_row(
                    &row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                ))
            },
        )
        .optional()
        .ok()
        .flatten()
    }

    async fn list_sessions(&self) -> Vec<(SessionId, SessionStatus)> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            match conn.prepare("SELECT session_id, status_kind, status_job FROM session_record") {
                Ok(stmt) => stmt,
                Err(_) => return Vec::new(),
            };
        let rows = stmt.query_map([], |row| {
            Ok((
                SessionId::new(row.get::<_, String>(0)?),
                status_from_row(&row.get::<_, String>(1)?, row.get::<_, Option<String>>(2)?),
            ))
        });
        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn stats(&self) -> StoreStats {
        let conn = self.conn.lock().unwrap();
        let count = |sql: &str| -> usize {
            conn.query_row(sql, [], |row| row.get::<_, i64>(0))
                .unwrap_or(0) as usize
        };
        StoreStats {
            pending_jobs: count("SELECT COUNT(*) FROM job_outbox"),
            pending_wakes: count("SELECT COUNT(*) FROM wake_outbox"),
            sessions: count("SELECT COUNT(*) FROM session_record"),
        }
    }

    async fn append_trace(
        &self,
        stream: &JournalStreamId,
        segment: u64,
        entry: TraceEntry,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        // Append-only + idempotent per `(stream, segment, seq)`; keyed by stream, not by session, so
        // a non-durable unit journals without a session record. The autoincrement `cursor` is the
        // stream-monotonic pagination key.
        conn.execute(
            "INSERT OR IGNORE INTO journal_entries (stream, segment, seq, bytes, content_hash) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                stream.as_str(),
                segment as i64,
                entry.seq as i64,
                entry.bytes,
                entry.content_hash.as_bytes().as_slice(),
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn commit_trace_segment(
        &self,
        stream: &JournalStreamId,
        segment: u64,
        root: MerkleRoot,
        signature: Vec<u8>,
        fence: Option<FenceToken>,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        // Durable path: fenced exactly like a checkpoint. Non-durable streams pass `None`.
        if let Some(fence) = fence {
            let id = SessionId::new(stream.as_str());
            let current = Self::fence_of(&conn, &id)?;
            if fence < current {
                return Err(StoreError::Fenced {
                    have: fence.0,
                    current: current.0,
                });
            }
        }
        conn.execute(
            "INSERT OR REPLACE INTO journal_roots (stream, segment, root, signature) \
             VALUES (?1, ?2, ?3, ?4)",
            params![
                stream.as_str(),
                segment as i64,
                root.as_bytes().as_slice(),
                signature,
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn load_trace_segment(
        &self,
        stream: &JournalStreamId,
        segment: u64,
    ) -> Option<TraceSegment> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT seq, bytes, content_hash FROM journal_entries \
                 WHERE stream = ?1 AND segment = ?2 ORDER BY seq",
            )
            .ok()?;
        let entries: Vec<TraceEntry> = stmt
            .query_map(params![stream.as_str(), segment as i64], |row| {
                let hash_bytes: Vec<u8> = row.get(2)?;
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&hash_bytes);
                Ok(TraceEntry {
                    seq: row.get::<_, i64>(0)? as u64,
                    bytes: row.get::<_, Vec<u8>>(1)?,
                    content_hash: ContentHash::new(hash),
                })
            })
            .ok()?
            .filter_map(Result::ok)
            .collect();

        let committed = Self::committed_root(&conn, stream, segment);

        if entries.is_empty() && committed.is_none() {
            return None;
        }
        Some(TraceSegment {
            stream: stream.clone(),
            segment,
            entries,
            committed,
        })
    }

    async fn load_journal(
        &self,
        stream: &JournalStreamId,
        after_cursor: u64,
        max: u32,
    ) -> JournalPage {
        let conn = self.conn.lock().unwrap();
        let head_cursor: u64 = conn
            .query_row(
                "SELECT COALESCE(MAX(cursor), 0) FROM journal_entries WHERE stream = ?1",
                params![stream.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .map(|c| c as u64)
            .unwrap_or(0);
        let limit: i64 = if max == 0 { -1 } else { max as i64 };
        let mut stmt = match conn.prepare(
            "SELECT cursor, segment, seq, bytes, content_hash FROM journal_entries \
             WHERE stream = ?1 AND cursor > ?2 ORDER BY cursor LIMIT ?3",
        ) {
            Ok(stmt) => stmt,
            Err(_) => return JournalPage::default(),
        };
        let rows = stmt.query_map(
            params![stream.as_str(), after_cursor as i64, limit],
            |row| {
                let hash_bytes: Vec<u8> = row.get(4)?;
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&hash_bytes);
                Ok(JournalEntry {
                    cursor: row.get::<_, i64>(0)? as u64,
                    segment: row.get::<_, i64>(1)? as u64,
                    entry: TraceEntry {
                        seq: row.get::<_, i64>(2)? as u64,
                        bytes: row.get::<_, Vec<u8>>(3)?,
                        content_hash: ContentHash::new(hash),
                    },
                })
            },
        );
        let entries: Vec<JournalEntry> = match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => return JournalPage::default(),
        };
        let next_cursor = entries.last().map(|e| e.cursor).unwrap_or(after_cursor);
        let mut segments: Vec<u64> = entries.iter().map(|e| e.segment).collect();
        segments.sort_unstable();
        segments.dedup();
        let segment_roots = segments
            .into_iter()
            .filter_map(|seg| Self::committed_root(&conn, stream, seg).map(|root| (seg, root)))
            .collect();
        JournalPage {
            entries,
            segment_roots,
            next_cursor,
            head_cursor,
        }
    }
}
