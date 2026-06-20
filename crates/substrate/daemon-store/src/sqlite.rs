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
    Activation, Checkpoint, CommittedRoot, FaultPoint, JobCommand, JobCompletion, SessionStatus,
    SessionStore, StoreError, StoreStats, TraceEntry, TraceSegment,
};
use async_trait::async_trait;
use daemon_common::{
    ContentHash, DaemonError, Epoch, FenceToken, JobId, MerkleRoot, PartitionId, SessionId,
    SnapshotBlob,
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

CREATE TABLE IF NOT EXISTS trace_entries (
    session_id   TEXT NOT NULL,
    epoch        INTEGER NOT NULL,
    seq          INTEGER NOT NULL,
    bytes        BLOB NOT NULL,
    content_hash BLOB NOT NULL,
    PRIMARY KEY (session_id, epoch, seq)
);

CREATE TABLE IF NOT EXISTS trace_roots (
    session_id TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    root       BLOB NOT NULL,
    signature  BLOB NOT NULL,
    PRIMARY KEY (session_id, epoch)
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
        let conn = self.conn.lock().unwrap();
        let current = Self::fence_of(&conn, &checkpoint.session_id)?;
        if fence < current {
            return Err(StoreError::Fenced {
                have: fence.0,
                current: current.0,
            });
        }
        conn.execute(
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
        Ok(())
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
                |row| Ok((row.get::<_, i64>(0)?, SessionId::new(row.get::<_, String>(1)?))),
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
        let mut stmt = match conn
            .prepare("SELECT session_id, status_kind, status_job FROM session_record")
        {
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
        session: &SessionId,
        epoch: Epoch,
        entry: TraceEntry,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM session_record WHERE session_id = ?1",
                params![session.as_str()],
                |_| Ok(()),
            )
            .optional()
            .map_err(sql_err)?
            .is_some();
        if !exists {
            return Err(StoreError::NotFound(session.clone()));
        }
        // Append-only + idempotent: a redelivered `seq` is a no-op.
        conn.execute(
            "INSERT OR IGNORE INTO trace_entries (session_id, epoch, seq, bytes, content_hash) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                session.as_str(),
                epoch.0 as i64,
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
        session: &SessionId,
        epoch: Epoch,
        root: MerkleRoot,
        signature: Vec<u8>,
        fence: FenceToken,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let current = Self::fence_of(&conn, session)?;
        // Fenced exactly like a checkpoint: a stale incarnation cannot seal a segment root.
        if fence < current {
            return Err(StoreError::Fenced {
                have: fence.0,
                current: current.0,
            });
        }
        conn.execute(
            "INSERT OR REPLACE INTO trace_roots (session_id, epoch, root, signature) \
             VALUES (?1, ?2, ?3, ?4)",
            params![
                session.as_str(),
                epoch.0 as i64,
                root.as_bytes().as_slice(),
                signature,
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn load_trace_segment(&self, session: &SessionId, epoch: Epoch) -> Option<TraceSegment> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT seq, bytes, content_hash FROM trace_entries \
                 WHERE session_id = ?1 AND epoch = ?2 ORDER BY seq",
            )
            .ok()?;
        let entries: Vec<TraceEntry> = stmt
            .query_map(params![session.as_str(), epoch.0 as i64], |row| {
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

        let committed = conn
            .query_row(
                "SELECT root, signature FROM trace_roots WHERE session_id = ?1 AND epoch = ?2",
                params![session.as_str(), epoch.0 as i64],
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
            .flatten();

        if entries.is_empty() && committed.is_none() {
            return None;
        }
        Some(TraceSegment {
            session_id: session.clone(),
            epoch,
            entries,
            committed,
        })
    }
}
