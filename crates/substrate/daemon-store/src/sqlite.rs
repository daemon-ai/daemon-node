// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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
    AcpEntry, Activation, ChatRoute, Checkpoint, ChildLifetime, CommittedRoot, FaultPoint,
    JobCommand, JobCompletion, JournalEntry, JournalPage, JournalSeal, ParkedApproval, Room,
    RoomMember, SessionMeta, SessionRole, SessionSearchHit, SessionStatus, SessionStore,
    StoreError, StoreStats, StoredCronJob, StoredCronRun, StoredCronSuggestion, TraceEntry,
    TraceSegment, CRON_RUN_RETENTION,
};
use async_trait::async_trait;
use daemon_common::{
    ContentHash, DaemonError, Epoch, FenceToken, JobId, JournalStreamId, MerkleRoot, PartitionId,
    ProfileRef, SessionId, SnapshotBlob, UsageDelta,
};
use rusqlite::{params, Connection, OptionalExtension};
use rusqlite_migration::{Migrations, M};
use std::path::Path;
use std::sync::{LazyLock, Mutex};

/// The durable schema = the host-spec §4 tables (session record, completion inbox, job/wake
/// outboxes, the enqueued-job dedupe set) plus the verifiable trace journal (entries + sealed
/// roots). The activation lease is the monotonic `fence` column on `session_record`.
const SCHEMA: &str = r#"
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
    payload    BLOB NOT NULL,
    lifetime   TEXT
);

CREATE TABLE IF NOT EXISTS chat_routes (
    key        TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    profile    TEXT,
    descriptor BLOB NOT NULL
);

-- daemon-rooms-spec.md: first-class Rooms (the internal loopback transport). `descriptor` is the
-- opaque host CBOR of the wire Room metadata (protocol-free, mirroring `chat_routes`); membership is
-- the companion `room_members` table, keyed `(room_id, member)`.
CREATE TABLE IF NOT EXISTS rooms (
    id         TEXT PRIMARY KEY,
    name       TEXT,
    policy     TEXT NOT NULL,
    descriptor BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS room_members (
    room_id    TEXT NOT NULL,
    member     TEXT NOT NULL,
    profile    TEXT,
    session_id TEXT NOT NULL,
    PRIMARY KEY (room_id, member)
);
CREATE INDEX IF NOT EXISTS room_members_room ON room_members (room_id);

CREATE TABLE IF NOT EXISTS acp_catalog (
    name  TEXT PRIMARY KEY,
    entry BLOB NOT NULL
);

-- I15 cron backing. `spec` is the opaque host CBOR of the wire CronSpec (protocol-free). The
-- scheduler's hot path is the `cron_due` query, indexed on `next_fire_unix` filtered by `paused`.
CREATE TABLE IF NOT EXISTS cron_jobs (
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
);
CREATE INDEX IF NOT EXISTS cron_jobs_due ON cron_jobs (paused, next_fire_unix);

CREATE TABLE IF NOT EXISTS cron_runs (
    rowseq        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT NOT NULL,
    started_unix  INTEGER NOT NULL,
    finished_unix INTEGER,
    ok            INTEGER NOT NULL,
    detail        TEXT,
    session       TEXT,
    manual        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS cron_runs_job ON cron_runs (job_id, rowseq);

CREATE TABLE IF NOT EXISTS cron_suggestions (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL DEFAULT '',
    spec         BLOB NOT NULL,
    dedup_key    TEXT NOT NULL UNIQUE,
    status       TEXT NOT NULL DEFAULT 'pending',
    created_unix INTEGER NOT NULL
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

-- §12 durable edit-approval HITL: a gated tool action a headless/dormant session suspended on,
-- awaiting an operator decision. Unlike `delegations` its completion comes from an operator answer
-- (`answer_approval`), not a child's terminal state. A NULL `decision` row keeps the session dormant.
CREATE TABLE IF NOT EXISTS pending_approvals (
    rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    job_id     TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    prompt     TEXT NOT NULL,
    path       TEXT,
    decision   INTEGER,
    UNIQUE(session_id, job_id)
);

-- §4.3 attached, non-joining edges (background spawn): parent->child tree edge for audit, but no
-- bound parent job — so `mark_completed` (which only reads `delegations`) never wakes the parent and
-- the child self-closes. `origin` marks the edge family for the tree projection.
CREATE TABLE IF NOT EXISTS background_edges (
    rowseq         INTEGER PRIMARY KEY AUTOINCREMENT,
    child          TEXT NOT NULL UNIQUE,
    parent_session TEXT NOT NULL,
    work_label     TEXT NOT NULL,
    origin         TEXT NOT NULL DEFAULT 'background'
);

-- Host-level per-session metadata: the profile a session resolves its engine from plus an opaque
-- overlay blob (the host's CBOR `SessionOverlay`). The store treats `overlay` as opaque bytes
-- (protocol-free). Read by the resolver at engine construction, so a live override is restored on
-- rehydration. A sidecar table (not columns on `session_record`) so it never touches the hot
-- checkpoint/fence row logic.
CREATE TABLE IF NOT EXISTS session_meta (
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
);

CREATE TABLE IF NOT EXISTS session_usage (
    session_id          TEXT PRIMARY KEY,
    input_tokens        INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    api_calls           INTEGER NOT NULL,
    cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens  INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens    INTEGER NOT NULL DEFAULT 0,
    cost_micros         INTEGER NOT NULL DEFAULT 0
);

-- A full-text index over per-session metadata (title + body): the durable substrate for session
-- search. The store is handed already-extracted text by the host (it never parses snapshots), so
-- this stays protocol-free. `session_id` is UNINDEXED (a stored key, not a search term).
CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5 (
    session_id UNINDEXED,
    title,
    body,
    tokenize = 'unicode61'
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

CREATE TABLE IF NOT EXISTS journal_seals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stream         TEXT NOT NULL,
    seal_cursor    INTEGER NOT NULL,
    retained_turns INTEGER NOT NULL,
    epoch          INTEGER NOT NULL,
    recorded_unix  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS journal_seals_stream ON journal_seals (stream, id);
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

fn migrate_err(e: rusqlite_migration::Error) -> StoreError {
    StoreError::Common(DaemonError::Other(format!("sqlite migrate: {e}")))
}

/// The ordered schema-migration ladder, gated by SQLite's `PRAGMA user_version` (rusqlite_migration).
/// `M1` is the entire current schema (`CREATE TABLE IF NOT EXISTS …`, every column inline), so a
/// fresh database is built in one step and stamped to `user_version = 1`. Append an
/// `M::up("ALTER TABLE …")` for each future schema change — never edit a released migration. Pragmas
/// (WAL etc.) are applied in `open()` *outside* this ladder: `to_latest` runs in a transaction and
/// `journal_mode` cannot change inside one.
static MIGRATIONS: LazyLock<Migrations<'static>> = LazyLock::new(|| {
    Migrations::new(vec![
        M::up(SCHEMA),
        // M2 (Auth 4 ownership): a server-side owner key on both per-session metadata and scheduled
        // jobs. `session_meta.owner` is the per-resource ownership column (stamped on every creation
        // path); `cron_jobs.owner` records the job creator so the worker can stamp the spawned cron
        // session's owner. Both default NULL on legacy rows (visible only to a `SessionSeeAll`
        // holder). Never edit a released migration — this only appends columns.
        M::up(
            "ALTER TABLE session_meta ADD COLUMN owner TEXT;\n\
             ALTER TABLE cron_jobs ADD COLUMN owner TEXT;",
        ),
        // M3 (pending-input seam): opaque inbound inputs queued for a durable session's next
        // activation, drained FIFO at hydrate (the durable `send` path). Its own queue table
        // (mirroring `wake_outbox`), not a `session_meta` column — it is a FIFO, not a property.
        M::up(
            "CREATE TABLE pending_session_input (\n\
                 rowseq     INTEGER PRIMARY KEY AUTOINCREMENT,\n\
                 session_id TEXT NOT NULL,\n\
                 payload    BLOB NOT NULL\n\
             );\n\
             CREATE INDEX pending_session_input_session ON pending_session_input (session_id, rowseq);",
        ),
    ])
});

fn role_to_str(role: SessionRole) -> &'static str {
    match role {
        SessionRole::Primary => "primary",
        SessionRole::ManagedChild => "managed_child",
        SessionRole::EphemeralSubagent => "ephemeral_subagent",
    }
}

fn role_from_str(s: &str) -> Option<SessionRole> {
    match s {
        "primary" => Some(SessionRole::Primary),
        "managed_child" => Some(SessionRole::ManagedChild),
        "ephemeral_subagent" => Some(SessionRole::EphemeralSubagent),
        _ => None,
    }
}

/// Map a `cron_jobs` row to a [`StoredCronJob`] (column order matches every cron_jobs SELECT).
fn cron_job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredCronJob> {
    Ok(StoredCronJob {
        id: row.get::<_, String>(0)?,
        schedule: row.get::<_, String>(1)?,
        spec: row.get::<_, Vec<u8>>(2)?,
        next_fire_unix: row.get::<_, Option<i64>>(3)?.map(|v| v as u64),
        paused: row.get::<_, i64>(4)? != 0,
        last_run_unix: row.get::<_, Option<i64>>(5)?.map(|v| v as u64),
        last_ok: row.get::<_, Option<i64>>(6)?.map(|v| v != 0),
        last_detail: row.get::<_, Option<String>>(7)?,
        fire_count: row.get::<_, i64>(8)? as u32,
        created_unix: row.get::<_, i64>(9)? as u64,
        owner: row.get::<_, Option<String>>(10)?,
    })
}

/// Map a `cron_runs` row to a [`StoredCronRun`] (column order matches every cron_runs SELECT).
fn cron_run_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredCronRun> {
    Ok(StoredCronRun {
        job_id: row.get::<_, String>(0)?,
        started_unix: row.get::<_, i64>(1)? as u64,
        finished_unix: row.get::<_, Option<i64>>(2)?.map(|v| v as u64),
        ok: row.get::<_, i64>(3)? != 0,
        detail: row.get::<_, Option<String>>(4)?,
        session: row.get::<_, Option<String>>(5)?.map(SessionId::new),
        manual: row.get::<_, i64>(6)? != 0,
    })
}

/// Map a `cron_suggestions` row to a [`StoredCronSuggestion`] (column order matches each SELECT).
fn cron_suggestion_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredCronSuggestion> {
    Ok(StoredCronSuggestion {
        id: row.get::<_, String>(0)?,
        title: row.get::<_, String>(1)?,
        description: row.get::<_, String>(2)?,
        source: row.get::<_, String>(3)?,
        spec: row.get::<_, Vec<u8>>(4)?,
        dedup_key: row.get::<_, String>(5)?,
        status: row.get::<_, String>(6)?,
        created_unix: row.get::<_, i64>(7)? as u64,
    })
}

fn lifetime_to_str(lifetime: ChildLifetime) -> &'static str {
    match lifetime {
        ChildLifetime::Persistent => "persistent",
        ChildLifetime::Ephemeral => "ephemeral",
    }
}

fn lifetime_from_str(s: Option<String>) -> ChildLifetime {
    match s.as_deref() {
        Some("ephemeral") => ChildLifetime::Ephemeral,
        // Legacy rows (NULL) and the persistent marker both map to a managed child.
        _ => ChildLifetime::Persistent,
    }
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
        let mut conn = Connection::open(path).map_err(sql_err)?;
        Self::prepare(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            fault: Mutex::new(None),
        })
    }

    /// Open an ephemeral in-memory SQLite store (tests; the single connection keeps it alive).
    pub fn open_in_memory() -> Result<Self, StoreError> {
        let mut conn = Connection::open_in_memory().map_err(sql_err)?;
        Self::prepare(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            fault: Mutex::new(None),
        })
    }

    /// Apply connection pragmas (outside the migration transaction) then run the schema ladder to
    /// the latest `user_version`. The connection is exclusive here (pre-`Mutex`).
    fn prepare(conn: &mut Connection) -> Result<(), StoreError> {
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .map_err(sql_err)?;
        MIGRATIONS.to_latest(conn).map_err(migrate_err)?;
        Ok(())
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
                    "INSERT INTO job_outbox (job_id, session_id, epoch, payload, lifetime) \
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        job.job_id.as_str(),
                        job.session_id.as_str(),
                        job.epoch.0 as i64,
                        job.payload,
                        lifetime_to_str(job.lifetime),
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
            // Prefer the structured completion payload (DelegationResult: summary + artifact refs)
            // when the incarnation captured one; else the legacy `child:{id}` marker.
            let payload = checkpoint
                .completion_payload
                .clone()
                .unwrap_or_else(|| format!("child:{}", checkpoint.session_id).into_bytes());
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

    async fn record_child_edge(
        &self,
        parent: SessionId,
        child: SessionId,
        work_label: String,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        // Deliberately *not* an INSERT into `delegations`: no parent job is bound, so the child's
        // terminal `mark_completed` finds nothing to fulfill and never wakes the parent (self-close).
        conn.execute(
            "INSERT OR IGNORE INTO background_edges (child, parent_session, work_label) \
             VALUES (?1, ?2, ?3)",
            params![child.as_str(), parent.as_str(), work_label],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn children_of(&self, parent: &SessionId) -> Vec<SessionId> {
        let conn = self.conn.lock().unwrap();
        // Both edge families are tree-visible (audit): delegation children (delegation order) then
        // attached background children. Only the former can wake the parent (see `mark_completed`).
        let read = |sql: &str| -> Vec<SessionId> {
            let mut stmt = match conn.prepare(sql) {
                Ok(stmt) => stmt,
                Err(_) => return Vec::new(),
            };
            stmt.query_map(params![parent.as_str()], |row| {
                Ok(SessionId::new(row.get::<_, String>(0)?))
            })
            .and_then(|r| r.collect::<Result<Vec<_>, _>>())
            .unwrap_or_default()
        };
        let mut children =
            read("SELECT child FROM delegations WHERE parent_session = ?1 ORDER BY rowseq");
        children.extend(read(
            "SELECT child FROM background_edges WHERE parent_session = ?1 ORDER BY rowseq",
        ));
        children
    }

    async fn enqueue_wake(&self, id: SessionId) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO wake_outbox (session_id) VALUES (?1)",
            params![id.as_str()],
        );
    }

    async fn enqueue_session_input(&self, id: &SessionId, input: Vec<u8>) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO pending_session_input (session_id, payload) VALUES (?1, ?2)",
            params![id.as_str(), input],
        );
    }

    async fn take_session_inputs(&self, id: &SessionId) -> Vec<Vec<u8>> {
        // Drain-and-delete under the held connection lock: select FIFO, then clear the session's
        // queue. No interleaving is possible (one connection behind the Mutex), so this is atomic
        // with respect to every other store op.
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT payload FROM pending_session_input WHERE session_id = ?1 ORDER BY rowseq",
        ) {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        let inputs: Vec<Vec<u8>> = stmt
            .query_map(params![id.as_str()], |row| row.get::<_, Vec<u8>>(0))
            .and_then(|rows| rows.collect())
            .unwrap_or_default();
        drop(stmt);
        if !inputs.is_empty() {
            let _ = conn.execute(
                "DELETE FROM pending_session_input WHERE session_id = ?1",
                params![id.as_str()],
            );
        }
        inputs
    }

    async fn park_approval(
        &self,
        checkpoint: Checkpoint,
        approvals: Vec<ParkedApproval>,
        fence: FenceToken,
    ) -> Result<(), StoreError> {
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
        self.take_fault(FaultPoint::BeforeSnapshot)?;
        // Atomic commit: snapshot + epoch + Suspended status + parked rows land together. No job is
        // enqueued — the session stays dormant until an operator decision wakes it.
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sql_err)?;
        if let Some(first) = approvals.first() {
            tx.execute(
                "UPDATE session_record \
                 SET snapshot = ?2, epoch = ?3, status_kind = 'suspended', status_job = ?4 \
                 WHERE session_id = ?1",
                params![
                    checkpoint.session_id.as_str(),
                    checkpoint.snapshot.as_bytes(),
                    checkpoint.epoch.0 as i64,
                    first.job_id.as_str(),
                ],
            )
            .map_err(sql_err)?;
        }
        for approval in &approvals {
            // Dedupe a re-parked row on deterministic recovery (UNIQUE(session_id, job_id)).
            tx.execute(
                "INSERT OR IGNORE INTO pending_approvals \
                 (session_id, job_id, epoch, prompt, path, decision) \
                 VALUES (?1, ?2, ?3, ?4, ?5, NULL)",
                params![
                    approval.session_id.as_str(),
                    approval.job_id.as_str(),
                    approval.epoch.0 as i64,
                    approval.prompt,
                    approval.path,
                ],
            )
            .map_err(sql_err)?;
        }
        tx.commit().map_err(sql_err)?;
        self.take_fault(FaultPoint::AfterSnapshot)?;
        Ok(())
    }

    async fn answer_approval(
        &self,
        session: &SessionId,
        job_id: &JobId,
        allow: bool,
    ) -> Result<bool, StoreError> {
        // Stamp the decision, record the completion, and publish the wake in one transaction.
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sql_err)?;
        let row: Option<(i64, Option<i64>)> = tx
            .query_row(
                "SELECT epoch, decision FROM pending_approvals WHERE session_id = ?1 AND job_id = ?2",
                params![session.as_str(), job_id.as_str()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(sql_err)?;
        let epoch = match row {
            // Already answered: idempotent no-op (a redelivered decision).
            Some((_, Some(_))) => return Ok(true),
            Some((epoch, None)) => epoch,
            None => return Ok(false),
        };
        tx.execute(
            "UPDATE pending_approvals SET decision = ?3 WHERE session_id = ?1 AND job_id = ?2",
            params![session.as_str(), job_id.as_str(), allow as i64],
        )
        .map_err(sql_err)?;
        let payload: &[u8] = if allow { b"allow" } else { b"deny" };
        let fresh = tx
            .execute(
                "INSERT OR IGNORE INTO completion_inbox (session_id, epoch, job_id, payload) \
                 VALUES (?1, ?2, ?3, ?4)",
                params![session.as_str(), epoch, job_id.as_str(), payload],
            )
            .map_err(sql_err)?;
        if fresh > 0 {
            tx.execute(
                "UPDATE session_record SET status_kind = 'ready' WHERE session_id = ?1",
                params![session.as_str()],
            )
            .map_err(sql_err)?;
            tx.execute(
                "INSERT INTO wake_outbox (session_id) VALUES (?1)",
                params![session.as_str()],
            )
            .map_err(sql_err)?;
        }
        tx.commit().map_err(sql_err)?;
        Ok(true)
    }

    async fn pending_approvals_of(&self, session: Option<&SessionId>) -> Vec<ParkedApproval> {
        let conn = self.conn.lock().unwrap();
        let map_row = |r: &rusqlite::Row<'_>| -> rusqlite::Result<ParkedApproval> {
            Ok(ParkedApproval {
                session_id: SessionId::new(r.get::<_, String>(0)?),
                job_id: JobId::new(r.get::<_, String>(1)?),
                epoch: Epoch(r.get::<_, i64>(2)? as u64),
                prompt: r.get::<_, String>(3)?,
                path: r.get::<_, Option<String>>(4)?,
                decision: None,
            })
        };
        match session {
            Some(id) => {
                let mut stmt = match conn.prepare(
                    "SELECT session_id, job_id, epoch, prompt, path FROM pending_approvals \
                     WHERE session_id = ?1 AND decision IS NULL ORDER BY rowseq",
                ) {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
                stmt.query_map(params![id.as_str()], map_row)
                    .and_then(|r| r.collect::<Result<Vec<_>, _>>())
                    .unwrap_or_default()
            }
            None => {
                let mut stmt = match conn.prepare(
                    "SELECT session_id, job_id, epoch, prompt, path FROM pending_approvals \
                     WHERE decision IS NULL ORDER BY rowseq",
                ) {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
                stmt.query_map([], map_row)
                    .and_then(|r| r.collect::<Result<Vec<_>, _>>())
                    .unwrap_or_default()
            }
        }
    }

    async fn delegation_work(&self, child: &SessionId) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        let delegated = conn
            .query_row(
                "SELECT payload FROM delegations WHERE child = ?1",
                params![child.as_str()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()
            .ok()
            .flatten()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
        // Fall back to the attached non-joining edge label (§4.3 background spawn).
        delegated.or_else(|| {
            conn.query_row(
                "SELECT work_label FROM background_edges WHERE child = ?1",
                params![child.as_str()],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .ok()
            .flatten()
        })
    }

    async fn record_usage(&self, id: &SessionId, delta: UsageDelta) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO session_usage \
               (session_id, input_tokens, output_tokens, api_calls, \
                cache_read_tokens, cache_write_tokens, reasoning_tokens, cost_micros) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
             ON CONFLICT(session_id) DO UPDATE SET \
               input_tokens = input_tokens + excluded.input_tokens, \
               output_tokens = output_tokens + excluded.output_tokens, \
               api_calls = api_calls + excluded.api_calls, \
               cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens, \
               cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens, \
               reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens, \
               cost_micros = cost_micros + excluded.cost_micros",
            params![
                id.as_str(),
                delta.input_tokens as i64,
                delta.output_tokens as i64,
                delta.api_calls as i64,
                delta.cache_read_tokens as i64,
                delta.cache_write_tokens as i64,
                delta.reasoning_tokens as i64,
                delta.cost_micros as i64,
            ],
        );
    }

    async fn usage_of(&self, id: &SessionId) -> UsageDelta {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT input_tokens, output_tokens, api_calls, \
                    cache_read_tokens, cache_write_tokens, reasoning_tokens, cost_micros \
             FROM session_usage WHERE session_id = ?1",
            params![id.as_str()],
            |row| {
                Ok(UsageDelta {
                    input_tokens: row.get::<_, i64>(0)? as u64,
                    output_tokens: row.get::<_, i64>(1)? as u64,
                    api_calls: row.get::<_, i64>(2)? as u32,
                    cache_read_tokens: row.get::<_, i64>(3)? as u64,
                    cache_write_tokens: row.get::<_, i64>(4)? as u64,
                    reasoning_tokens: row.get::<_, i64>(5)? as u64,
                    cost_micros: row.get::<_, i64>(6)? as u64,
                })
            },
        )
        .optional()
        .ok()
        .flatten()
        .unwrap_or_default()
    }

    async fn index_session_text(&self, id: &SessionId, title: Option<String>, body: &str) {
        let conn = self.conn.lock().unwrap();
        // Replace any prior row for this session (the FTS index carries one row per session).
        let _ = conn.execute(
            "DELETE FROM session_fts WHERE session_id = ?1",
            params![id.as_str()],
        );
        let _ = conn.execute(
            "INSERT INTO session_fts (session_id, title, body) VALUES (?1, ?2, ?3)",
            params![id.as_str(), title.unwrap_or_default(), body],
        );
    }

    async fn search_sessions(&self, query: &str, limit: u32) -> Vec<SessionSearchHit> {
        let conn = self.conn.lock().unwrap();
        let limit = if limit == 0 { 50 } else { limit };
        let mut stmt = match conn.prepare(
            "SELECT session_id, title, snippet(session_fts, 2, '[', ']', '…', 12) \
             FROM session_fts WHERE session_fts MATCH ?1 ORDER BY rank LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map(params![query, limit as i64], |row| {
            Ok(SessionSearchHit {
                session_id: SessionId::new(row.get::<_, String>(0)?),
                title: row.get::<_, String>(1)?,
                snippet: row.get::<_, String>(2)?,
            })
        });
        match rows {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(_) => Vec::new(),
        }
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
                "SELECT rowseq, job_id, session_id, epoch, payload, lifetime FROM job_outbox \
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
                            lifetime: lifetime_from_str(row.get::<_, Option<String>>(5)?),
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

    async fn peek_snapshot(&self, id: &SessionId) -> Option<SnapshotBlob> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT snapshot FROM session_record WHERE session_id = ?1",
            params![id.as_str()],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .optional()
        .ok()
        .flatten()
        .map(SnapshotBlob::new)
    }

    async fn set_session_meta(&self, id: &SessionId, meta: SessionMeta) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let bound = meta.bound_profile.as_ref().map(|p| p.as_str());
        let role = meta.role.map(role_to_str);
        let parent = meta.parent.as_ref().map(|p| p.as_str());
        let last_activity = meta.last_activity_ms.map(|v| v as i64);
        let scheduled_job = meta.scheduled_job.as_ref().map(|j| j.as_str());
        conn.execute(
            "INSERT INTO session_meta (session_id, bound_profile, overlay, title, last_activity_ms, role, parent, pinned, archived, scheduled_job, activation_epoch, owner) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12) \
             ON CONFLICT(session_id) DO UPDATE SET bound_profile = ?2, overlay = ?3, title = ?4, \
             last_activity_ms = ?5, role = ?6, parent = ?7, pinned = ?8, archived = ?9, scheduled_job = ?10, \
             activation_epoch = ?11, owner = ?12",
            params![
                id.as_str(),
                bound,
                meta.overlay,
                meta.title,
                last_activity,
                role,
                parent,
                meta.pinned as i64,
                meta.archived as i64,
                scheduled_job,
                meta.activation_epoch as i64,
                meta.owner,
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn session_meta(&self, id: &SessionId) -> Option<SessionMeta> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT bound_profile, overlay, title, last_activity_ms, role, parent, pinned, archived, scheduled_job, activation_epoch, owner \
             FROM session_meta WHERE session_id = ?1",
            params![id.as_str()],
            |row| {
                let bound: Option<String> = row.get(0)?;
                let overlay: Vec<u8> = row.get(1)?;
                let title: Option<String> = row.get(2)?;
                let last_activity_ms: Option<i64> = row.get(3)?;
                let role: Option<String> = row.get(4)?;
                let parent: Option<String> = row.get(5)?;
                let pinned: i64 = row.get(6)?;
                let archived: i64 = row.get(7)?;
                let scheduled_job: Option<String> = row.get(8)?;
                let activation_epoch: i64 = row.get(9)?;
                let owner: Option<String> = row.get(10)?;
                Ok(SessionMeta {
                    bound_profile: bound.map(ProfileRef::new),
                    overlay,
                    title,
                    last_activity_ms: last_activity_ms.map(|v| v as u64),
                    role: role.as_deref().and_then(role_from_str),
                    parent: parent.map(SessionId::new),
                    pinned: pinned != 0,
                    archived: archived != 0,
                    scheduled_job: scheduled_job.map(JobId::from),
                    activation_epoch: activation_epoch as u64,
                    owner,
                })
            },
        )
        .optional()
        .ok()
        .flatten()
    }

    async fn routing_list(&self) -> Vec<ChatRoute> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn
            .prepare("SELECT key, session_id, profile, descriptor FROM chat_routes ORDER BY key")
        {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map([], |row| {
            Ok(ChatRoute {
                key: row.get::<_, String>(0)?,
                session_id: SessionId::new(row.get::<_, String>(1)?),
                profile: row.get::<_, Option<String>>(2)?.map(ProfileRef::new),
                descriptor: row.get::<_, Vec<u8>>(3)?,
            })
        });
        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn routing_get(&self, key: &str) -> Option<ChatRoute> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT key, session_id, profile, descriptor FROM chat_routes WHERE key = ?1",
            params![key],
            |row| {
                Ok(ChatRoute {
                    key: row.get::<_, String>(0)?,
                    session_id: SessionId::new(row.get::<_, String>(1)?),
                    profile: row.get::<_, Option<String>>(2)?.map(ProfileRef::new),
                    descriptor: row.get::<_, Vec<u8>>(3)?,
                })
            },
        )
        .optional()
        .ok()
        .flatten()
    }

    async fn routing_set(&self, route: ChatRoute) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let profile = route.profile.as_ref().map(|p| p.as_str());
        conn.execute(
            "INSERT INTO chat_routes (key, session_id, profile, descriptor) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(key) DO UPDATE SET session_id = ?2, profile = ?3, descriptor = ?4",
            params![
                route.key,
                route.session_id.as_str(),
                profile,
                route.descriptor
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn routing_remove(&self, key: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM chat_routes WHERE key = ?1", params![key])
            .map_err(sql_err)?;
        Ok(())
    }

    async fn room_list(&self) -> Vec<Room> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            match conn.prepare("SELECT id, name, policy, descriptor FROM rooms ORDER BY id") {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
        let rows = stmt.query_map([], |row| {
            Ok(Room {
                id: row.get::<_, String>(0)?,
                name: row.get::<_, Option<String>>(1)?,
                policy: row.get::<_, String>(2)?,
                descriptor: row.get::<_, Vec<u8>>(3)?,
            })
        });
        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn room_get(&self, id: &str) -> Option<Room> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, name, policy, descriptor FROM rooms WHERE id = ?1",
            params![id],
            |row| {
                Ok(Room {
                    id: row.get::<_, String>(0)?,
                    name: row.get::<_, Option<String>>(1)?,
                    policy: row.get::<_, String>(2)?,
                    descriptor: row.get::<_, Vec<u8>>(3)?,
                })
            },
        )
        .optional()
        .ok()
        .flatten()
    }

    async fn room_set(&self, room: Room) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO rooms (id, name, policy, descriptor) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(id) DO UPDATE SET name = ?2, policy = ?3, descriptor = ?4",
            params![room.id, room.name, room.policy, room.descriptor],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn room_remove(&self, id: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM room_members WHERE room_id = ?1", params![id])
            .map_err(sql_err)?;
        conn.execute("DELETE FROM rooms WHERE id = ?1", params![id])
            .map_err(sql_err)?;
        Ok(())
    }

    async fn room_members(&self, room_id: &str) -> Vec<RoomMember> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT room_id, member, profile, session_id FROM room_members \
             WHERE room_id = ?1 ORDER BY member",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map(params![room_id], |row| {
            Ok(RoomMember {
                room_id: row.get::<_, String>(0)?,
                member: row.get::<_, String>(1)?,
                profile: row.get::<_, Option<String>>(2)?.map(ProfileRef::new),
                session_id: SessionId::new(row.get::<_, String>(3)?),
            })
        });
        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn room_member_set(&self, member: RoomMember) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let profile = member.profile.as_ref().map(|p| p.as_str());
        conn.execute(
            "INSERT INTO room_members (room_id, member, profile, session_id) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(room_id, member) DO UPDATE SET profile = ?3, session_id = ?4",
            params![
                member.room_id,
                member.member,
                profile,
                member.session_id.as_str()
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn room_member_remove(&self, room_id: &str, member: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM room_members WHERE room_id = ?1 AND member = ?2",
            params![room_id, member],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn acp_list(&self) -> Vec<AcpEntry> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare("SELECT name, entry FROM acp_catalog ORDER BY name") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map([], |row| {
            Ok(AcpEntry {
                name: row.get::<_, String>(0)?,
                entry: row.get::<_, Vec<u8>>(1)?,
            })
        });
        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn acp_set(&self, entry: AcpEntry) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO acp_catalog (name, entry) VALUES (?1, ?2) \
             ON CONFLICT(name) DO UPDATE SET entry = ?2",
            params![entry.name, entry.entry],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn acp_remove(&self, name: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM acp_catalog WHERE name = ?1", params![name])
            .map_err(sql_err)?;
        Ok(())
    }

    async fn cron_list(&self) -> Vec<StoredCronJob> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT id, schedule, spec, next_fire_unix, paused, last_run_unix, last_ok, last_detail, fire_count, created_unix, owner \
             FROM cron_jobs ORDER BY id",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map([], cron_job_from_row);
        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn cron_get(&self, id: &str) -> Option<StoredCronJob> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, schedule, spec, next_fire_unix, paused, last_run_unix, last_ok, last_detail, fire_count, created_unix, owner \
             FROM cron_jobs WHERE id = ?1",
            params![id],
            cron_job_from_row,
        )
        .optional()
        .ok()
        .flatten()
    }

    async fn cron_set(&self, job: StoredCronJob) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO cron_jobs (id, schedule, spec, next_fire_unix, paused, last_run_unix, last_ok, last_detail, fire_count, created_unix, owner) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11) \
             ON CONFLICT(id) DO UPDATE SET schedule = ?2, spec = ?3, next_fire_unix = ?4, paused = ?5, \
             last_run_unix = ?6, last_ok = ?7, last_detail = ?8, fire_count = ?9, created_unix = ?10, owner = ?11",
            params![
                job.id,
                job.schedule,
                job.spec,
                job.next_fire_unix.map(|v| v as i64),
                job.paused as i64,
                job.last_run_unix.map(|v| v as i64),
                job.last_ok.map(|v| v as i64),
                job.last_detail,
                job.fire_count as i64,
                job.created_unix as i64,
                job.owner,
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn cron_remove(&self, id: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM cron_jobs WHERE id = ?1", params![id])
            .map_err(sql_err)?;
        conn.execute("DELETE FROM cron_runs WHERE job_id = ?1", params![id])
            .map_err(sql_err)?;
        Ok(())
    }

    async fn cron_due(&self, now_unix: u64) -> Vec<StoredCronJob> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT id, schedule, spec, next_fire_unix, paused, last_run_unix, last_ok, last_detail, fire_count, created_unix, owner \
             FROM cron_jobs WHERE paused = 0 AND next_fire_unix IS NOT NULL AND next_fire_unix <= ?1 \
             ORDER BY next_fire_unix",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map(params![now_unix as i64], cron_job_from_row);
        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn cron_runs_list(&self, id: &str, max: usize) -> Vec<StoredCronRun> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT job_id, started_unix, finished_unix, ok, detail, session, manual \
             FROM cron_runs WHERE job_id = ?1 ORDER BY rowseq DESC LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map(params![id, max as i64], cron_run_from_row);
        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn cron_run_append(&self, run: StoredCronRun) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO cron_runs (job_id, started_unix, finished_unix, ok, detail, session, manual) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                run.job_id,
                run.started_unix as i64,
                run.finished_unix.map(|v| v as i64),
                run.ok as i64,
                run.detail,
                run.session.as_ref().map(|s| s.as_str()),
                run.manual as i64,
            ],
        )
        .map_err(sql_err)?;
        // Bounded retention: drop all but the most recent CRON_RUN_RETENTION rows for this job.
        conn.execute(
            "DELETE FROM cron_runs WHERE job_id = ?1 AND rowseq NOT IN \
             (SELECT rowseq FROM cron_runs WHERE job_id = ?1 ORDER BY rowseq DESC LIMIT ?2)",
            params![run.job_id, CRON_RUN_RETENTION as i64],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn cron_suggestions_list(&self) -> Vec<StoredCronSuggestion> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT id, title, description, source, spec, dedup_key, status, created_unix \
             FROM cron_suggestions ORDER BY created_unix",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map([], cron_suggestion_from_row);
        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    async fn cron_suggestion_get(&self, id: &str) -> Option<StoredCronSuggestion> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, title, description, source, spec, dedup_key, status, created_unix \
             FROM cron_suggestions WHERE id = ?1",
            params![id],
            cron_suggestion_from_row,
        )
        .optional()
        .ok()
        .flatten()
    }

    async fn cron_suggestion_set(&self, s: StoredCronSuggestion) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO cron_suggestions (id, title, description, source, spec, dedup_key, status, created_unix) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
             ON CONFLICT(id) DO UPDATE SET title = ?2, description = ?3, source = ?4, spec = ?5, \
             dedup_key = ?6, status = ?7, created_unix = ?8",
            params![
                s.id,
                s.title,
                s.description,
                s.source,
                s.spec,
                s.dedup_key,
                s.status,
                s.created_unix as i64,
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    async fn cron_suggestion_remove(&self, id: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM cron_suggestions WHERE id = ?1", params![id])
            .map_err(sql_err)?;
        Ok(())
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

    async fn record_journal_seal(
        &self,
        stream: &JournalStreamId,
        seal: JournalSeal,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO journal_seals (stream, seal_cursor, retained_turns, epoch, recorded_unix) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                stream.as_str(),
                seal.seal_cursor as i64,
                seal.retained_turns as i64,
                seal.epoch as i64,
                seal.recorded_unix as i64,
            ],
        )
        .map_err(|e| StoreError::Common(DaemonError::Other(e.to_string())))?;
        Ok(())
    }

    async fn active_journal_seal(&self, stream: &JournalStreamId) -> Option<JournalSeal> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT seal_cursor, retained_turns, epoch, recorded_unix FROM journal_seals \
             WHERE stream = ?1 ORDER BY id DESC LIMIT 1",
            params![stream.as_str()],
            |row| {
                Ok(JournalSeal {
                    seal_cursor: row.get::<_, i64>(0)? as u64,
                    retained_turns: row.get::<_, i64>(1)? as u64,
                    epoch: row.get::<_, i64>(2)? as u64,
                    recorded_unix: row.get::<_, i64>(3)? as u64,
                })
            },
        )
        .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The migration ladder is internally consistent, and a fresh store is stamped to the latest
    /// `user_version` (3: `M1 = SCHEMA`, the Auth 4 ownership ALTERs, and the pending-input queue).
    #[test]
    fn migration_ladder_valid_and_applied() {
        assert!(MIGRATIONS.validate().is_ok());
        let store = SqliteStore::open_in_memory().expect("open");
        let version: i64 = store
            .conn
            .lock()
            .unwrap()
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 3, "fresh DB is stamped to the latest migration");
    }

    fn dump_schema(conn: &Connection) -> String {
        let mut stmt = conn
            .prepare("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name")
            .unwrap();
        let mut out = String::new();
        for sql in stmt.query_map([], |r| r.get::<_, String>(0)).unwrap() {
            out.push_str(sql.unwrap().trim());
            out.push_str(";\n");
        }
        out
    }

    /// On-disk schema-drift gate (the analogue of the wire `codec-drift` check): the live schema
    /// must match the committed golden. Any DDL change must be made through a new migration AND
    /// refresh the golden — run `DAEMON_UPDATE_SCHEMA=1 cargo test -p daemon-store --features sqlite
    /// schema_matches_golden`.
    #[test]
    fn schema_matches_golden() {
        let store = SqliteStore::open_in_memory().expect("open");
        let dump = dump_schema(&store.conn.lock().unwrap());
        let golden_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/schema.golden.sql");
        if std::env::var_os("DAEMON_UPDATE_SCHEMA").is_some() {
            std::fs::write(golden_path, &dump).expect("write golden");
            return;
        }
        let golden = std::fs::read_to_string(golden_path).unwrap_or_default();
        assert_eq!(
            dump.trim(),
            golden.trim(),
            "schema drift: add a migration (M::up) and refresh src/schema.golden.sql via \
             DAEMON_UPDATE_SCHEMA=1",
        );
    }

    /// FTS5 is compiled into the bundled SQLite: indexing + a `MATCH` query returns a highlighted
    /// snippet, proving the `session_fts` virtual table is usable on this build.
    #[tokio::test]
    async fn fts_search_round_trips() {
        let store = SqliteStore::open_in_memory().unwrap();
        let a = SessionId::new("sess-a");
        let b = SessionId::new("sess-b");
        store
            .index_session_text(
                &a,
                Some("Refactor".into()),
                "we refactored the parser pipeline",
            )
            .await;
        store
            .index_session_text(&b, Some("Bugfix".into()), "fixed a crash in the renderer")
            .await;

        let hits = store.search_sessions("parser", 10).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, a);
        assert_eq!(hits[0].title, "Refactor");
        assert!(
            hits[0].snippet.contains("[parser]"),
            "snippet: {}",
            hits[0].snippet
        );

        // Re-indexing replaces the prior row (no duplicate hits).
        store
            .index_session_text(&a, Some("Refactor".into()), "now about the lexer")
            .await;
        assert!(store.search_sessions("parser", 10).await.is_empty());
        assert_eq!(store.search_sessions("lexer", 10).await.len(), 1);
    }

    /// All seven `UsageDelta` fields round-trip through the enriched `session_usage` schema and
    /// accumulate additively across `record_usage` calls.
    #[tokio::test]
    async fn enriched_usage_round_trips_and_accumulates() {
        let store = SqliteStore::open_in_memory().unwrap();
        let s = SessionId::new("sess");
        let delta = UsageDelta {
            input_tokens: 100,
            output_tokens: 40,
            api_calls: 1,
            cache_read_tokens: 60,
            cache_write_tokens: 20,
            reasoning_tokens: 10,
            cost_micros: 1234,
        };
        store.record_usage(&s, delta).await;
        store.record_usage(&s, delta).await;
        let total = store.usage_of(&s).await;
        assert_eq!(total.input_tokens, 200);
        assert_eq!(total.cache_read_tokens, 120);
        assert_eq!(total.cache_write_tokens, 40);
        assert_eq!(total.reasoning_tokens, 20);
        assert_eq!(total.cost_micros, 2468);
    }

    /// Conversation-rewind seals are append-only; the latest seal for a stream is the active one.
    #[tokio::test]
    async fn journal_seal_round_trips_latest_active() {
        let store = SqliteStore::open_in_memory().unwrap();
        let stream = JournalStreamId::session(&SessionId::new("rw"));
        assert!(store.active_journal_seal(&stream).await.is_none());

        store
            .record_journal_seal(
                &stream,
                JournalSeal {
                    seal_cursor: 10,
                    retained_turns: 2,
                    epoch: 1,
                    recorded_unix: 100,
                },
            )
            .await
            .unwrap();
        store
            .record_journal_seal(
                &stream,
                JournalSeal {
                    seal_cursor: 25,
                    retained_turns: 1,
                    epoch: 2,
                    recorded_unix: 200,
                },
            )
            .await
            .unwrap();

        let active = store.active_journal_seal(&stream).await.expect("seal");
        assert_eq!(active.seal_cursor, 25);
        assert_eq!(active.retained_turns, 1);
        assert_eq!(active.epoch, 2);
        // A different stream has no seal.
        let other = JournalStreamId::session(&SessionId::new("other"));
        assert!(store.active_journal_seal(&other).await.is_none());
    }
}
