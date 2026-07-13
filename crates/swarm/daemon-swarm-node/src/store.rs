// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `swarm.db` — the node's durable swarm-participation state (spec §10.3).
//!
//! A separate SQLite file (kept out of the session store to stay lean), opened + migrated the same
//! way as `daemon-auth`'s `auth.db`: bundled amalgamation, WAL, and `PRAGMA user_version` migrations
//! via `rusqlite_migration` (append-only — never edit a released `M`). Three tables:
//!
//! - `swarm_runs` — the joined-run intents + status. `desired_state` is the **durable join-intent**
//!   flag (ADR-006 idempotent intents); the node re-converges on restart by re-issuing `JoinRun` for
//!   every row with `desired_state = 'joined'` ([`SwarmStore::active_intents`]). Each row carries the
//!   node-computed `eligibility` (ADR-003 mirror) so the app never re-derives it.
//! - `swarm_contrib` — per-run contribution counters (the "what did my GPU do" ledger).
//! - `swarm_events` — the windowed (ADR-007) recent event log for the UI; pruned to a bounded ring
//!   per run on every append.

use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_api::{SwarmContribution, SwarmEligibility, SwarmEvent, SwarmPolicy};
use rusqlite::{params, Connection, OptionalExtension};
use rusqlite_migration::{Migrations, M};

/// How many recent events per run the windowed `swarm_events` log retains (ADR-007).
pub const EVENT_WINDOW: usize = 256;

/// The durable desired-state flag for a run (the join-intent that drives restart re-convergence).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DesiredState {
    /// The node intends to participate (rejoined on restart).
    Joined,
    /// The node has left (retained for the contribution ledger; not rejoined).
    Left,
}

impl DesiredState {
    fn as_str(self) -> &'static str {
        match self {
            DesiredState::Joined => "joined",
            DesiredState::Left => "left",
        }
    }
    fn from_str(s: &str) -> Self {
        match s {
            "joined" => DesiredState::Joined,
            _ => DesiredState::Left,
        }
    }
}

/// A persisted run row (spec §10.3 `swarm_runs`), decoded into typed form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistedRun {
    /// The run id.
    pub run_id: String,
    /// The coordinator endpoint discovery/join used.
    pub coordinator: String,
    /// The participation policy the node joined under.
    pub policy: SwarmPolicy,
    /// The durable join-intent (drives restart re-convergence).
    pub desired_state: DesiredState,
    /// An opaque credential store reference (daemon-credentials), if any.
    pub credentials_ref: Option<String>,
    /// The node-computed eligibility (ADR-003 mirror; the app renders it, never re-derives it).
    pub eligibility: SwarmEligibility,
    /// The last-known phase string.
    pub last_phase: String,
    /// The last-known round.
    pub last_round: u64,
}

/// A `swarm.db` error.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// An underlying SQLite error.
    #[error("sqlite: {0}")]
    Sql(#[from] rusqlite::Error),
    /// A migration failure.
    #[error("migrate: {0}")]
    Migrate(String),
    /// A JSON (de)serialization failure for a stored policy / eligibility / event blob.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// M1: the full schema (spec §10.3). `IF NOT EXISTS` makes a fresh open idempotent; later schema
/// changes append a new `M::up("ALTER …")` and NEVER edit this one.
const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS swarm_runs (
    run_id           TEXT PRIMARY KEY,
    coordinator      TEXT NOT NULL,
    policy_json      TEXT NOT NULL,
    desired_state    TEXT NOT NULL,
    credentials_ref  TEXT,
    eligibility_json TEXT NOT NULL,
    last_phase       TEXT NOT NULL DEFAULT '',
    last_round       INTEGER NOT NULL DEFAULT 0,
    updated_ms       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS swarm_contrib (
    run_id             TEXT PRIMARY KEY,
    rounds             INTEGER NOT NULL DEFAULT 0,
    tokens             INTEGER NOT NULL DEFAULT 0,
    bytes_up           INTEGER NOT NULL DEFAULT 0,
    bytes_down         INTEGER NOT NULL DEFAULT 0,
    witness_count      INTEGER NOT NULL DEFAULT 0,
    checkpoint_credits INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS swarm_events (
    seq    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts_ms  INTEGER NOT NULL,
    kind   TEXT NOT NULL,
    body   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS swarm_events_run ON swarm_events (run_id, seq);
";

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![M::up(SCHEMA)])
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// The durable swarm-state store (`swarm.db`).
pub struct SwarmStore {
    conn: Mutex<Connection>,
}

impl SwarmStore {
    /// Open (creating if absent) and migrate `swarm.db` at `path`. The parent directory must already
    /// exist (the node creates its `data_dir`). Idempotent: re-opening an existing db re-runs the
    /// migration ladder to the same `user_version` (a no-op).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let mut conn = Connection::open(path)?;
        Self::prepare(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an in-memory db (tests).
    pub fn open_in_memory() -> Result<Self, StoreError> {
        let mut conn = Connection::open_in_memory()?;
        Self::prepare(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn prepare(conn: &mut Connection) -> Result<(), StoreError> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        migrations()
            .to_latest(conn)
            .map_err(|e| StoreError::Migrate(e.to_string()))?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Persist (or refresh) a joined-run intent: sets `desired_state = joined`, records the
    /// coordinator + policy + node-computed eligibility, and ensures a contribution row exists.
    /// Idempotent per `run_id` (ADR-006): a repeated join with the same or a new policy converges to
    /// the latest, never duplicating.
    pub fn put_join_intent(
        &self,
        run_id: &str,
        coordinator: &str,
        policy: &SwarmPolicy,
        credentials_ref: Option<&str>,
        eligibility: &SwarmEligibility,
    ) -> Result<(), StoreError> {
        let policy_json = serde_json::to_string(policy)?;
        let elig_json = serde_json::to_string(eligibility)?;
        let conn = self.lock();
        conn.execute(
            "INSERT INTO swarm_runs
                (run_id, coordinator, policy_json, desired_state, credentials_ref,
                 eligibility_json, last_phase, last_round, updated_ms)
             VALUES (?1, ?2, ?3, 'joined', ?4, ?5, '', 0, ?6)
             ON CONFLICT(run_id) DO UPDATE SET
                coordinator      = excluded.coordinator,
                policy_json      = excluded.policy_json,
                desired_state    = 'joined',
                credentials_ref  = excluded.credentials_ref,
                eligibility_json = excluded.eligibility_json,
                updated_ms       = excluded.updated_ms",
            params![
                run_id,
                coordinator,
                policy_json,
                credentials_ref,
                elig_json,
                now_ms()
            ],
        )?;
        conn.execute(
            "INSERT OR IGNORE INTO swarm_contrib (run_id) VALUES (?1)",
            params![run_id],
        )?;
        Ok(())
    }

    /// Flip a run's durable desired-state (a leave keeps the row + contribution ledger). Idempotent.
    pub fn set_desired_state(&self, run_id: &str, state: DesiredState) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE swarm_runs SET desired_state = ?2, updated_ms = ?3 WHERE run_id = ?1",
            params![run_id, state.as_str(), now_ms()],
        )?;
        Ok(())
    }

    /// Record the node-computed eligibility for a run (ADR-003 mirror).
    pub fn set_eligibility(
        &self,
        run_id: &str,
        eligibility: &SwarmEligibility,
    ) -> Result<(), StoreError> {
        let elig_json = serde_json::to_string(eligibility)?;
        self.lock().execute(
            "UPDATE swarm_runs SET eligibility_json = ?2, updated_ms = ?3 WHERE run_id = ?1",
            params![run_id, elig_json, now_ms()],
        )?;
        Ok(())
    }

    /// Update a run's last-known phase + round (from a worker `RunPhase` event).
    pub fn set_phase(&self, run_id: &str, phase: &str, round: u64) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE swarm_runs SET last_phase = ?2, last_round = ?3, updated_ms = ?4
             WHERE run_id = ?1",
            params![run_id, phase, round as i64, now_ms()],
        )?;
        Ok(())
    }

    /// Fetch one run row, decoded (`None` if unknown).
    pub fn get_run(&self, run_id: &str) -> Result<Option<PersistedRun>, StoreError> {
        let conn = self.lock();
        conn.query_row(
            "SELECT run_id, coordinator, policy_json, desired_state, credentials_ref,
                    eligibility_json, last_phase, last_round
             FROM swarm_runs WHERE run_id = ?1",
            params![run_id],
            row_to_run,
        )
        .optional()?
        .transpose()
    }

    /// All run rows in `run_id` order.
    pub fn list_runs(&self) -> Result<Vec<PersistedRun>, StoreError> {
        self.query_runs(
            "SELECT run_id, coordinator, policy_json, desired_state, credentials_ref, \
             eligibility_json, last_phase, last_round FROM swarm_runs ORDER BY run_id",
        )
    }

    /// The runs with an active join-intent (`desired_state = 'joined'`) — the set the service
    /// re-issues `JoinRun` for on restart (re-convergence).
    pub fn active_intents(&self) -> Result<Vec<PersistedRun>, StoreError> {
        self.query_runs(
            "SELECT run_id, coordinator, policy_json, desired_state, credentials_ref, \
             eligibility_json, last_phase, last_round FROM swarm_runs \
             WHERE desired_state = 'joined' ORDER BY run_id",
        )
    }

    fn query_runs(&self, sql: &str) -> Result<Vec<PersistedRun>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map([], row_to_run)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r??);
        }
        Ok(out)
    }

    /// A run's contribution counters (zeros if no row yet).
    pub fn get_contribution(&self, run_id: &str) -> Result<SwarmContribution, StoreError> {
        let conn = self.lock();
        let c = conn
            .query_row(
                "SELECT rounds, tokens, bytes_up, bytes_down, witness_count, checkpoint_credits
                 FROM swarm_contrib WHERE run_id = ?1",
                params![run_id],
                |row| {
                    Ok(SwarmContribution {
                        rounds: row.get::<_, i64>(0)? as u64,
                        tokens: row.get::<_, i64>(1)? as u64,
                        bytes_up: row.get::<_, i64>(2)? as u64,
                        bytes_down: row.get::<_, i64>(3)? as u64,
                        witness_count: row.get::<_, i64>(4)? as u64,
                        checkpoint_credits: row.get::<_, i64>(5)? as u64,
                    })
                },
            )
            .optional()?;
        Ok(c.unwrap_or_default())
    }

    /// Add deltas to a run's contribution counters (creating the row if needed).
    #[allow(clippy::too_many_arguments)]
    pub fn bump_contribution(
        &self,
        run_id: &str,
        rounds: u64,
        tokens: u64,
        bytes_up: u64,
        bytes_down: u64,
        witness_count: u64,
        checkpoint_credits: u64,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO swarm_contrib
                (run_id, rounds, tokens, bytes_up, bytes_down, witness_count, checkpoint_credits)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(run_id) DO UPDATE SET
                rounds             = rounds + excluded.rounds,
                tokens             = tokens + excluded.tokens,
                bytes_up           = bytes_up + excluded.bytes_up,
                bytes_down         = bytes_down + excluded.bytes_down,
                witness_count      = witness_count + excluded.witness_count,
                checkpoint_credits = checkpoint_credits + excluded.checkpoint_credits",
            params![
                run_id,
                rounds as i64,
                tokens as i64,
                bytes_up as i64,
                bytes_down as i64,
                witness_count as i64,
                checkpoint_credits as i64
            ],
        )?;
        Ok(())
    }

    /// Append an event to the windowed log for a run, then prune to the newest [`EVENT_WINDOW`]
    /// (ADR-007). The event body is JSON (`SwarmEvent`), keyed by `kind` for cheap filtering.
    pub fn append_event(&self, event: &SwarmEvent) -> Result<(), StoreError> {
        let body = serde_json::to_vec(event)?;
        let run_id = event.run_id().to_string();
        let conn = self.lock();
        conn.execute(
            "INSERT INTO swarm_events (run_id, ts_ms, kind, body) VALUES (?1, ?2, ?3, ?4)",
            params![run_id, now_ms(), event.kind(), body],
        )?;
        conn.execute(
            "DELETE FROM swarm_events WHERE run_id = ?1 AND seq NOT IN
                (SELECT seq FROM swarm_events WHERE run_id = ?1 ORDER BY seq DESC LIMIT ?2)",
            params![run_id, EVENT_WINDOW as i64],
        )?;
        Ok(())
    }

    /// The most recent events for a run in chronological order (oldest → newest), capped at `limit`.
    pub fn recent_events(&self, run_id: &str, limit: usize) -> Result<Vec<SwarmEvent>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT body FROM swarm_events WHERE run_id = ?1 ORDER BY seq DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![run_id, limit as i64], |row| {
            row.get::<_, Vec<u8>>(0)
        })?;
        let mut out = Vec::new();
        for r in rows {
            let bytes = r?;
            out.push(serde_json::from_slice::<SwarmEvent>(&bytes)?);
        }
        out.reverse();
        Ok(out)
    }

    /// The number of events retained for a run (test/observability helper).
    pub fn event_count(&self, run_id: &str) -> Result<usize, StoreError> {
        let conn = self.lock();
        let n: i64 = conn.query_row(
            "SELECT COUNT(*) FROM swarm_events WHERE run_id = ?1",
            params![run_id],
            |row| row.get(0),
        )?;
        Ok(n as usize)
    }
}

/// Decode a `swarm_runs` row into a [`PersistedRun`]. The JSON columns decode outside the rusqlite
/// closure (its error type is `rusqlite::Error`), so the closure yields a `Result<PersistedRun,
/// StoreError>` that the caller flattens.
fn row_to_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<Result<PersistedRun, StoreError>> {
    let run_id: String = row.get(0)?;
    let coordinator: String = row.get(1)?;
    let policy_json: String = row.get(2)?;
    let desired: String = row.get(3)?;
    let credentials_ref: Option<String> = row.get(4)?;
    let elig_json: String = row.get(5)?;
    let last_phase: String = row.get(6)?;
    let last_round: i64 = row.get(7)?;
    Ok((|| {
        Ok(PersistedRun {
            run_id,
            coordinator,
            policy: serde_json::from_str(&policy_json)?,
            desired_state: DesiredState::from_str(&desired),
            credentials_ref,
            eligibility: serde_json::from_str(&elig_json)?,
            last_phase,
            last_round: last_round as u64,
        })
    })())
}
