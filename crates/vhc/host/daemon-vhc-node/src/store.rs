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
    /// The run's **`RunLabel`** — the human/registry-facing handle and the `swarm_runs` primary
    /// key (decisions D1; the old string `run_id`, unchanged on the wire).
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
    /// The cryptographic **`RunId`** — the 32-byte genesis-envelope hash (ABI §8.1). `None` for a
    /// v1 row that has never joined a v2 (genesis-hash) envelope (decisions D1 lazy backfill).
    pub run_id_hash: Option<[u8; 32]>,
    /// The transition-chain epoch this row's execution identity is at (D0; `0` for v1 rows).
    pub epoch: u64,
    /// The envelope-level role label (D0; empty for v1 rows — the implicit single Trainer).
    pub role: String,
    /// The never-reused durable u64 role-instance incarnation id (ABI §8.1; `0` for v1 rows).
    pub instance: u64,
    /// Sunset observability (decisions D5): the envelope schema major (1 vs 2).
    pub envelope_schema_major: u32,
    /// Sunset observability: the admitted worker module's `da_abi` major (1 vs 2), if known.
    pub module_abi_major: Option<u32>,
    /// Sunset observability: the selected driver (`"v1"` / `"v2"`), if known.
    pub selected_driver: Option<String>,
    /// Sunset observability: the current pinned module blake3, if known.
    pub module_hash: Option<[u8; 32]>,
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
    /// The `RunLabel` ↔ `RunId` cross-check failed: the row already carries a different genesis
    /// hash than the one presented (decisions D1 — a mismatched pair is a stale label or a spoof,
    /// surfaced as a typed refusal, never silently overwritten).
    #[error("run identity mismatch for `{run_id}`: row has {existing}, presented {presented}")]
    IdentityMismatch {
        /// The `RunLabel` (row key).
        run_id: String,
        /// The hex of the hash already stored.
        existing: String,
        /// The hex of the presented hash.
        presented: String,
    },
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

/// M2 (D0): migrate `swarm.db` to the run/epoch/role-instance execution-identity model
/// (decisions D1) and add the D5 sunset-observability columns (decisions D5). **Append-only** —
/// never edit `SCHEMA`/M1. All adds are nullable or defaulted, so an existing M1 db (v1 rows)
/// migrates in place: the `run_id` TEXT primary key remains the **`RunLabel`**; the cryptographic
/// **`RunId`** (32-byte genesis hash) is the new nullable `run_id_hash`, backfilled lazily when a
/// row is next touched by a join against a v2 (genesis-hash) envelope (a v1-only row keeps a NULL
/// hash for its whole life — decisions D1 point 5). `envelope_schema_major` defaults to `1` so
/// legacy rows read as v1 for the sunset audit without a data migration.
const M2_IDENTITY_OBSERVABILITY: &str = "\
ALTER TABLE swarm_runs ADD COLUMN run_id_hash BLOB;
ALTER TABLE swarm_runs ADD COLUMN epoch INTEGER NOT NULL DEFAULT 0;
ALTER TABLE swarm_runs ADD COLUMN role TEXT NOT NULL DEFAULT '';
ALTER TABLE swarm_runs ADD COLUMN instance INTEGER NOT NULL DEFAULT 0;
ALTER TABLE swarm_runs ADD COLUMN envelope_schema_major INTEGER NOT NULL DEFAULT 1;
ALTER TABLE swarm_runs ADD COLUMN module_abi_major INTEGER;
ALTER TABLE swarm_runs ADD COLUMN selected_driver TEXT;
ALTER TABLE swarm_runs ADD COLUMN module_hash BLOB;
";

/// M3 (Phase E, decisions D6/D1): the node-durable **incarnation counter** — the never-reused,
/// monotonic u64 role-instance incarnation id the execution identity carries (ABI §8.1; a
/// reusable slot value would let a fresh role-instance inherit a retired incarnation's durable
/// sequence stream) — and the **owner-priority store** (D6 point 4: preemption priority is
/// node-side owner state, never the envelope). Append-only.
const M3_ARBITER: &str = "\
CREATE TABLE IF NOT EXISTS swarm_counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO swarm_counters (name, value) VALUES ('incarnation', 0);
CREATE TABLE IF NOT EXISTS swarm_owner_priority (
    run_id   TEXT PRIMARY KEY,
    priority INTEGER NOT NULL
);
";

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![
        M::up(SCHEMA),
        M::up(M2_IDENTITY_OBSERVABILITY),
        M::up(M3_ARBITER),
    ])
}

/// The column list every `swarm_runs` read shares — kept as one constant so the three readers
/// ([`SwarmStore::get_run`], [`SwarmStore::list_runs`], [`SwarmStore::active_intents`]) and
/// [`row_to_run`] never drift apart.
const RUN_COLUMNS: &str = "run_id, coordinator, policy_json, desired_state, credentials_ref, \
     eligibility_json, last_phase, last_round, run_id_hash, epoch, role, instance, \
     envelope_schema_major, module_abi_major, selected_driver, module_hash";

fn hex_of(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
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

    /// Record a run's D0 execution identity (decisions D1): the transition-chain `epoch`, the
    /// envelope-level `role`, and the never-reused durable u64 `instance` incarnation. Idempotent.
    pub fn set_execution_identity(
        &self,
        run_id: &str,
        epoch: u64,
        role: &str,
        instance: u64,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE swarm_runs SET epoch = ?2, role = ?3, instance = ?4, updated_ms = ?5 \
             WHERE run_id = ?1",
            params![run_id, epoch as i64, role, instance as i64, now_ms()],
        )?;
        Ok(())
    }

    /// Record the D5 sunset-observability fields for a run: the envelope schema major, the worker
    /// module's `da_abi` major, the selected driver (`"v1"`/`"v2"`), and the current module hash.
    /// The sunset audit ("no live v1 runs") is a query over these columns (decisions D5).
    pub fn set_observability(
        &self,
        run_id: &str,
        envelope_schema_major: u32,
        module_abi_major: Option<u32>,
        selected_driver: Option<&str>,
        module_hash: Option<&[u8; 32]>,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE swarm_runs SET envelope_schema_major = ?2, module_abi_major = ?3, \
             selected_driver = ?4, module_hash = ?5, updated_ms = ?6 WHERE run_id = ?1",
            params![
                run_id,
                envelope_schema_major as i64,
                module_abi_major.map(|v| v as i64),
                selected_driver,
                module_hash.map(<[u8; 32]>::as_slice),
                now_ms()
            ],
        )?;
        Ok(())
    }

    /// Deterministic-lazy **`RunLabel` → `RunId` backfill with cross-check** (decisions D1 point 5):
    /// write the 32-byte genesis hash into a row whose `run_id_hash` is still NULL; if the row
    /// already carries a hash, it MUST equal `run_id_hash` or this returns
    /// [`StoreError::IdentityMismatch`] (a typed refusal — a `RunLabel` resolving to a different
    /// genesis is a stale label or a spoof). A v1-only row is never forced to synthesize a hash;
    /// this is called only when a row is touched by a join against a v2 (genesis-hash) envelope.
    pub fn backfill_run_id(&self, run_id: &str, run_id_hash: &[u8; 32]) -> Result<(), StoreError> {
        let conn = self.lock();
        let existing: Option<Option<Vec<u8>>> = conn
            .query_row(
                "SELECT run_id_hash FROM swarm_runs WHERE run_id = ?1",
                params![run_id],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(Some(bytes)) = existing {
            if bytes.as_slice() != run_id_hash.as_slice() {
                return Err(StoreError::IdentityMismatch {
                    run_id: run_id.to_string(),
                    existing: hex_of(&bytes),
                    presented: hex_of(run_id_hash),
                });
            }
            return Ok(()); // already backfilled + agrees
        }
        conn.execute(
            "UPDATE swarm_runs SET run_id_hash = ?2, updated_ms = ?3 WHERE run_id = ?1",
            params![run_id, run_id_hash.as_slice(), now_ms()],
        )?;
        Ok(())
    }

    /// Fetch one run row, decoded (`None` if unknown).
    pub fn get_run(&self, run_id: &str) -> Result<Option<PersistedRun>, StoreError> {
        let conn = self.lock();
        conn.query_row(
            &format!("SELECT {RUN_COLUMNS} FROM swarm_runs WHERE run_id = ?1"),
            params![run_id],
            row_to_run,
        )
        .optional()?
        .transpose()
    }

    /// All run rows in `run_id` order.
    pub fn list_runs(&self) -> Result<Vec<PersistedRun>, StoreError> {
        self.query_runs(&format!(
            "SELECT {RUN_COLUMNS} FROM swarm_runs ORDER BY run_id"
        ))
    }

    /// The runs with an active join-intent (`desired_state = 'joined'`) — the set the service
    /// re-issues `JoinRun` for on restart (re-convergence).
    pub fn active_intents(&self) -> Result<Vec<PersistedRun>, StoreError> {
        self.query_runs(&format!(
            "SELECT {RUN_COLUMNS} FROM swarm_runs WHERE desired_state = 'joined' ORDER BY run_id"
        ))
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

    /// The **D5 sunset audit** (decisions D5; refactor §9): the active-intent runs that still
    /// need the retired v1 driver, judged from the D0 observability columns — never an
    /// inference. Scoped to **local desired-run state** (`desired_state = 'joined'`): local
    /// durable state cannot witness the whole world; the registry half of D5's criterion (a) and
    /// criterion (b) (registries stop accepting v1 genesis) are registry-side facts.
    ///
    /// A run needs the v1 driver iff its **module** is v1: `selected_driver = 'v1'`, or
    /// `module_abi_major = 1`, or — conservatively — a run whose module major was never recorded
    /// under a schema-major-1 envelope (a pre-observability v1-era row). A schema-major-1
    /// envelope with a **major-2 module** is mixed-fleet cell 5 — interim-supported, NOT flagged.
    pub fn v1_sunset_audit(&self) -> Result<Vec<PersistedRun>, StoreError> {
        self.query_runs(&format!(
            "SELECT {RUN_COLUMNS} FROM swarm_runs WHERE desired_state = 'joined' AND (\
                 selected_driver = 'v1' \
                 OR module_abi_major = 1 \
                 OR (module_abi_major IS NULL AND envelope_schema_major = 1)\
             ) ORDER BY run_id"
        ))
    }

    /// Mint the next **role-instance incarnation id** — never-reused, node-durable, monotonic
    /// (ABI §8.1; decisions D1). The first minted id is `1` (`0` is the pre-multi-instance
    /// default in existing rows). Atomic: the increment and the read are one SQL statement.
    pub fn mint_incarnation(&self) -> Result<u64, StoreError> {
        let conn = self.lock();
        let value: i64 = conn.query_row(
            "UPDATE swarm_counters SET value = value + 1 WHERE name = 'incarnation' \
             RETURNING value",
            [],
            |row| row.get(0),
        )?;
        Ok(value as u64)
    }

    /// Set the owner's preemption priority for a run (decisions D6 point 4: node-side owner
    /// state — the envelope can never set or influence its own priority). Idempotent upsert.
    pub fn set_run_priority(&self, run_id: &str, priority: u8) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO swarm_owner_priority (run_id, priority) VALUES (?1, ?2)
             ON CONFLICT(run_id) DO UPDATE SET priority = excluded.priority",
            params![run_id, i64::from(priority)],
        )?;
        Ok(())
    }

    /// The owner's preemption priority for a run (default `100` when unset).
    pub fn run_priority(&self, run_id: &str) -> Result<u8, StoreError> {
        let conn = self.lock();
        let p: Option<i64> = conn
            .query_row(
                "SELECT priority FROM swarm_owner_priority WHERE run_id = ?1",
                params![run_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(p.map_or(100, |v| u8::try_from(v).unwrap_or(u8::MAX)))
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
    let run_id_hash: Option<Vec<u8>> = row.get(8)?;
    let epoch: i64 = row.get(9)?;
    let role: String = row.get(10)?;
    let instance: i64 = row.get(11)?;
    let envelope_schema_major: i64 = row.get(12)?;
    let module_abi_major: Option<i64> = row.get(13)?;
    let selected_driver: Option<String> = row.get(14)?;
    let module_hash: Option<Vec<u8>> = row.get(15)?;
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
            run_id_hash: run_id_hash.and_then(|v| <[u8; 32]>::try_from(v.as_slice()).ok()),
            epoch: epoch as u64,
            role,
            instance: instance as u64,
            envelope_schema_major: envelope_schema_major as u32,
            module_abi_major: module_abi_major.map(|v| v as u32),
            selected_driver,
            module_hash: module_hash.and_then(|v| <[u8; 32]>::try_from(v.as_slice()).ok()),
        })
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::SwarmPolicyMode;

    fn v1_policy() -> SwarmPolicy {
        SwarmPolicy {
            mode: SwarmPolicyMode::Idle,
            vram_cap_mb: 8_000,
            duty_cycle_pct: 90,
            schedule: None,
        }
    }

    /// The D0 `M2` migration must upgrade an existing **M1** db (a pre-D0 v1 row) in place:
    /// the row survives, the new execution-identity columns default (RunId NULL — lazy backfill),
    /// and the observability columns read v1 (decisions D1/D5; refactor §8/D0 "upgrade test").
    #[test]
    fn m2_upgrade_preserves_v1_rows_and_defaults_new_columns() {
        let mut conn = Connection::open_in_memory().unwrap();
        // Bring the db to **M1 only** (the pre-D0 schema), then insert a legacy v1 row.
        Migrations::new(vec![M::up(SCHEMA)])
            .to_latest(&mut conn)
            .unwrap();
        let policy_json = serde_json::to_string(&v1_policy()).unwrap();
        let elig_json = serde_json::to_string(&SwarmEligibility::default()).unwrap();
        conn.execute(
            "INSERT INTO swarm_runs (run_id, coordinator, policy_json, desired_state, \
             eligibility_json) VALUES ('legacy-run', 'ws://c', ?1, 'joined', ?2)",
            params![policy_json, elig_json],
        )
        .unwrap();

        // Now run the FULL ladder (M1 + M2). rusqlite_migration applies only the missing M2.
        migrations().to_latest(&mut conn).unwrap();

        let store = SwarmStore {
            conn: Mutex::new(conn),
        };
        let run = store
            .get_run("legacy-run")
            .unwrap()
            .expect("row survives M2");
        assert_eq!(run.run_id, "legacy-run");
        assert_eq!(run.desired_state, DesiredState::Joined);
        assert_eq!(run.policy, v1_policy(), "v1 policy preserved verbatim");
        // Execution-identity columns default; RunId is NULL (lazy backfill — decisions D1).
        assert_eq!(run.run_id_hash, None);
        assert_eq!(run.epoch, 0);
        assert_eq!(run.role, "");
        assert_eq!(run.instance, 0);
        // Sunset observability: a legacy row reads as v1 without a data migration (decisions D5).
        assert_eq!(run.envelope_schema_major, 1);
        assert_eq!(run.module_abi_major, None);
        assert_eq!(run.selected_driver, None);
        assert_eq!(run.module_hash, None);
    }

    /// The deterministic-lazy `RunLabel → RunId` backfill writes a NULL hash once, is idempotent
    /// on agreement, and refuses a conflicting hash with a typed [`StoreError::IdentityMismatch`]
    /// (decisions D1 cross-check).
    #[test]
    fn run_id_backfill_is_lazy_idempotent_and_cross_checked() {
        let store = SwarmStore::open_in_memory().unwrap();
        store
            .put_join_intent(
                "run-A",
                "ws://c",
                &v1_policy(),
                None,
                &SwarmEligibility::default(),
            )
            .unwrap();
        assert_eq!(store.get_run("run-A").unwrap().unwrap().run_id_hash, None);

        let genesis = [0xABu8; 32];
        store.backfill_run_id("run-A", &genesis).unwrap();
        assert_eq!(
            store.get_run("run-A").unwrap().unwrap().run_id_hash,
            Some(genesis)
        );
        // Idempotent on agreement.
        store.backfill_run_id("run-A", &genesis).unwrap();
        // A different genesis for the same RunLabel is a typed refusal.
        let other = [0x11u8; 32];
        assert!(matches!(
            store.backfill_run_id("run-A", &other),
            Err(StoreError::IdentityMismatch { .. })
        ));
    }

    /// M3: the incarnation counter is durable, monotonic, and never reuses a value (mints
    /// survive across store re-opens on one db); owner priority round-trips with the 100
    /// default (decisions D6/D1).
    #[test]
    fn m3_incarnation_counter_and_owner_priority() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("swarm.db");

        let store = SwarmStore::open(&path).unwrap();
        assert_eq!(store.mint_incarnation().unwrap(), 1);
        assert_eq!(store.mint_incarnation().unwrap(), 2);
        drop(store);
        // A re-open (node restart) continues the durable counter — never reuses an id.
        let store = SwarmStore::open(&path).unwrap();
        assert_eq!(store.mint_incarnation().unwrap(), 3);

        assert_eq!(store.run_priority("run-X").unwrap(), 100, "default");
        store.set_run_priority("run-X", 7).unwrap();
        assert_eq!(store.run_priority("run-X").unwrap(), 7);
    }

    /// THE sunset gate's observability assert (E3 brief; decisions D5): the D0 fields
    /// (`envelope_schema_major` / `module_abi_major` / `selected_driver` / `module_hash`) are
    /// present in `swarm.db` and the "no live v1 runs" audit is a QUERY over them — a v1-module
    /// run and a pre-observability v1-era row are flagged; a cell-5 run (v1 envelope, major-2
    /// module), a v2-genesis run, and a LEFT v1 run are not.
    #[test]
    fn v1_sunset_audit_is_a_query_over_the_d0_observability_fields() {
        let store = SwarmStore::open_in_memory().unwrap();
        let put = |id: &str| {
            store
                .put_join_intent(id, "c", &v1_policy(), None, &SwarmEligibility::default())
                .unwrap();
        };
        // A v1-module run (the thing the sunset drains).
        put("v1-module");
        store
            .set_observability("v1-module", 1, Some(1), Some("v1"), None)
            .unwrap();
        // A pre-observability v1-era row: schema-major-1, module major never recorded.
        put("v1-era-unknown");
        store
            .set_observability("v1-era-unknown", 1, None, None, None)
            .unwrap();
        // Cell 5: a major-2 module under a v1 envelope — interim-supported, NOT flagged.
        put("cell5");
        store
            .set_observability("cell5", 1, Some(2), Some("v2"), Some(&[0x55; 32]))
            .unwrap();
        // A v2-genesis run.
        put("v2-genesis");
        store
            .set_observability("v2-genesis", 2, Some(2), Some("v2"), Some(&[0x66; 32]))
            .unwrap();
        // A v1 run that already LEFT (desired_state != joined): outside "live" scope.
        put("v1-left");
        store
            .set_observability("v1-left", 1, Some(1), Some("v1"), None)
            .unwrap();
        store
            .set_desired_state("v1-left", DesiredState::Left)
            .unwrap();

        let flagged: Vec<String> = store
            .v1_sunset_audit()
            .unwrap()
            .into_iter()
            .map(|r| r.run_id)
            .collect();
        assert_eq!(flagged, ["v1-era-unknown", "v1-module"]);

        // Draining the flagged intents empties the audit — the D5 "no live v1 runs" half (a),
        // scoped to local desired-run state.
        store
            .set_desired_state("v1-module", DesiredState::Left)
            .unwrap();
        store
            .set_desired_state("v1-era-unknown", DesiredState::Left)
            .unwrap();
        assert!(store.v1_sunset_audit().unwrap().is_empty());
    }

    /// The v2 execution-identity + observability writers populate the D0 columns for the sunset
    /// audit query (decisions D5).
    #[test]
    fn v2_identity_and_observability_writers() {
        let store = SwarmStore::open_in_memory().unwrap();
        store
            .put_join_intent(
                "run-B",
                "iroh://c",
                &v1_policy(),
                None,
                &SwarmEligibility::default(),
            )
            .unwrap();
        store
            .set_execution_identity("run-B", 3, "worker", 42)
            .unwrap();
        let mh = [0x22u8; 32];
        store
            .set_observability("run-B", 2, Some(2), Some("v2"), Some(&mh))
            .unwrap();
        let run = store.get_run("run-B").unwrap().unwrap();
        assert_eq!(
            (run.epoch, run.role.as_str(), run.instance),
            (3, "worker", 42)
        );
        assert_eq!(run.envelope_schema_major, 2);
        assert_eq!(run.module_abi_major, Some(2));
        assert_eq!(run.selected_driver.as_deref(), Some("v2"));
        assert_eq!(run.module_hash, Some(mh));
    }
}
