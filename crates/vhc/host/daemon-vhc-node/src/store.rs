// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `vhc.db` — the node's durable vhc-participation state (spec §10.3).
//!
//! A separate SQLite file (kept out of the session store to stay lean), opened + migrated the same
//! way as `daemon-auth`'s `auth.db`: bundled amalgamation, WAL, and `PRAGMA user_version` migrations
//! via `rusqlite_migration` (append-only — never edit a released `M`). Three tables:
//!
//! - `vhc_runs` — the joined-run intents + status. `desired_state` is the **durable join-intent**
//!   flag (ADR-006 idempotent intents); the node re-converges on restart by re-issuing `JoinRun` for
//!   every row with `desired_state = 'joined'` ([`VhcStore::active_intents`]). Each row carries the
//!   node-computed `eligibility` (ADR-003 mirror) so the app never re-derives it.
//! - `vhc_contrib` — per-run contribution counters (the "what did my GPU do" ledger).
//! - `vhc_events` — the windowed (ADR-007) recent event log for the UI; pruned to a bounded ring
//!   per run on every append.

use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_api::{VhcContribution, VhcEligibility, VhcEvent, VhcPolicy};
use rusqlite::{params, Connection, OptionalExtension};
use rusqlite_migration::{Migrations, M};

/// How many recent events per run the windowed `vhc_events` log retains (ADR-007).
pub const EVENT_WINDOW: usize = 256;

/// The durable desired-state flag for a run — the OWNER-INTENT axis of the two-axis run-instance
/// lifecycle (architecture §6.4). Intent alone decides whether the node
/// still wants in; the OBSERVED [`RunState`] axis decides whether the last instance may resume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DesiredState {
    /// The node intends to participate (rejoined on restart).
    Joined,
    /// The owner paused the run: durable intent surviving restart — a paused run is never
    /// reconverged (and holds no ledger reservation) until the owner resumes it.
    Paused,
    /// The node has left (retained for the contribution ledger; not rejoined).
    Left,
}

impl DesiredState {
    fn as_str(self) -> &'static str {
        match self {
            DesiredState::Joined => "joined",
            DesiredState::Paused => "paused",
            DesiredState::Left => "left",
        }
    }
    fn from_str(s: &str) -> Self {
        match s {
            "joined" => DesiredState::Joined,
            "paused" => DesiredState::Paused,
            _ => DesiredState::Left,
        }
    }
}

/// The OBSERVED per-incarnation lifecycle — the second axis of the run-instance state machine.
/// Driven by worker terminal events (`RunTerminated`) and pump-stream observation, never by owner
/// intent. A `Completed` instance is NEVER rejoined; `FailedTerminal` requires owner action;
/// `FailedRetryable` reconverges under the retry budget; an observed `Left` with a standing
/// `joined` intent (e.g. a shutdown drain) reconverges on restart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunState {
    /// A join transaction is in flight: the reservation is held and the worker supervised, but
    /// session readiness (`RunPhase "running"`) has not been observed. A `Starting` instance is
    /// attributable (errors and stream closure land on it) but never *published* as running —
    /// the promotion is event-driven, not command-write-driven.
    Starting,
    /// A live role instance: session readiness was OBSERVED (never inferred from a dispatched
    /// command).
    Running,
    /// The module signaled run end. Terminal: never restarted.
    Completed,
    /// A recoverable environment fault; the reconciliation loop reconverges under the retry
    /// budget while the owner intent stands.
    FailedRetryable,
    /// A non-recoverable failure (trap, admission/cert refusal, retry-budget exhaustion).
    /// Terminal: owner action required.
    FailedTerminal,
    /// The instance ended by a leave command (owner leave or shutdown drain).
    Left,
}

impl RunState {
    /// The stable column token (also the app-facing lifecycle string).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            RunState::Starting => "starting",
            RunState::Running => "running",
            RunState::Completed => "completed",
            RunState::FailedRetryable => "failed_retryable",
            RunState::FailedTerminal => "failed_terminal",
            RunState::Left => "left",
        }
    }
    fn from_str(s: &str) -> Self {
        match s {
            "starting" => RunState::Starting,
            "completed" => RunState::Completed,
            "failed_retryable" => RunState::FailedRetryable,
            "failed_terminal" => RunState::FailedTerminal,
            "left" => RunState::Left,
            _ => RunState::Running,
        }
    }

    /// Whether a fresh join for this run may RETAIN the persisted incarnation (a live or
    /// recoverable instance) — terminal states always mint a new one. A `Starting` row (the node
    /// died mid-transaction) is recoverable: the restart reconverges the intent.
    #[must_use]
    pub fn resumable(self) -> bool {
        matches!(
            self,
            RunState::Starting | RunState::Running | RunState::FailedRetryable
        )
    }
}

/// The app-facing effective lifecycle state: the owner-intent axis projected over the observed
/// axis (spec §6.1 — the six-state view `running | completed | paused | failed_retryable |
/// failed_terminal | left`). `Paused` intent masks a non-terminal observed state; terminal
/// observations win (a completed run reads completed even while "paused").
#[must_use]
pub fn effective_state(desired: DesiredState, run_state: RunState) -> &'static str {
    match (desired, run_state) {
        (_, RunState::Completed) => "completed",
        (_, RunState::FailedTerminal) => "failed_terminal",
        (DesiredState::Paused, _) => "paused",
        (DesiredState::Left, _) => "left",
        (DesiredState::Joined, s) => s.as_str(),
    }
}

/// A persisted run row (spec §10.3 `vhc_runs`), decoded into typed form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistedRun {
    /// The run's **`RunLabel`** — the human/registry-facing handle and the `vhc_runs` primary
    /// key (decisions D1; the old string `run_id`, unchanged on the wire).
    pub run_id: String,
    /// The coordinator endpoint discovery/join used.
    pub coordinator: String,
    /// The participation policy the node joined under.
    pub policy: VhcPolicy,
    /// The durable join-intent (drives restart re-convergence).
    pub desired_state: DesiredState,
    /// An opaque credential store reference (daemon-credentials), if any.
    pub credentials_ref: Option<String>,
    /// The node-computed eligibility (ADR-003 mirror; the app renders it, never re-derives it).
    pub eligibility: VhcEligibility,
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
    /// The immutable admitted tuple this join intent was assessed under (architecture §6.3),
    /// canonical-CBOR. `None` for a row assessed before the tuple existed.
    pub admitted_tuple: Option<Vec<u8>>,
    /// The OBSERVED per-incarnation lifecycle state (the second axis; see [`RunState`]).
    pub run_state: RunState,
    /// The crash-window release marker: `Some(target)` means worker teardown was OBSERVED and the
    /// ledger release + terminal commit were in flight when the row was last written. Startup
    /// reconciliation finishes the transition (`run_state = target`, marker cleared).
    pub pending_run_state: Option<RunState>,
    /// Consecutive reconvergence attempts since the last stable interval (the retry budget).
    pub retry_count: u32,
    /// When the next reconvergence attempt is due (unix ms); `None` = none scheduled.
    pub next_retry_ms: Option<i64>,
    /// When the current incarnation (re)entered `Running` (unix ms); the uptime-reset input.
    pub running_since_ms: Option<i64>,
    /// The typed reason recorded with a terminal transition (operator-facing detail).
    pub terminal_reason: Option<String>,
    /// The storage gate (M8): the last terminal was `HostStorageExhausted`, so the reconcile
    /// loop redispatches this `failed_retryable` row only after the free-space check passes —
    /// and a gated wait never consumes the retry budget.
    pub storage_gated: bool,
}

/// A `vhc.db` error.
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
    /// A node-owned monotonic counter reached the bounded ordinal ceiling (`i64::MAX`, the
    /// shared domain across SQLite / the wire / the TS registry) — identity minting on this
    /// installation is exhausted; typed, never a silent wrap.
    #[error("counter `{0}` exhausted: the bounded ordinal domain (i64::MAX) is spent")]
    CounterExhausted(&'static str),
    /// A presented ordinal (e.g. a floor to mint above) is outside the bounded domain
    /// (`> i64::MAX`) — refused typed, never truncated into a counter.
    #[error("ordinal {0} is outside the bounded domain (max {max})", max = i64::MAX)]
    OrdinalOutOfDomain(u64),
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
CREATE TABLE IF NOT EXISTS vhc_runs (
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
CREATE TABLE IF NOT EXISTS vhc_contrib (
    run_id             TEXT PRIMARY KEY,
    rounds             INTEGER NOT NULL DEFAULT 0,
    tokens             INTEGER NOT NULL DEFAULT 0,
    bytes_up           INTEGER NOT NULL DEFAULT 0,
    bytes_down         INTEGER NOT NULL DEFAULT 0,
    witness_count      INTEGER NOT NULL DEFAULT 0,
    checkpoint_credits INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS vhc_events (
    seq    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts_ms  INTEGER NOT NULL,
    kind   TEXT NOT NULL,
    body   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS vhc_events_run ON vhc_events (run_id, seq);
";

/// M2 (D0): migrate `vhc.db` to the run/epoch/role-instance execution-identity model
/// (decisions D1) and add the D5 sunset-observability columns (decisions D5). **Append-only** —
/// never edit `SCHEMA`/M1. All adds are nullable or defaulted, so an existing M1 db (v1 rows)
/// migrates in place: the `run_id` TEXT primary key remains the **`RunLabel`**; the cryptographic
/// **`RunId`** (32-byte genesis hash) is the new nullable `run_id_hash`, backfilled lazily when a
/// row is next touched by a join against a v2 (genesis-hash) envelope (a v1-only row keeps a NULL
/// hash for its whole life — decisions D1 point 5). `envelope_schema_major` defaults to `1` so
/// legacy rows read as v1 for the sunset audit without a data migration.
const M2_IDENTITY_OBSERVABILITY: &str = "\
ALTER TABLE vhc_runs ADD COLUMN run_id_hash BLOB;
ALTER TABLE vhc_runs ADD COLUMN epoch INTEGER NOT NULL DEFAULT 0;
ALTER TABLE vhc_runs ADD COLUMN role TEXT NOT NULL DEFAULT '';
ALTER TABLE vhc_runs ADD COLUMN instance INTEGER NOT NULL DEFAULT 0;
ALTER TABLE vhc_runs ADD COLUMN envelope_schema_major INTEGER NOT NULL DEFAULT 1;
ALTER TABLE vhc_runs ADD COLUMN module_abi_major INTEGER;
ALTER TABLE vhc_runs ADD COLUMN selected_driver TEXT;
ALTER TABLE vhc_runs ADD COLUMN module_hash BLOB;
";

/// M3 (Phase E, decisions D6/D1): the node-durable **incarnation counter** — the never-reused,
/// monotonic u64 role-instance incarnation id the execution identity carries (ABI §8.1; a
/// reusable slot value would let a fresh role-instance inherit a retired incarnation's durable
/// sequence stream) — and the **owner-priority store** (D6 point 4: preemption priority is
/// node-side owner state, never the envelope). Append-only.
const M3_ARBITER: &str = "\
CREATE TABLE IF NOT EXISTS vhc_counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO vhc_counters (name, value) VALUES ('incarnation', 0);
CREATE TABLE IF NOT EXISTS vhc_owner_priority (
    run_id   TEXT PRIMARY KEY,
    priority INTEGER NOT NULL
);
";

/// M4: the immutable **admitted tuple** (architecture §6.3) stored alongside the durable join
/// intent, plus the two node-owned monotonic revision counters the tuple stamps — the
/// device-profile revision (bumped when the assessed device profile's canonical encoding changes)
/// and the owner-policy revision (bumped on any owner policy mutation). Append-only; the tuple
/// column is nullable (a pre-tuple row carries NULL), and the counters seed at 0.
const M4_ADMITTED_TUPLE: &str = "\
ALTER TABLE vhc_runs ADD COLUMN admitted_tuple BLOB;
INSERT OR IGNORE INTO vhc_counters (name, value) VALUES ('device_profile_rev', 0);
INSERT OR IGNORE INTO vhc_counters (name, value) VALUES ('owner_policy_rev', 0);
";

/// M5: the durable **run-instance state machine** columns (the two-axis lifecycle; architecture
/// §6.4; ABI §12.10). `run_state` is the observed per-incarnation lifecycle
/// (`running | completed | failed_retryable | failed_terminal | left`); `pending_run_state` is the
/// crash-window release marker (teardown observed, terminal commit in flight — startup
/// reconciliation finishes it); the retry columns carry the bounded reconvergence budget; and
/// `running_since_ms` feeds the uptime-based retry reset. Append-only; every add is defaulted or
/// nullable, so pre-M5 rows read as live (`running`) intents and are re-judged by the startup
/// reconciliation pass.
const M5_RUN_LIFECYCLE: &str = "\
ALTER TABLE vhc_runs ADD COLUMN run_state TEXT NOT NULL DEFAULT 'running';
ALTER TABLE vhc_runs ADD COLUMN pending_run_state TEXT;
ALTER TABLE vhc_runs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE vhc_runs ADD COLUMN next_retry_ms INTEGER;
ALTER TABLE vhc_runs ADD COLUMN running_since_ms INTEGER;
ALTER TABLE vhc_runs ADD COLUMN terminal_reason TEXT;
";

/// M6: the node-local mirror of each run's **committed transition-chain records** (architecture
/// §5.4): one row per committed epoch, holding the canonical-CBOR `UpgradeRecord` the node
/// validated and consumed. The mirror is what lets the next switch's fail-closed validation
/// rebuild the chain from genesis without refetching history; it is written only after the
/// record authorized + appended cleanly AND the local switch activated. Append-only.
const M6_UPGRADE_RECORDS: &str = "\
CREATE TABLE IF NOT EXISTS vhc_upgrade_records (
    run_id     TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    record     BLOB NOT NULL,
    created_ms INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, epoch)
);
";

/// M7: the node's **highest verified leadership term** per `(run, role)` seat slot ([SEAT-1]
/// v2 — leadership terms are a separate order relation from execution incarnations, so they get
/// separate durable state, never the `vhc_counters` execution counter). Fed only by verified
/// evidence: seat grants this node authored and won, and stored leases that passed the full
/// peer-side authorization. Monotonic MAX per slot; never advanced by naked registry metadata.
const M7_SEAT_TERMS: &str = "\
CREATE TABLE IF NOT EXISTS vhc_seat_terms (
    run_id   TEXT NOT NULL,
    role     TEXT NOT NULL,
    term     INTEGER NOT NULL,
    claimant BLOB NOT NULL,
    PRIMARY KEY (run_id, role)
);
";

/// M8: the **storage gate** flag (the typed storage taxonomy, ABI §12.10): set when an instance
/// terminated `HostStorageExhausted` (ENOSPC/quota — a host capacity condition, never a module
/// fault). A gated `failed_retryable` row is redispatched by the reconcile loop only once the
/// node-state filesystem's free-space check passes, and a gated wait never consumes the retry
/// budget. Interim mechanism until the disk custodian owns resume authorization. Append-only.
const M8_STORAGE_GATE: &str = "\
ALTER TABLE vhc_runs ADD COLUMN storage_gated INTEGER NOT NULL DEFAULT 0;
";

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![
        M::up(SCHEMA),
        M::up(M2_IDENTITY_OBSERVABILITY),
        M::up(M3_ARBITER),
        M::up(M4_ADMITTED_TUPLE),
        M::up(M5_RUN_LIFECYCLE),
        M::up(M6_UPGRADE_RECORDS),
        M::up(M7_SEAT_TERMS),
        M::up(M8_STORAGE_GATE),
    ])
}

/// The column list every `vhc_runs` read shares — kept as one constant so the three readers
/// ([`VhcStore::get_run`], [`VhcStore::list_runs`], [`VhcStore::active_intents`]) and
/// [`row_to_run`] never drift apart.
const RUN_COLUMNS: &str = "run_id, coordinator, policy_json, desired_state, credentials_ref, \
     eligibility_json, last_phase, last_round, run_id_hash, epoch, role, instance, \
     envelope_schema_major, module_abi_major, selected_driver, module_hash, admitted_tuple, \
     run_state, pending_run_state, retry_count, next_retry_ms, running_since_ms, terminal_reason, \
     storage_gated";

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

/// The durable vhc-state store (`vhc.db`).
pub struct VhcStore {
    conn: Mutex<Connection>,
}

impl VhcStore {
    /// Open (creating if absent) and migrate `vhc.db` at `path`. The parent directory must already
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
        policy: &VhcPolicy,
        credentials_ref: Option<&str>,
        eligibility: &VhcEligibility,
    ) -> Result<(), StoreError> {
        let policy_json = serde_json::to_string(policy)?;
        let elig_json = serde_json::to_string(eligibility)?;
        let conn = self.lock();
        // An owner join (re)arms the lifecycle: intent joined, the observed state STARTING —
        // never 'running', which is an observation the worker's session readiness event makes
        // ([`Self::mark_running`]), not a fact a dispatched command establishes — and the
        // retry/terminal bookkeeping of any previous incarnation cleared (an explicit join is the
        // owner action that reopens a terminal row — with a freshly-minted incarnation).
        conn.execute(
            "INSERT INTO vhc_runs
                (run_id, coordinator, policy_json, desired_state, credentials_ref,
                 eligibility_json, last_phase, last_round, updated_ms, run_state,
                 running_since_ms)
             VALUES (?1, ?2, ?3, 'joined', ?4, ?5, '', 0, ?6, 'starting', ?6)
             ON CONFLICT(run_id) DO UPDATE SET
                coordinator       = excluded.coordinator,
                policy_json       = excluded.policy_json,
                desired_state     = 'joined',
                credentials_ref   = excluded.credentials_ref,
                eligibility_json  = excluded.eligibility_json,
                updated_ms        = excluded.updated_ms,
                run_state         = 'starting',
                pending_run_state = NULL,
                retry_count       = 0,
                next_retry_ms     = NULL,
                running_since_ms  = excluded.updated_ms,
                terminal_reason   = NULL,
                storage_gated     = 0",
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
            "INSERT OR IGNORE INTO vhc_contrib (run_id) VALUES (?1)",
            params![run_id],
        )?;
        Ok(())
    }

    /// Flip a run's durable desired-state (a leave keeps the row + contribution ledger). Idempotent.
    pub fn set_desired_state(&self, run_id: &str, state: DesiredState) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET desired_state = ?2, updated_ms = ?3 WHERE run_id = ?1",
            params![run_id, state.as_str(), now_ms()],
        )?;
        Ok(())
    }

    /// Record the node-computed eligibility for a run (ADR-003 mirror).
    pub fn set_eligibility(
        &self,
        run_id: &str,
        eligibility: &VhcEligibility,
    ) -> Result<(), StoreError> {
        let elig_json = serde_json::to_string(eligibility)?;
        self.lock().execute(
            "UPDATE vhc_runs SET eligibility_json = ?2, updated_ms = ?3 WHERE run_id = ?1",
            params![run_id, elig_json, now_ms()],
        )?;
        Ok(())
    }

    /// Update a run's last-known phase + round (from a worker `RunPhase` event).
    pub fn set_phase(&self, run_id: &str, phase: &str, round: u64) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET last_phase = ?2, last_round = ?3, updated_ms = ?4
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
            "UPDATE vhc_runs SET epoch = ?2, role = ?3, instance = ?4, updated_ms = ?5 \
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
            "UPDATE vhc_runs SET envelope_schema_major = ?2, module_abi_major = ?3, \
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
                "SELECT run_id_hash FROM vhc_runs WHERE run_id = ?1",
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
            "UPDATE vhc_runs SET run_id_hash = ?2, updated_ms = ?3 WHERE run_id = ?1",
            params![run_id, run_id_hash.as_slice(), now_ms()],
        )?;
        Ok(())
    }

    /// Persist the immutable admitted tuple (architecture §6.3) for a run's join intent
    /// (canonical-CBOR bytes). Idempotent per `run_id`.
    pub fn set_admitted_tuple(&self, run_id: &str, tuple: &[u8]) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET admitted_tuple = ?2, updated_ms = ?3 WHERE run_id = ?1",
            params![run_id, tuple, now_ms()],
        )?;
        Ok(())
    }

    // -- run-instance lifecycle (the observed axis + the crash-window release protocol) ---------

    /// Begin a terminal release: persist the marker that worker teardown was OBSERVED for this
    /// instance and that `target` is the terminal state to commit once the ledger releases. The
    /// marker is what makes the release crash-repairable: a node crash between observing teardown
    /// and committing the terminal is finished by [`Self::repair_pending_releases`] on startup.
    pub fn begin_release(
        &self,
        run_id: &str,
        target: RunState,
        reason: Option<&str>,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET pending_run_state = ?2, terminal_reason = ?3, updated_ms = ?4 \
             WHERE run_id = ?1",
            params![run_id, target.as_str(), reason, now_ms()],
        )?;
        Ok(())
    }

    /// Commit a begun release: the ledger reservation is surrendered, so the terminal state the
    /// marker recorded becomes the observed state and the marker clears. Idempotent (a row with
    /// no marker is untouched).
    pub fn commit_release(&self, run_id: &str) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET run_state = pending_run_state, pending_run_state = NULL, \
             updated_ms = ?2 WHERE run_id = ?1 AND pending_run_state IS NOT NULL",
            params![run_id, now_ms()],
        )?;
        Ok(())
    }

    /// The startup crash-window repair: finish every release whose marker was persisted but whose
    /// terminal commit never landed (the node crashed between observing worker teardown and the
    /// final write). Process absence makes teardown definitional on a fresh start, so the
    /// recorded target simply commits. Returns the number of repaired rows.
    pub fn repair_pending_releases(&self) -> Result<usize, StoreError> {
        let n = self.lock().execute(
            "UPDATE vhc_runs SET run_state = pending_run_state, pending_run_state = NULL, \
             updated_ms = ?1 WHERE pending_run_state IS NOT NULL",
            params![now_ms()],
        )?;
        Ok(n)
    }

    /// Mark a run's join transaction in flight: observed state `starting`, any release marker
    /// cleared. The retry schedule survives — a `Starting` instance has proven nothing yet, and
    /// the readiness promotion ([`Self::mark_running`], driven by the worker's session event)
    /// is what clears it. The live-instance map guards against a competing reconvergence in the
    /// window, not this row.
    pub fn mark_starting(&self, run_id: &str) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET run_state = 'starting', pending_run_state = NULL, \
             updated_ms = ?2 WHERE run_id = ?1",
            params![run_id, now_ms()],
        )?;
        Ok(())
    }

    /// Mark a run's instance live: observed state `running`, `running_since` stamped, any release
    /// marker cleared. Driven ONLY by an observed worker readiness event (`RunPhase "running"`) —
    /// never by a dispatched command; a `JoinRun` written to stdio proves nothing. The retry
    /// budget is NOT cleared here — that is the uptime reset's job
    /// ([`Self::reset_recovered_retries`]), so a crash-looping instance cannot launder its budget
    /// by merely restarting.
    pub fn mark_running(&self, run_id: &str) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET run_state = 'running', pending_run_state = NULL, \
             next_retry_ms = NULL, running_since_ms = ?2, updated_ms = ?2 WHERE run_id = ?1",
            params![run_id, now_ms()],
        )?;
        Ok(())
    }

    /// Set / clear the storage gate (M8): a `HostStorageExhausted` terminal sets it; a passed
    /// free-space check (or an explicit owner join) clears it. Idempotent.
    pub fn set_storage_gated(&self, run_id: &str, gated: bool) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET storage_gated = ?2, updated_ms = ?3 WHERE run_id = ?1",
            params![run_id, i64::from(gated), now_ms()],
        )?;
        Ok(())
    }

    /// Reschedule the next reconvergence check WITHOUT consuming budget — the storage-gated
    /// deferral: the disk being full is a capacity condition to wait out, so deferring the
    /// redispatch must not count as a failed attempt ([`Self::bump_retry`] does).
    pub fn defer_retry(&self, run_id: &str, next_retry_ms: i64) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET next_retry_ms = ?2, updated_ms = ?3 WHERE run_id = ?1",
            params![run_id, next_retry_ms, now_ms()],
        )?;
        Ok(())
    }

    /// Record one consumed reconvergence attempt and when the next is due.
    pub fn bump_retry(&self, run_id: &str, next_retry_ms: i64) -> Result<u32, StoreError> {
        let conn = self.lock();
        let count: i64 = conn.query_row(
            "UPDATE vhc_runs SET retry_count = retry_count + 1, next_retry_ms = ?2, \
             updated_ms = ?3 WHERE run_id = ?1 RETURNING retry_count",
            params![run_id, next_retry_ms, now_ms()],
            |row| row.get(0),
        )?;
        Ok(count as u32)
    }

    /// Clear the retry bookkeeping (owner pause/leave took manual control, or the uptime reset).
    pub fn clear_retry(&self, run_id: &str) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE vhc_runs SET retry_count = 0, next_retry_ms = NULL, updated_ms = ?2 \
             WHERE run_id = ?1",
            params![run_id, now_ms()],
        )?;
        Ok(())
    }

    /// The recoverable intents whose backoff has elapsed — the reconciliation tick's work list.
    pub fn runs_awaiting_retry(&self, now_ms: i64) -> Result<Vec<PersistedRun>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT {RUN_COLUMNS} FROM vhc_runs WHERE desired_state = 'joined' \
             AND run_state = 'failed_retryable' AND next_retry_ms IS NOT NULL \
             AND next_retry_ms <= ?1 ORDER BY run_id"
        ))?;
        let rows = stmt.query_map(params![now_ms], row_to_run)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r??);
        }
        Ok(out)
    }

    /// The uptime-based retry reset: an instance that stayed `running` beyond `min_uptime_ms`
    /// has genuinely recovered, so its consumed retry budget resets to zero. Returns the number
    /// of reset rows. (The coarse stability signal — the node never inspects rounds.)
    pub fn reset_recovered_retries(
        &self,
        now_ms: i64,
        min_uptime_ms: i64,
    ) -> Result<usize, StoreError> {
        let n = self.lock().execute(
            "UPDATE vhc_runs SET retry_count = 0, next_retry_ms = NULL, updated_ms = ?1 \
             WHERE run_state = 'running' AND retry_count > 0 AND running_since_ms IS NOT NULL \
             AND running_since_ms + ?2 <= ?1",
            params![now_ms, min_uptime_ms],
        )?;
        Ok(n)
    }

    /// The current value of a node-owned monotonic counter (`device_profile_rev` /
    /// `owner_policy_rev`); `0` when unseeded.
    pub fn counter(&self, name: &str) -> Result<u64, StoreError> {
        let conn = self.lock();
        let v: Option<i64> = conn
            .query_row(
                "SELECT value FROM vhc_counters WHERE name = ?1",
                params![name],
                |row| row.get(0),
            )
            .optional()?;
        Ok(v.unwrap_or(0) as u64)
    }

    /// Bump a node-owned monotonic counter by one and return the new value (atomic). Used for the
    /// device-profile revision (assessed device profile changed) and the owner-policy revision
    /// (owner policy mutated) the admitted tuple stamps (architecture §6.3).
    pub fn bump_counter(&self, name: &str) -> Result<u64, StoreError> {
        let conn = self.lock();
        conn.execute(
            "INSERT INTO vhc_counters (name, value) VALUES (?1, 1)
             ON CONFLICT(name) DO UPDATE SET value = value + 1",
            params![name],
        )?;
        let value: i64 = conn.query_row(
            "SELECT value FROM vhc_counters WHERE name = ?1",
            params![name],
            |row| row.get(0),
        )?;
        Ok(value as u64)
    }

    /// Fetch one run row, decoded (`None` if unknown).
    pub fn get_run(&self, run_id: &str) -> Result<Option<PersistedRun>, StoreError> {
        let conn = self.lock();
        conn.query_row(
            &format!("SELECT {RUN_COLUMNS} FROM vhc_runs WHERE run_id = ?1"),
            params![run_id],
            row_to_run,
        )
        .optional()?
        .transpose()
    }

    /// All run rows in `run_id` order.
    pub fn list_runs(&self) -> Result<Vec<PersistedRun>, StoreError> {
        self.query_runs(&format!(
            "SELECT {RUN_COLUMNS} FROM vhc_runs ORDER BY run_id"
        ))
    }

    /// The runs the restart reconciliation pass re-issues `JoinRun` for: a standing `joined`
    /// intent whose observed lifecycle is not TERMINAL. A `completed` instance is never rejoined
    /// (the module ended the run); `failed_terminal` requires owner action; a `paused` intent is
    /// not reconverged until the owner resumes. An observed `left`/`failed_retryable` with a
    /// standing joined intent (shutdown drain / recoverable fault) reconverges.
    pub fn active_intents(&self) -> Result<Vec<PersistedRun>, StoreError> {
        self.query_runs(&format!(
            "SELECT {RUN_COLUMNS} FROM vhc_runs WHERE desired_state = 'joined' \
             AND run_state NOT IN ('completed', 'failed_terminal') ORDER BY run_id"
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
    pub fn get_contribution(&self, run_id: &str) -> Result<VhcContribution, StoreError> {
        let conn = self.lock();
        let c = conn
            .query_row(
                "SELECT rounds, tokens, bytes_up, bytes_down, witness_count, checkpoint_credits
                 FROM vhc_contrib WHERE run_id = ?1",
                params![run_id],
                |row| {
                    Ok(VhcContribution {
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
            "INSERT INTO vhc_contrib
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
    /// (ADR-007). The event body is JSON (`VhcEvent`), keyed by `kind` for cheap filtering.
    pub fn append_event(&self, event: &VhcEvent) -> Result<(), StoreError> {
        let body = serde_json::to_vec(event)?;
        let run_id = event.run_id().to_string();
        let conn = self.lock();
        conn.execute(
            "INSERT INTO vhc_events (run_id, ts_ms, kind, body) VALUES (?1, ?2, ?3, ?4)",
            params![run_id, now_ms(), event.kind(), body],
        )?;
        conn.execute(
            "DELETE FROM vhc_events WHERE run_id = ?1 AND seq NOT IN
                (SELECT seq FROM vhc_events WHERE run_id = ?1 ORDER BY seq DESC LIMIT ?2)",
            params![run_id, EVENT_WINDOW as i64],
        )?;
        Ok(())
    }

    /// The most recent events for a run in chronological order (oldest → newest), capped at `limit`.
    pub fn recent_events(&self, run_id: &str, limit: usize) -> Result<Vec<VhcEvent>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn
            .prepare("SELECT body FROM vhc_events WHERE run_id = ?1 ORDER BY seq DESC LIMIT ?2")?;
        let rows = stmt.query_map(params![run_id, limit as i64], |row| {
            row.get::<_, Vec<u8>>(0)
        })?;
        let mut out = Vec::new();
        for r in rows {
            let bytes = r?;
            out.push(serde_json::from_slice::<VhcEvent>(&bytes)?);
        }
        out.reverse();
        Ok(out)
    }

    // The D5 sunset-audit query (`v1_sunset_audit` — active-intent runs still needing the
    // retired v1 driver, judged from the D0 observability columns) was the E3 transition gate's
    // "no live v1 runs" proof and was REMOVED post-sunset: with the typed v1 refusal in place no
    // new v1 run can be admitted, so its only remaining function was detecting pre-sunset rows in
    // an existing vhc.db — a back-compat accommodation the ratified posture drops. The D0
    // observability COLUMNS (envelope_schema_major / module_abi_major / selected_driver /
    // module_hash) stay: they are the per-run provenance/diagnostics record, not compat.

    /// Mint the next **role-instance incarnation id** — never-reused, node-durable, monotonic
    /// (ABI §8.1; decisions D1). The first minted id is `1` (`0` is the pre-multi-instance
    /// default in existing rows). Atomic: the increment and the read are one SQL statement,
    /// guarded against the bounded-domain ceiling (`i64::MAX`, the shared ordinal domain) —
    /// exhaustion is a typed error, never a silent wrap.
    pub fn mint_incarnation(&self) -> Result<u64, StoreError> {
        let conn = self.lock();
        let value: i64 = conn
            .query_row(
                "UPDATE vhc_counters SET value = value + 1 \
             WHERE name = 'incarnation' AND value < ?1 RETURNING value",
                params![i64::MAX],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => StoreError::CounterExhausted("incarnation"),
                other => StoreError::from(other),
            })?;
        Ok(value as u64)
    }

    /// The node's persisted **highest verified leadership term** for a `(run, role)` seat slot,
    /// with the claimant it bound ([SEAT-1] v2 — a separate order relation from the execution
    /// counter). `None` until a verified grant was ever observed.
    pub fn seat_term(
        &self,
        run_label: &str,
        role: &str,
    ) -> Result<Option<(u64, [u8; 32])>, StoreError> {
        let conn = self.lock();
        let row = conn
            .query_row(
                "SELECT term, claimant FROM vhc_seat_terms WHERE run_id = ?1 AND role = ?2",
                params![run_label, role],
                |row| {
                    let term: i64 = row.get(0)?;
                    let claimant: Vec<u8> = row.get(1)?;
                    Ok((term, claimant))
                },
            )
            .optional()?;
        Ok(row.and_then(|(term, claimant)| {
            let claimant: [u8; 32] = claimant.try_into().ok()?;
            Some((term as u64, claimant))
        }))
    }

    /// Observe a **verified** seat grant's leadership term into the persisted floor — monotonic
    /// MAX per slot (a lower/equal term is a no-op that never regresses the binding). TRUST GATE
    /// (caller-enforced): only terms from grants this node authored and won, or stored leases
    /// that passed the full peer-side authorization ([`crate::seat::authorize_incumbent`]) —
    /// never naked registry metadata.
    pub fn observe_seat_term(
        &self,
        run_label: &str,
        role: &str,
        term: u64,
        claimant: &[u8; 32],
    ) -> Result<(), StoreError> {
        let term = i64::try_from(term).map_err(|_| StoreError::OrdinalOutOfDomain(term))?;
        self.lock().execute(
            "INSERT INTO vhc_seat_terms (run_id, role, term, claimant) VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(run_id, role) DO UPDATE \
             SET term = excluded.term, claimant = excluded.claimant \
             WHERE excluded.term > vhc_seat_terms.term",
            params![run_label, role, term, claimant.as_slice()],
        )?;
        Ok(())
    }

    /// Mint the next incarnation id, guaranteed **strictly above `floor`** — the live-upgrade /
    /// floor-repair minting rule (ABI §8.1/§10.3): the successor incarnation must supersede a
    /// value observed out-of-band (a running instance across a switch; a verified own-base
    /// roster floor). Atomic: the counter is raised to `max(counter, floor) + 1` and read in
    /// one statement, so the never-reused guarantee holds for every later
    /// [`mint_incarnation`](Self::mint_incarnation) too.
    ///
    /// `floor` MUST already be inside the bounded ordinal domain (`<= i64::MAX`) — an
    /// out-of-domain floor is a typed refusal, NEVER truncated into the counter (the audited
    /// `floor as i64` cast wrapped negative and silently voided the "strictly above" contract);
    /// a floor at the ceiling exhausts the counter typed.
    pub fn mint_incarnation_above(&self, floor: u64) -> Result<u64, StoreError> {
        let floor = i64::try_from(floor).map_err(|_| StoreError::OrdinalOutOfDomain(floor))?;
        let conn = self.lock();
        let value: i64 = conn
            .query_row(
                "UPDATE vhc_counters SET value = MAX(value, ?1) + 1 \
             WHERE name = 'incarnation' AND MAX(value, ?1) < ?2 RETURNING value",
                params![floor, i64::MAX],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => StoreError::CounterExhausted("incarnation"),
                other => StoreError::from(other),
            })?;
        Ok(value as u64)
    }

    /// Persist one committed transition-chain record for a run (the M6 node-local mirror):
    /// canonical-CBOR `UpgradeRecord` bytes keyed by the epoch the record establishes.
    /// Idempotent per `(run_id, epoch)` — a re-consumed record overwrites with identical bytes.
    pub fn put_upgrade_record(
        &self,
        run_id: &str,
        epoch: u64,
        record: &[u8],
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO vhc_upgrade_records (run_id, epoch, record, created_ms) \
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(run_id, epoch) DO UPDATE SET record = excluded.record",
            params![run_id, epoch as i64, record, now_ms()],
        )?;
        Ok(())
    }

    /// The run's persisted transition-chain records, ordered by epoch (the chain-rebuild input
    /// for the next switch's fail-closed validation).
    pub fn upgrade_records(&self, run_id: &str) -> Result<Vec<(u64, Vec<u8>)>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT epoch, record FROM vhc_upgrade_records WHERE run_id = ?1 ORDER BY epoch",
        )?;
        let rows = stmt
            .query_map(params![run_id], |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get::<_, Vec<u8>>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Set the owner's preemption priority for a run (decisions D6 point 4: node-side owner
    /// state — the envelope can never set or influence its own priority). Idempotent upsert.
    pub fn set_run_priority(&self, run_id: &str, priority: u8) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO vhc_owner_priority (run_id, priority) VALUES (?1, ?2)
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
                "SELECT priority FROM vhc_owner_priority WHERE run_id = ?1",
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
            "SELECT COUNT(*) FROM vhc_events WHERE run_id = ?1",
            params![run_id],
            |row| row.get(0),
        )?;
        Ok(n as usize)
    }
}

/// Decode a `vhc_runs` row into a [`PersistedRun`]. The JSON columns decode outside the rusqlite
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
    let admitted_tuple: Option<Vec<u8>> = row.get(16)?;
    let run_state: String = row.get(17)?;
    let pending_run_state: Option<String> = row.get(18)?;
    let retry_count: i64 = row.get(19)?;
    let next_retry_ms: Option<i64> = row.get(20)?;
    let running_since_ms: Option<i64> = row.get(21)?;
    let terminal_reason: Option<String> = row.get(22)?;
    let storage_gated: i64 = row.get(23)?;
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
            admitted_tuple,
            run_state: RunState::from_str(&run_state),
            pending_run_state: pending_run_state.as_deref().map(RunState::from_str),
            retry_count: retry_count as u32,
            next_retry_ms,
            running_since_ms,
            terminal_reason,
            storage_gated: storage_gated != 0,
        })
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::VhcPolicyMode;

    fn v1_policy() -> VhcPolicy {
        VhcPolicy {
            mode: VhcPolicyMode::Idle,
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
        let elig_json = serde_json::to_string(&VhcEligibility::default()).unwrap();
        conn.execute(
            "INSERT INTO vhc_runs (run_id, coordinator, policy_json, desired_state, \
             eligibility_json) VALUES ('legacy-run', 'ws://c', ?1, 'joined', ?2)",
            params![policy_json, elig_json],
        )
        .unwrap();

        // Now run the FULL ladder (M1 + M2). rusqlite_migration applies only the missing M2.
        migrations().to_latest(&mut conn).unwrap();

        let store = VhcStore {
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
        let store = VhcStore::open_in_memory().unwrap();
        store
            .put_join_intent(
                "run-A",
                "ws://c",
                &v1_policy(),
                None,
                &VhcEligibility::default(),
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
        let path = dir.path().join("vhc.db");

        let store = VhcStore::open(&path).unwrap();
        assert_eq!(store.mint_incarnation().unwrap(), 1);
        assert_eq!(store.mint_incarnation().unwrap(), 2);
        drop(store);
        // A re-open (node restart) continues the durable counter — never reuses an id.
        let store = VhcStore::open(&path).unwrap();
        assert_eq!(store.mint_incarnation().unwrap(), 3);

        assert_eq!(store.run_priority("run-X").unwrap(), 100, "default");
        store.set_run_priority("run-X", 7).unwrap();
        assert_eq!(store.run_priority("run-X").unwrap(), 7);
    }

    /// M7: the ordinal domain is bounded and every edge is TYPED — a poisoned floor can never
    /// wrap, truncate, or silently void the "strictly above" contract (the audited `floor as
    /// i64` cast defect class).
    #[test]
    fn m7_bounded_ordinal_arithmetic_refuses_poisoned_floors_typed() {
        let store = VhcStore::open_in_memory().unwrap();

        // The floor-taking mint is strictly above any in-domain floor, and the plain counter
        // continues above it (never-reused holds across both mint paths).
        assert_eq!(store.mint_incarnation_above(5).unwrap(), 6);
        assert_eq!(store.mint_incarnation().unwrap(), 7);

        // An out-of-domain floor (> i64::MAX) is a typed refusal, never a wrap: the counter is
        // untouched and keeps minting normally.
        let err = store.mint_incarnation_above(u64::MAX).unwrap_err();
        assert!(matches!(err, StoreError::OrdinalOutOfDomain(_)), "{err}");
        let err = store
            .mint_incarnation_above(i64::MAX as u64 + 1)
            .unwrap_err();
        assert!(matches!(err, StoreError::OrdinalOutOfDomain(_)), "{err}");
        assert_eq!(store.mint_incarnation().unwrap(), 8);

        // A floor AT the ceiling exhausts the counter typed ("+1" would leave the domain);
        // the counter is untouched by the refused attempt.
        let err = store.mint_incarnation_above(i64::MAX as u64).unwrap_err();
        assert!(matches!(err, StoreError::CounterExhausted(_)), "{err}");
        assert_eq!(store.mint_incarnation().unwrap(), 9);
    }

    /// M7: the persisted leadership-term floor is monotonic per `(run, role)` slot — a lower or
    /// equal term never regresses the binding, and slots are independent.
    #[test]
    fn m7_seat_term_floor_is_monotonic_and_slot_scoped() {
        let store = VhcStore::open_in_memory().unwrap();
        let a = [0xAA_u8; 32];
        let b = [0xBB_u8; 32];

        assert_eq!(store.seat_term("run-T", "coordinator").unwrap(), None);
        store
            .observe_seat_term("run-T", "coordinator", 10, &a)
            .unwrap();
        assert_eq!(
            store.seat_term("run-T", "coordinator").unwrap(),
            Some((10, a))
        );

        // A higher verified term advances the floor AND the bound claimant (sparse jump).
        store
            .observe_seat_term("run-T", "coordinator", 25, &b)
            .unwrap();
        assert_eq!(
            store.seat_term("run-T", "coordinator").unwrap(),
            Some((25, b))
        );

        // A lower or equal term is a no-op — the floor never regresses, the binding holds.
        store
            .observe_seat_term("run-T", "coordinator", 10, &a)
            .unwrap();
        store
            .observe_seat_term("run-T", "coordinator", 25, &a)
            .unwrap();
        assert_eq!(
            store.seat_term("run-T", "coordinator").unwrap(),
            Some((25, b))
        );

        // An out-of-domain term is a typed refusal, never a truncated row.
        let err = store
            .observe_seat_term("run-T", "coordinator", u64::MAX, &a)
            .unwrap_err();
        assert!(matches!(err, StoreError::OrdinalOutOfDomain(_)), "{err}");

        // Slots are independent per (run, role).
        store
            .observe_seat_term("run-U", "coordinator", 3, &a)
            .unwrap();
        assert_eq!(
            store.seat_term("run-T", "coordinator").unwrap(),
            Some((25, b))
        );
        assert_eq!(
            store.seat_term("run-U", "coordinator").unwrap(),
            Some((3, a))
        );
    }

    /// M4: the admitted-tuple column persists beside the join intent, and the two node-owned
    /// revision counters are durable + monotonic (architecture §6.3).
    #[test]
    fn m4_admitted_tuple_and_revision_counters() {
        let store = VhcStore::open_in_memory().unwrap();
        store
            .put_join_intent(
                "run-T",
                "ws://c",
                &v1_policy(),
                None,
                &VhcEligibility::default(),
            )
            .unwrap();
        // Absent until stamped.
        assert_eq!(
            store.get_run("run-T").unwrap().unwrap().admitted_tuple,
            None
        );
        let tuple = vec![0xA1, 0xB2, 0xC3];
        store.set_admitted_tuple("run-T", &tuple).unwrap();
        assert_eq!(
            store.get_run("run-T").unwrap().unwrap().admitted_tuple,
            Some(tuple)
        );

        // The node-owned counters seed at 0 and bump monotonically.
        assert_eq!(store.counter("device_profile_rev").unwrap(), 0);
        assert_eq!(store.counter("owner_policy_rev").unwrap(), 0);
        assert_eq!(store.bump_counter("device_profile_rev").unwrap(), 1);
        assert_eq!(store.bump_counter("device_profile_rev").unwrap(), 2);
        assert_eq!(store.counter("device_profile_rev").unwrap(), 2);
        assert_eq!(store.counter("owner_policy_rev").unwrap(), 0);
    }

    /// M5: the two-axis lifecycle columns default a fresh join to an in-flight `starting`
    /// observation (readiness is OBSERVED via `mark_running`, never assumed from the dispatched
    /// command), the release protocol (begin → commit) transitions the observed axis exactly
    /// once, and the startup repair finishes a marker whose commit never landed (the crash
    /// window).
    #[test]
    fn m5_release_protocol_begin_commit_and_crash_repair() {
        let store = VhcStore::open_in_memory().unwrap();
        store
            .put_join_intent(
                "run-L",
                "ws://c",
                &v1_policy(),
                None,
                &VhcEligibility::default(),
            )
            .unwrap();
        let run = store.get_run("run-L").unwrap().unwrap();
        assert_eq!(run.run_state, RunState::Starting);
        assert_eq!(run.pending_run_state, None);

        // The readiness promotion: only the observed session event makes the row `running`.
        store.mark_running("run-L").unwrap();
        let run = store.get_run("run-L").unwrap().unwrap();
        assert_eq!(run.run_state, RunState::Running);
        assert!(run.running_since_ms.is_some());

        // begin → the marker + reason are durable while the observed state is still `running`.
        store
            .begin_release("run-L", RunState::Completed, Some("outcome 0"))
            .unwrap();
        let run = store.get_run("run-L").unwrap().unwrap();
        assert_eq!(run.run_state, RunState::Running);
        assert_eq!(run.pending_run_state, Some(RunState::Completed));
        assert_eq!(run.terminal_reason.as_deref(), Some("outcome 0"));

        // commit → the marker's target becomes the observed state; the marker clears.
        store.commit_release("run-L").unwrap();
        let run = store.get_run("run-L").unwrap().unwrap();
        assert_eq!(run.run_state, RunState::Completed);
        assert_eq!(run.pending_run_state, None);
        // A second commit with no marker is a no-op (idempotent).
        store.commit_release("run-L").unwrap();
        assert_eq!(
            store.get_run("run-L").unwrap().unwrap().run_state,
            RunState::Completed
        );

        // Crash window: a begun-but-uncommitted release is finished by the startup repair.
        store
            .put_join_intent(
                "run-M",
                "ws://c",
                &v1_policy(),
                None,
                &VhcEligibility::default(),
            )
            .unwrap();
        store
            .begin_release("run-M", RunState::FailedRetryable, Some("stream closed"))
            .unwrap();
        assert_eq!(store.repair_pending_releases().unwrap(), 1);
        let run = store.get_run("run-M").unwrap().unwrap();
        assert_eq!(run.run_state, RunState::FailedRetryable);
        assert_eq!(run.pending_run_state, None);
    }

    /// M5: a `completed` / `failed_terminal` observation drops out of the reconvergence set even
    /// under a standing joined intent (a module run end NEVER restarts); `failed_retryable` and
    /// an observed `left` stay in; a `paused` intent leaves the set entirely.
    #[test]
    fn m5_active_intents_respect_the_observed_axis() {
        let store = VhcStore::open_in_memory().unwrap();
        for run in ["r-run", "r-done", "r-fatal", "r-retry", "r-left", "r-pause"] {
            store
                .put_join_intent(
                    run,
                    "ws://c",
                    &v1_policy(),
                    None,
                    &VhcEligibility::default(),
                )
                .unwrap();
        }
        let terminal = |run: &str, state: RunState| {
            store.begin_release(run, state, None).unwrap();
            store.commit_release(run).unwrap();
        };
        terminal("r-done", RunState::Completed);
        terminal("r-fatal", RunState::FailedTerminal);
        terminal("r-retry", RunState::FailedRetryable);
        terminal("r-left", RunState::Left);
        store
            .set_desired_state("r-pause", DesiredState::Paused)
            .unwrap();

        let intents: Vec<String> = store
            .active_intents()
            .unwrap()
            .into_iter()
            .map(|r| r.run_id)
            .collect();
        assert_eq!(intents, ["r-left", "r-retry", "r-run"]);
    }

    /// M5: the retry bookkeeping — bump returns the consumed count, the due query honors the
    /// schedule, and the uptime reset clears only a stably-running row's budget.
    #[test]
    fn m5_retry_budget_bookkeeping_and_uptime_reset() {
        let store = VhcStore::open_in_memory().unwrap();
        store
            .put_join_intent(
                "run-R",
                "ws://c",
                &v1_policy(),
                None,
                &VhcEligibility::default(),
            )
            .unwrap();
        store
            .begin_release("run-R", RunState::FailedRetryable, Some("transport"))
            .unwrap();
        store.commit_release("run-R").unwrap();

        let due_at = now_ms() + 50;
        assert_eq!(store.bump_retry("run-R", due_at).unwrap(), 1);
        assert_eq!(store.bump_retry("run-R", due_at).unwrap(), 2);
        // Not due yet at a time before the schedule; due at/after it.
        assert!(store.runs_awaiting_retry(due_at - 10).unwrap().is_empty());
        let due = store.runs_awaiting_retry(due_at).unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].retry_count, 2);

        // Reconverged: running again, but the budget survives the mere restart...
        store.mark_running("run-R").unwrap();
        assert_eq!(store.get_run("run-R").unwrap().unwrap().retry_count, 2);
        // ...until the instance has been stably up past the minimum uptime.
        assert_eq!(store.reset_recovered_retries(now_ms(), 60_000).unwrap(), 0);
        assert_eq!(
            store
                .reset_recovered_retries(now_ms() + 60_001, 60_000)
                .unwrap(),
            1
        );
        assert_eq!(store.get_run("run-R").unwrap().unwrap().retry_count, 0);
    }

    /// M8 storage-gate bookkeeping: the gate round-trips durably, `defer_retry` reschedules
    /// WITHOUT consuming budget (the storage wait is not a failed attempt), and an explicit
    /// owner join clears the gate along with the rest of the lifecycle bookkeeping.
    #[test]
    fn m8_storage_gate_and_budget_free_deferral() {
        let store = VhcStore::open_in_memory().unwrap();
        store
            .put_join_intent(
                "run-S",
                "ws://c",
                &v1_policy(),
                None,
                &VhcEligibility::default(),
            )
            .unwrap();
        assert!(!store.get_run("run-S").unwrap().unwrap().storage_gated);

        store
            .begin_release("run-S", RunState::FailedRetryable, Some("storage"))
            .unwrap();
        store.commit_release("run-S").unwrap();
        store.set_storage_gated("run-S", true).unwrap();
        let due_at = now_ms() + 50;
        store.defer_retry("run-S", due_at).unwrap();

        let row = store.get_run("run-S").unwrap().unwrap();
        assert!(row.storage_gated, "the gate is durable");
        assert_eq!(row.retry_count, 0, "a deferral consumes no budget");
        assert_eq!(row.next_retry_ms, Some(due_at), "the next check is due");
        // The gated row still surfaces in the reconcile work list (the tick applies the gate).
        assert_eq!(store.runs_awaiting_retry(due_at).unwrap().len(), 1);

        store.set_storage_gated("run-S", false).unwrap();
        assert!(!store.get_run("run-S").unwrap().unwrap().storage_gated);

        // An explicit owner join re-arms the lifecycle: the gate clears with it.
        store.set_storage_gated("run-S", true).unwrap();
        store
            .put_join_intent(
                "run-S",
                "ws://c",
                &v1_policy(),
                None,
                &VhcEligibility::default(),
            )
            .unwrap();
        assert!(!store.get_run("run-S").unwrap().unwrap().storage_gated);
    }

    /// The effective app-facing projection of the two axes (spec §6.1 six-state view): terminal
    /// observations win; a paused intent masks recoverable states; intent `left` reads left.
    #[test]
    fn effective_state_projects_intent_over_observation() {
        use super::effective_state as eff;
        assert_eq!(eff(DesiredState::Joined, RunState::Running), "running");
        assert_eq!(eff(DesiredState::Joined, RunState::Completed), "completed");
        assert_eq!(
            eff(DesiredState::Joined, RunState::FailedRetryable),
            "failed_retryable"
        );
        assert_eq!(
            eff(DesiredState::Joined, RunState::FailedTerminal),
            "failed_terminal"
        );
        assert_eq!(eff(DesiredState::Paused, RunState::Running), "paused");
        assert_eq!(
            eff(DesiredState::Paused, RunState::FailedRetryable),
            "paused"
        );
        // Terminal observations are never masked by pause/leave intent.
        assert_eq!(eff(DesiredState::Paused, RunState::Completed), "completed");
        assert_eq!(
            eff(DesiredState::Left, RunState::FailedTerminal),
            "failed_terminal"
        );
        assert_eq!(eff(DesiredState::Left, RunState::Running), "left");
        assert_eq!(eff(DesiredState::Left, RunState::Left), "left");
    }

    /// The v2 execution-identity + observability writers populate the D0 columns (decisions D5;
    /// the columns outlive the retired `v1_sunset_audit` transition query as the per-run
    /// provenance/diagnostics record).
    #[test]
    fn identity_and_observability_writers() {
        let store = VhcStore::open_in_memory().unwrap();
        store
            .put_join_intent(
                "run-B",
                "iroh://c",
                &v1_policy(),
                None,
                &VhcEligibility::default(),
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
