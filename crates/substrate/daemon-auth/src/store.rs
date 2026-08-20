// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The SQLite-backed identity store: users, password (Argon2id PHC) credentials, opaque
//! server-side session tokens, role assignments, and the reserved tables for SCRAM material, API
//! keys, and (future) per-resource grants.
//!
//! Concurrency + migration follow the workspace convention (see `daemon-store`/`daemon-mnemosyne`):
//! a single `Mutex<Connection>` in WAL mode, schema owned by a `PRAGMA user_version` ladder with
//! pragmas applied in [`AuthStore::init`] outside the migration transaction. Tokens are never stored
//! raw — only their SHA-256 hash — mirroring OWASP session-management guidance (opaque, server-side,
//! revocable). SCRAM/API-key/grant tables exist now but are populated by later phases.

use crate::capability::{Principal, Role};
use crate::error::{Error, Result};
use crate::scram::{self, ScramMaterial, SCRAM_SHA_256};
use rusqlite::{params, Connection, OptionalExtension};
use rusqlite_migration::{Migrations, M};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default lifetime of a minted session token (7 days). Reconnects refresh against this.
pub const DEFAULT_SESSION_TTL_SECS: i64 = 7 * 24 * 60 * 60;

/// The full identity schema. `M1` (idempotent `CREATE TABLE IF NOT EXISTS`); future schema changes
/// append `M::up("ALTER …")`. Pragmas live in [`AuthStore::init`], not here.
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    username   TEXT NOT NULL UNIQUE,
    disabled   INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS password_credentials (
    user_id    TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    phc_hash   TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

-- SCRAM-SHA-256 stored material (RFC 5802): salt + iterations + StoredKey/ServerKey. Populated by
-- the rsasl authenticator phase; PLAIN/Argon2 login above does not need it.
CREATE TABLE IF NOT EXISTS scram_credentials (
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mechanism  TEXT NOT NULL,
    salt       BLOB NOT NULL,
    iterations INTEGER NOT NULL,
    stored_key BLOB NOT NULL,
    server_key BLOB NOT NULL,
    PRIMARY KEY (user_id, mechanism)
);

-- Machine/API tokens: only a keyed hash is stored (never the raw token).
CREATE TABLE IF NOT EXISTS api_keys (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash BLOB NOT NULL UNIQUE,
    scopes     TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    expires_at INTEGER,
    revoked_at INTEGER
);

-- Opaque, server-side session tokens (the reconnect fast-path). Stored as SHA-256(token).
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash  BLOB PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL,
    auth_method TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS auth_sessions_user ON auth_sessions (user_id);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role    TEXT NOT NULL,
    PRIMARY KEY (user_id, role)
);

-- Reserved for future fine-grained per-resource ACL (the option-B seam): grant one capability over
-- one resource (session/profile/agent) to one user. Created now so enabling sharing later is a
-- pure-additive change with no migration churn; NOT enforced yet.
CREATE TABLE IF NOT EXISTS resource_grants (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    resource_kind TEXT NOT NULL,
    resource_id   TEXT NOT NULL,
    capability    TEXT NOT NULL,
    granted_by    TEXT,
    created_at    INTEGER NOT NULL
);
"#;

static MIGRATIONS: LazyLock<Migrations<'static>> = LazyLock::new(|| {
    Migrations::new(vec![
        M::up(SCHEMA),
        // M2 (Auth 4 EXTERNAL): map a verified mTLS client-certificate fingerprint to a user. The
        // SASL EXTERNAL read path (`external_identity`) resolves a presented cert to its owner;
        // `set_external_identity` is the store-level writer (the admin enrollment op is deferred).
        // `fingerprint` is the primary key — one certificate maps to at most one user. Never edit a
        // released migration; this only appends a table.
        M::up(
            "CREATE TABLE IF NOT EXISTS external_identities (\
                fingerprint TEXT PRIMARY KEY, \
                user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, \
                created_at  INTEGER NOT NULL\
             );\n\
             CREATE INDEX IF NOT EXISTS external_identities_user ON external_identities (user_id);",
        ),
        // M3 (pairing spec §5.4): device metadata on the fingerprint mapping — the display name
        // the device sent during pairing, and the last successful EXTERNAL login. `created_at`
        // already exists (M2). Never edit a released migration; this only adds columns.
        M::up(
            "ALTER TABLE external_identities ADD COLUMN display_name TEXT;\n\
             ALTER TABLE external_identities ADD COLUMN last_seen_at INTEGER;",
        ),
    ])
});

/// A persisted user row (without credential material).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserRecord {
    /// Stable opaque id.
    pub id: String,
    /// Unique username.
    pub username: String,
    /// Whether an admin has disabled the account.
    pub disabled: bool,
    /// Unix seconds at creation.
    pub created_at: i64,
}

/// One `external_identities` row joined with its owning user (the paired-devices admin view).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalIdentityRecord {
    /// SHA-256 of the client leaf certificate DER, lowercase hex.
    pub fingerprint: String,
    /// The mapped user's stable id.
    pub user_id: String,
    /// The mapped user's username.
    pub username: String,
    /// The device display name captured at pairing (M3).
    pub display_name: Option<String>,
    /// Unix seconds at enrollment.
    pub created_at: i64,
    /// Unix seconds of the last successful EXTERNAL login (M3).
    pub last_seen_at: Option<i64>,
    /// Whether the mapped user is disabled (a disabled user's cert never authenticates).
    pub disabled: bool,
}

/// The SQLite-backed identity / credential / session store.
pub struct AuthStore {
    conn: Mutex<Connection>,
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// `nbytes` of OS entropy, hex-encoded (`2 * nbytes` chars). Used by the first-admin bootstrap to
/// mint a strong auto-generated admin password and a random username suffix; kept here (beside
/// [`random_hex`]) so all CSPRNG use funnels through this crate's `getrandom` + hex helpers.
pub fn generate_secret_hex(nbytes: usize) -> Result<String> {
    let mut buf = vec![0u8; nbytes];
    getrandom::getrandom(&mut buf).map_err(|e| Error::Entropy(e.to_string()))?;
    Ok(hex(&buf))
}

/// 32 bytes of OS entropy, hex-encoded — used for user ids and session tokens.
fn random_hex() -> Result<String> {
    let mut buf = [0u8; 32];
    getrandom::getrandom(&mut buf).map_err(|e| Error::Entropy(e.to_string()))?;
    Ok(hex(&buf))
}

fn token_hash(token: &str) -> Vec<u8> {
    Sha256::digest(token.as_bytes()).to_vec()
}

/// Derive + upsert the `SCRAM-SHA-256` row for `user_id` from `password`, on the caller's open
/// connection (so it shares the same write as the Argon2id PHC). Keeps PLAIN (Argon2) and SCRAM
/// coherent for a user: both are derived from the same password whenever it is set.
fn upsert_scram(conn: &Connection, user_id: &str, password: &str) -> Result<()> {
    let m = scram::derive_scram_material(password)?;
    conn.execute(
        "INSERT INTO scram_credentials \
            (user_id, mechanism, salt, iterations, stored_key, server_key) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(user_id, mechanism) DO UPDATE SET \
            salt = excluded.salt, iterations = excluded.iterations, \
            stored_key = excluded.stored_key, server_key = excluded.server_key",
        params![
            user_id,
            SCRAM_SHA_256,
            m.salt,
            m.iterations,
            m.stored_key,
            m.server_key
        ],
    )?;
    Ok(())
}

impl AuthStore {
    /// Open (creating if absent) the identity store at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Error::Sqlite(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
                        Some(e.to_string()),
                    ))
                })?;
            }
        }
        Self::init(Connection::open(path)?)
    }

    /// Open an ephemeral in-memory store (tests).
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(mut conn: Connection) -> Result<Self> {
        // Connection pragmas OUTSIDE the migration transaction (journal_mode cannot change inside
        // one). `foreign_keys=ON` makes the `ON DELETE CASCADE` references effective.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;",
        )?;
        MIGRATIONS.to_latest(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        // A poisoned lock means a prior panic mid-write; recover the guard rather than cascading the
        // panic — the next op either succeeds or returns a normal SQLite error.
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    // --- users ------------------------------------------------------------------------------

    /// Create a user with a password and an initial role set. Fails if the username is taken, or if
    /// the username is [reserved](crate::is_reserved_username) for a synthetic in-process principal
    /// (`system` / `internal`) — a real user must never be able to claim an identity whose
    /// ownership stamp collides with a trusted internal caller.
    pub fn create_user(
        &self,
        username: &str,
        password: &str,
        roles: &[Role],
    ) -> Result<UserRecord> {
        if crate::is_reserved_username(username) {
            return Err(Error::ReservedUsername);
        }
        let id = random_hex()?;
        // Ids are 32 bytes of CSPRNG hex, so a collision with a short reserved word is
        // astronomically impossible; reject anyway (defense-in-depth) so the invariant "no real row
        // carries a reserved id/username" holds unconditionally.
        if crate::is_reserved_username(&id) {
            return Err(Error::ReservedUsername);
        }
        let phc = password_auth::generate_hash(password.as_bytes());
        let created_at = now_secs();
        let conn = self.lock();
        conn.execute(
            "INSERT INTO users (id, username, disabled, created_at) VALUES (?1, ?2, 0, ?3)",
            params![id, username, created_at],
        )?;
        conn.execute(
            "INSERT INTO password_credentials (user_id, phc_hash, updated_at) VALUES (?1, ?2, ?3)",
            params![id, phc, created_at],
        )?;
        // Derive the parallel SCRAM-SHA-256 material from the same password (same write).
        upsert_scram(&conn, &id, password)?;
        for role in roles {
            conn.execute(
                "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?1, ?2)",
                params![id, role.as_str()],
            )?;
        }
        Ok(UserRecord {
            id,
            username: username.to_string(),
            disabled: false,
            created_at,
        })
    }

    /// Create a user with NO password credentials (pairing spec §5.4): the account authenticates
    /// exclusively via a mapped external identity (mTLS client cert + SASL EXTERNAL). No
    /// `password_credentials` / `scram_credentials` rows are written, so a SCRAM/PLAIN attempt
    /// against the username fails closed through the authenticator's decoy path — exactly like a
    /// wrong password, with no probe oracle for "this user is passwordless".
    pub fn create_external_user(&self, username: &str, roles: &[Role]) -> Result<UserRecord> {
        if crate::is_reserved_username(username) {
            return Err(Error::ReservedUsername);
        }
        let id = random_hex()?;
        if crate::is_reserved_username(&id) {
            return Err(Error::ReservedUsername);
        }
        let created_at = now_secs();
        let conn = self.lock();
        conn.execute(
            "INSERT INTO users (id, username, disabled, created_at) VALUES (?1, ?2, 0, ?3)",
            params![id, username, created_at],
        )?;
        for role in roles {
            conn.execute(
                "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?1, ?2)",
                params![id, role.as_str()],
            )?;
        }
        Ok(UserRecord {
            id,
            username: username.to_string(),
            disabled: false,
            created_at,
        })
    }

    /// Pairing enrollment (pairing spec §5.4), one transaction: ensure the device-scoped user
    /// exists (created passwordless via the [`Self::create_external_user`] shape; a previously
    /// revoked device user is re-enabled on re-pair), then upsert the fingerprint mapping with the
    /// device display name. Role is always [`Role::User`] — elevation is an explicit post-pairing
    /// admin act, never part of the ceremony.
    pub fn enroll_paired_device(
        &self,
        username: &str,
        fingerprint: &str,
        display_name: Option<&str>,
    ) -> Result<UserRecord> {
        if crate::is_reserved_username(username) {
            return Err(Error::ReservedUsername);
        }
        let minted_id = random_hex()?;
        if crate::is_reserved_username(&minted_id) {
            return Err(Error::ReservedUsername);
        }
        let now = now_secs();
        let conn = self.lock();
        let tx = conn.unchecked_transaction()?;
        let existing: Option<(String, i64)> = tx
            .query_row(
                "SELECT id, created_at FROM users WHERE username = ?1",
                params![username],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (user_id, created_at) = match existing {
            Some((id, created_at)) => {
                // Re-pairing a previously revoked (disabled, never deleted) device user.
                tx.execute("UPDATE users SET disabled = 0 WHERE id = ?1", params![id])?;
                (id, created_at)
            }
            None => {
                tx.execute(
                    "INSERT INTO users (id, username, disabled, created_at) VALUES (?1, ?2, 0, ?3)",
                    params![minted_id, username, now],
                )?;
                tx.execute(
                    "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?1, ?2)",
                    params![minted_id, Role::User.as_str()],
                )?;
                (minted_id, now)
            }
        };
        tx.execute(
            "INSERT INTO external_identities (fingerprint, user_id, created_at, display_name) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(fingerprint) DO UPDATE SET user_id = excluded.user_id, \
             created_at = excluded.created_at, display_name = excluded.display_name",
            params![fingerprint, user_id, now, display_name],
        )?;
        tx.commit()?;
        Ok(UserRecord {
            id: user_id,
            username: username.to_string(),
            disabled: false,
            created_at,
        })
    }

    /// Look a user up by username.
    pub fn find_user(&self, username: &str) -> Result<Option<UserRecord>> {
        let conn = self.lock();
        conn.query_row(
            "SELECT id, username, disabled, created_at FROM users WHERE username = ?1",
            params![username],
            |r| {
                Ok(UserRecord {
                    id: r.get(0)?,
                    username: r.get(1)?,
                    disabled: r.get::<_, i64>(2)? != 0,
                    created_at: r.get(3)?,
                })
            },
        )
        .optional()
        .map_err(Error::from)
    }

    /// List all users (admin surface).
    pub fn list_users(&self) -> Result<Vec<UserRecord>> {
        let conn = self.lock();
        let mut stmt =
            conn.prepare("SELECT id, username, disabled, created_at FROM users ORDER BY username")?;
        let rows = stmt.query_map([], |r| {
            Ok(UserRecord {
                id: r.get(0)?,
                username: r.get(1)?,
                disabled: r.get::<_, i64>(2)? != 0,
                created_at: r.get(3)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Error::from)
    }

    /// Number of stored users. Mirrors [`Self::session_count`]; lets the first-admin bootstrap gate
    /// on an empty users table (`user_count()? == 0`) without materializing every row.
    pub fn user_count(&self) -> Result<i64> {
        let conn = self.lock();
        conn.query_row("SELECT COUNT(*) FROM users", [], |r| r.get(0))
            .map_err(Error::from)
    }

    /// Replace a user's password (Argon2id PHC).
    pub fn set_password(&self, user_id: &str, password: &str) -> Result<()> {
        let phc = password_auth::generate_hash(password.as_bytes());
        let conn = self.lock();
        let n = conn.execute(
            "INSERT INTO password_credentials (user_id, phc_hash, updated_at) VALUES (?1, ?2, ?3) \
             ON CONFLICT(user_id) DO UPDATE SET phc_hash = excluded.phc_hash, \
             updated_at = excluded.updated_at",
            params![user_id, phc, now_secs()],
        )?;
        if n == 0 {
            return Err(Error::NotFound);
        }
        // Re-derive the SCRAM-SHA-256 material so PLAIN (Argon2) and SCRAM stay coherent.
        upsert_scram(&conn, user_id, password)?;
        Ok(())
    }

    /// Enable/disable an account. A disabled account cannot authenticate; existing sessions are
    /// revoked so the change takes effect immediately.
    pub fn set_disabled(&self, user_id: &str, disabled: bool) -> Result<()> {
        let conn = self.lock();
        let n = conn.execute(
            "UPDATE users SET disabled = ?2 WHERE id = ?1",
            params![user_id, i64::from(disabled)],
        )?;
        if n == 0 {
            return Err(Error::NotFound);
        }
        if disabled {
            conn.execute(
                "DELETE FROM auth_sessions WHERE user_id = ?1",
                params![user_id],
            )?;
        }
        Ok(())
    }

    // --- roles ------------------------------------------------------------------------------

    /// The roles assigned to a user (unknown stored strings are skipped defensively).
    pub fn roles_of(&self, user_id: &str) -> Result<Vec<Role>> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT role FROM user_roles WHERE user_id = ?1")?;
        let rows = stmt.query_map(params![user_id], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for s in rows {
            if let Some(role) = Role::from_wire(&s?) {
                out.push(role);
            }
        }
        Ok(out)
    }

    /// Replace a user's role set wholesale.
    pub fn set_roles(&self, user_id: &str, roles: &[Role]) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "DELETE FROM user_roles WHERE user_id = ?1",
            params![user_id],
        )?;
        for role in roles {
            conn.execute(
                "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?1, ?2)",
                params![user_id, role.as_str()],
            )?;
        }
        Ok(())
    }

    // --- authentication ---------------------------------------------------------------------

    /// Verify a username + password (the PLAIN-over-TLS / Argon2id login path) and resolve the
    /// caller's [`Principal`]. Errors are deliberately coarse to avoid a username-probing oracle.
    pub fn authenticate_password(&self, username: &str, password: &str) -> Result<Principal> {
        let user = self.find_user(username)?.ok_or(Error::InvalidCredentials)?;
        if user.disabled {
            return Err(Error::Disabled);
        }
        let phc: Option<String> = {
            let conn = self.lock();
            conn.query_row(
                "SELECT phc_hash FROM password_credentials WHERE user_id = ?1",
                params![user.id],
                |r| r.get(0),
            )
            .optional()?
        };
        let phc = phc.ok_or(Error::InvalidCredentials)?;
        password_auth::verify_password(password.as_bytes(), &phc)
            .map_err(|_| Error::InvalidCredentials)?;
        self.principal_for_user(&user.id, &user.username)
    }

    /// Build the [`Principal`] for an already-identified user (loads + unions roles).
    pub fn principal_for_user(&self, user_id: &str, username: &str) -> Result<Principal> {
        let roles = self.roles_of(user_id)?;
        Ok(Principal::from_roles(user_id, username, roles))
    }

    // --- SCRAM credentials ------------------------------------------------------------------

    /// Derive + persist the `SCRAM-SHA-256` material for `user_id` from `password` (the explicit
    /// entry point; `create_user`/`set_password` call the same derivation inline). Errors with
    /// [`Error::NotFound`] if the user row does not exist.
    pub fn set_scram_credentials(&self, user_id: &str, password: &str) -> Result<()> {
        let conn = self.lock();
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM users WHERE id = ?1",
                params![user_id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !exists {
            return Err(Error::NotFound);
        }
        upsert_scram(&conn, user_id, password)
    }

    /// Fetch the persisted SCRAM material for `user_id` + `mechanism` (e.g. `"SCRAM-SHA-256"`), or
    /// `None` if absent. The authenticator feeds this into rsasl's `ScramStoredPassword` property.
    /// Legacy users created before SCRAM material was derived have no row until a password re-set.
    pub fn scram_credentials_for(
        &self,
        user_id: &str,
        mechanism: &str,
    ) -> Result<Option<ScramMaterial>> {
        let conn = self.lock();
        conn.query_row(
            "SELECT salt, iterations, stored_key, server_key FROM scram_credentials \
             WHERE user_id = ?1 AND mechanism = ?2",
            params![user_id, mechanism],
            |r| {
                Ok(ScramMaterial {
                    salt: r.get(0)?,
                    iterations: r.get::<_, i64>(1)? as u32,
                    stored_key: r.get(2)?,
                    server_key: r.get(3)?,
                })
            },
        )
        .optional()
        .map_err(Error::from)
    }

    // --- session tokens ---------------------------------------------------------------------

    /// Mint an opaque session token for `user_id` valid for `ttl_secs`, returning the *plaintext*
    /// token (only its hash is persisted). `method` records how the session was established (audit).
    pub fn mint_session(&self, user_id: &str, ttl_secs: i64, method: &str) -> Result<String> {
        let token = random_hex()?;
        let now = now_secs();
        let conn = self.lock();
        conn.execute(
            "INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at, auth_method) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![token_hash(&token), user_id, now, now + ttl_secs, method],
        )?;
        Ok(token)
    }

    /// Resolve a session token to its [`Principal`], or `Err(NotFound)` if unknown/expired. Expired
    /// rows are pruned opportunistically.
    pub fn principal_for_token(&self, token: &str) -> Result<Principal> {
        let hash = token_hash(token);
        let row: Option<(String, i64)> = {
            let conn = self.lock();
            conn.query_row(
                "SELECT user_id, expires_at FROM auth_sessions WHERE token_hash = ?1",
                params![hash],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?
        };
        let (user_id, expires_at) = row.ok_or(Error::NotFound)?;
        if expires_at <= now_secs() {
            let conn = self.lock();
            conn.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?1",
                params![hash],
            )?;
            return Err(Error::NotFound);
        }
        let user = {
            let conn = self.lock();
            conn.query_row(
                "SELECT username, disabled FROM users WHERE id = ?1",
                params![user_id],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? != 0)),
            )
            .optional()?
        };
        let (username, disabled) = user.ok_or(Error::NotFound)?;
        if disabled {
            return Err(Error::Disabled);
        }
        self.principal_for_user(&user_id, &username)
    }

    /// Revoke a single session token.
    pub fn revoke_token(&self, token: &str) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "DELETE FROM auth_sessions WHERE token_hash = ?1",
            params![token_hash(token)],
        )?;
        Ok(())
    }

    /// Count of currently-stored session tokens (live plus any not-yet-pruned expired rows). A
    /// lightweight audit/observability aid — also lets tests assert a token is minted only on a
    /// successful authentication.
    pub fn session_count(&self) -> Result<i64> {
        let conn = self.lock();
        conn.query_row("SELECT COUNT(*) FROM auth_sessions", [], |r| r.get(0))
            .map_err(Error::from)
    }

    /// Revoke every session for a user (e.g. on password change / lockout).
    pub fn revoke_user_sessions(&self, user_id: &str) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "DELETE FROM auth_sessions WHERE user_id = ?1",
            params![user_id],
        )?;
        Ok(())
    }

    // --- EXTERNAL (mTLS) identities (Auth 4) ------------------------------------------------

    /// Map a verified client-certificate `fingerprint` to a user (SASL EXTERNAL enrollment). Upsert:
    /// re-binding a fingerprint moves it to the new user. Errors with [`Error::NotFound`] if the
    /// target user row does not exist. The admin op that exposes this is deferred (Auth 5); this is
    /// the store-level writer so the read path is testable end-to-end.
    pub fn set_external_identity(
        &self,
        user_id: &str,
        fingerprint: &str,
        display_name: Option<&str>,
    ) -> Result<()> {
        let conn = self.lock();
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM users WHERE id = ?1",
                params![user_id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !exists {
            return Err(Error::NotFound);
        }
        conn.execute(
            "INSERT INTO external_identities (fingerprint, user_id, created_at, display_name) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(fingerprint) DO UPDATE SET user_id = excluded.user_id, \
             created_at = excluded.created_at, display_name = excluded.display_name",
            params![fingerprint, user_id, now_secs(), display_name],
        )?;
        Ok(())
    }

    /// Record a successful EXTERNAL login on the fingerprint mapping (`last_seen_at`, M3).
    /// Best-effort from the auth path: an unknown fingerprint is a no-op, never an error.
    pub fn touch_external_identity(&self, fingerprint: &str) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "UPDATE external_identities SET last_seen_at = ?2 WHERE fingerprint = ?1",
            params![fingerprint, now_secs()],
        )?;
        Ok(())
    }

    /// Remove a fingerprint mapping. Returns whether a row was deleted (an unknown fingerprint is
    /// `false`, not an error). Revocation of the *user* (disable, session teardown) is the admin
    /// surface's responsibility — this only severs the certificate mapping.
    pub fn remove_external_identity(&self, fingerprint: &str) -> Result<bool> {
        let conn = self.lock();
        let n = conn.execute(
            "DELETE FROM external_identities WHERE fingerprint = ?1",
            params![fingerprint],
        )?;
        Ok(n > 0)
    }

    /// All fingerprint mappings joined with their owning user (the paired-devices admin view,
    /// pairing spec §5.5), newest first.
    pub fn list_external_identities(&self) -> Result<Vec<ExternalIdentityRecord>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT e.fingerprint, e.user_id, u.username, e.display_name, e.created_at, \
                    e.last_seen_at, u.disabled \
             FROM external_identities e JOIN users u ON u.id = e.user_id \
             ORDER BY e.created_at DESC, e.fingerprint",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ExternalIdentityRecord {
                fingerprint: r.get(0)?,
                user_id: r.get(1)?,
                username: r.get(2)?,
                display_name: r.get(3)?,
                created_at: r.get(4)?,
                last_seen_at: r.get(5)?,
                disabled: r.get::<_, i64>(6)? != 0,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Error::from)
    }

    /// Resolve a verified client-certificate `fingerprint` to its `(user_id, username)`, or `None`
    /// when unmapped or the mapped user is disabled (fail-closed: an unmapped/disabled cert is never
    /// trusted). The SASL EXTERNAL mechanism feeds the result into the authenticated principal.
    pub fn external_identity(&self, fingerprint: &str) -> Result<Option<(String, String)>> {
        let conn = self.lock();
        conn.query_row(
            "SELECT u.id, u.username FROM external_identities e \
             JOIN users u ON u.id = e.user_id \
             WHERE e.fingerprint = ?1 AND u.disabled = 0",
            params![fingerprint],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(Error::from)
    }

    // --- last-admin lockout (atomic) --------------------------------------------------------
    //
    // The check-and-modify runs in ONE SQLite transaction so two concurrent demotions/disables
    // cannot each observe "another admin still exists" and both commit, orphaning the node. These
    // are the only access-admin mutations that can reduce the enabled-admin set, so they are the
    // only ones that need the guard (role *grants* and *enables* can never orphan admins).

    /// Count enabled admins **other than** `exclude_user_id` (the survivors if that user is
    /// disabled/demoted). Run inside the guarded transaction.
    fn other_enabled_admins(conn: &Connection, exclude_user_id: &str) -> Result<i64> {
        conn.query_row(
            "SELECT COUNT(DISTINCT u.id) FROM users u \
             JOIN user_roles r ON r.user_id = u.id \
             WHERE r.role = ?1 AND u.disabled = 0 AND u.id <> ?2",
            params![Role::Admin.as_str(), exclude_user_id],
            |row| row.get(0),
        )
        .map_err(Error::from)
    }

    /// Whether `user_id` is currently an enabled administrator. Run inside the guarded transaction.
    fn is_enabled_admin(conn: &Connection, user_id: &str) -> Result<bool> {
        conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM users u \
             JOIN user_roles r ON r.user_id = u.id \
             WHERE u.id = ?1 AND r.role = ?2 AND u.disabled = 0)",
            params![user_id, Role::Admin.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .map(|n| n != 0)
        .map_err(Error::from)
    }

    /// Enable/disable an account atomically, refusing (`Err(Error::LastAdmin)`) to disable the last
    /// enabled administrator. On disable, the user's sessions are revoked in the same transaction so
    /// the lockout decision and its effects commit together. `Err(Error::NotFound)` for an unknown
    /// user. Prefer this over [`set_disabled`] on the admin surface.
    pub fn set_disabled_guarded(&self, user_id: &str, disabled: bool) -> Result<()> {
        let mut guard = self.lock();
        let tx = guard.transaction()?;
        if disabled
            && Self::is_enabled_admin(&tx, user_id)?
            && Self::other_enabled_admins(&tx, user_id)? == 0
        {
            return Err(Error::LastAdmin);
        }
        let n = tx.execute(
            "UPDATE users SET disabled = ?2 WHERE id = ?1",
            params![user_id, i64::from(disabled)],
        )?;
        if n == 0 {
            return Err(Error::NotFound);
        }
        if disabled {
            tx.execute(
                "DELETE FROM auth_sessions WHERE user_id = ?1",
                params![user_id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Replace a user's role set atomically, refusing (`Err(Error::LastAdmin)`) to demote the last
    /// enabled administrator out of `admin`. Granting roles (or keeping `admin`) never trips the
    /// guard. Prefer this over [`set_roles`] on the admin surface.
    pub fn set_roles_guarded(&self, user_id: &str, roles: &[Role]) -> Result<()> {
        let mut guard = self.lock();
        let tx = guard.transaction()?;
        let new_is_admin = roles.contains(&Role::Admin);
        if !new_is_admin
            && Self::is_enabled_admin(&tx, user_id)?
            && Self::other_enabled_admins(&tx, user_id)? == 0
        {
            return Err(Error::LastAdmin);
        }
        tx.execute(
            "DELETE FROM user_roles WHERE user_id = ?1",
            params![user_id],
        )?;
        for role in roles {
            tx.execute(
                "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?1, ?2)",
                params![user_id, role.as_str()],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::Capability;

    fn store() -> AuthStore {
        AuthStore::open_in_memory().expect("open")
    }

    /// The synthetic in-process usernames (`system` / `internal`) are reserved: a real store user
    /// cannot be created with them (any case), so a network identity can never forge an ownership
    /// stamp that collides with a trusted internal caller. A normal username still succeeds.
    #[test]
    fn reserved_usernames_cannot_be_created() {
        let s = store();
        for reserved in ["system", "internal", "System", "INTERNAL"] {
            assert!(
                matches!(
                    s.create_user(reserved, "pw", &[Role::User]),
                    Err(Error::ReservedUsername)
                ),
                "creating reserved username {reserved:?} must fail"
            );
        }
        // A normal username is unaffected.
        assert!(s.create_user("alice", "pw", &[Role::User]).is_ok());
    }

    #[test]
    fn migration_ladder_valid_and_applied() {
        assert!(MIGRATIONS.validate().is_ok());
        let s = store();
        let v: i64 = s
            .lock()
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(v, 3);
    }

    #[test]
    fn external_identity_resolves_written_fingerprint_and_denies_unknown() {
        let s = store();
        let u = s.create_user("svc", "pw", &[Role::Operator]).unwrap();
        // Unmapped fingerprint: deny (fail-closed).
        assert!(s.external_identity("ab12cd34").unwrap().is_none());
        // Writing then reading resolves to the right user.
        s.set_external_identity(&u.id, "ab12cd34", None).unwrap();
        let (uid, name) = s.external_identity("ab12cd34").unwrap().expect("mapped");
        assert_eq!(uid, u.id);
        assert_eq!(name, "svc");
        // A still-unknown fingerprint stays denied.
        assert!(s.external_identity("ffffffff").unwrap().is_none());
    }

    #[test]
    fn external_identity_requires_existing_user_and_hides_disabled() {
        let s = store();
        // Binding to a nonexistent user is rejected.
        assert!(matches!(
            s.set_external_identity("ghost", "aa", None),
            Err(Error::NotFound)
        ));
        // A disabled user's mapping is not resolvable (disable hides the EXTERNAL identity too).
        let u = s.create_user("dora", "pw", &[Role::User]).unwrap();
        s.set_external_identity(&u.id, "bb22", None).unwrap();
        assert!(s.external_identity("bb22").unwrap().is_some());
        s.set_disabled(&u.id, true).unwrap();
        assert!(s.external_identity("bb22").unwrap().is_none());
    }

    #[test]
    fn external_identity_rebinds_and_cascades_on_user_delete() {
        let s = store();
        let a = s.create_user("a", "pw", &[Role::User]).unwrap();
        let b = s.create_user("b", "pw", &[Role::User]).unwrap();
        // Re-binding a fingerprint moves it to the new user (upsert).
        s.set_external_identity(&a.id, "deadbeef", None).unwrap();
        s.set_external_identity(&b.id, "deadbeef", None).unwrap();
        assert_eq!(
            s.external_identity("deadbeef").unwrap().map(|(id, _)| id),
            Some(b.id.clone())
        );
        // ON DELETE CASCADE: removing the user drops the mapping.
        s.lock()
            .execute("DELETE FROM users WHERE id = ?1", params![b.id])
            .unwrap();
        assert!(s.external_identity("deadbeef").unwrap().is_none());
    }

    #[test]
    fn external_user_has_no_password_credentials() {
        let s = store();
        let u = s
            .create_external_user("device-ab12cd34dead", &[Role::User])
            .unwrap();
        assert_eq!(u.username, "device-ab12cd34dead");
        // No password/SCRAM rows: PLAIN authentication fails closed.
        assert!(matches!(
            s.authenticate_password("device-ab12cd34dead", "anything"),
            Err(Error::InvalidCredentials)
        ));
        let scram_rows: i64 = s
            .lock()
            .query_row(
                "SELECT COUNT(*) FROM scram_credentials WHERE user_id = ?1",
                params![u.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(scram_rows, 0, "an external user never gets SCRAM material");
        // Reserved usernames stay rejected on this path too.
        assert!(matches!(
            s.create_external_user("system", &[Role::User]),
            Err(Error::ReservedUsername)
        ));
    }

    /// The M3 device metadata: display name at enrollment, `last_seen_at` on touch, the joined
    /// admin list view, and mapping removal.
    #[test]
    fn external_identity_device_metadata_roundtrip() {
        let s = store();
        let u = s.create_external_user("device-aa", &[Role::User]).unwrap();
        s.set_external_identity(&u.id, "aabbccdd", Some("Pixel 9"))
            .unwrap();

        let rows = s.list_external_identities().unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.fingerprint, "aabbccdd");
        assert_eq!(row.user_id, u.id);
        assert_eq!(row.username, "device-aa");
        assert_eq!(row.display_name.as_deref(), Some("Pixel 9"));
        assert!(row.created_at > 0);
        assert_eq!(row.last_seen_at, None, "never logged in yet");
        assert!(!row.disabled);

        // A successful EXTERNAL login stamps last_seen_at; unknown fingerprints are a no-op.
        s.touch_external_identity("aabbccdd").unwrap();
        s.touch_external_identity("ffffffff").unwrap();
        let row = &s.list_external_identities().unwrap()[0];
        assert!(row.last_seen_at.is_some());

        // Removal severs the mapping (and reports whether anything was removed).
        assert!(s.remove_external_identity("aabbccdd").unwrap());
        assert!(!s.remove_external_identity("aabbccdd").unwrap());
        assert!(s.external_identity("aabbccdd").unwrap().is_none());
        assert!(s.list_external_identities().unwrap().is_empty());
    }

    #[test]
    fn create_authenticate_and_resolve_roles() {
        let s = store();
        let u = s
            .create_user("alice", "correct horse", &[Role::User])
            .unwrap();
        // Wrong password is rejected; right password resolves a principal with User caps.
        assert!(matches!(
            s.authenticate_password("alice", "nope"),
            Err(Error::InvalidCredentials)
        ));
        let p = s.authenticate_password("alice", "correct horse").unwrap();
        assert_eq!(p.user_id, u.id);
        assert!(p.has(Capability::SessionWrite));
        assert!(!p.has(Capability::AccessAdmin));
    }

    #[test]
    fn unknown_user_is_indistinguishable_from_bad_password() {
        let s = store();
        assert!(matches!(
            s.authenticate_password("ghost", "x"),
            Err(Error::InvalidCredentials)
        ));
    }

    #[test]
    fn disabled_account_cannot_authenticate() {
        let s = store();
        let u = s.create_user("bob", "pw", &[Role::User]).unwrap();
        s.set_disabled(&u.id, true).unwrap();
        assert!(matches!(
            s.authenticate_password("bob", "pw"),
            Err(Error::Disabled)
        ));
    }

    #[test]
    fn session_token_round_trips_and_revokes() {
        let s = store();
        let u = s.create_user("carol", "pw", &[Role::Operator]).unwrap();
        let token = s
            .mint_session(&u.id, DEFAULT_SESSION_TTL_SECS, "password")
            .unwrap();
        let p = s.principal_for_token(&token).unwrap();
        assert_eq!(p.username, "carol");
        assert!(p.can_see_all_sessions());
        s.revoke_token(&token).unwrap();
        assert!(matches!(
            s.principal_for_token(&token),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn expired_token_is_rejected_and_pruned() {
        let s = store();
        let u = s.create_user("dave", "pw", &[Role::Viewer]).unwrap();
        // Negative TTL => already expired.
        let token = s.mint_session(&u.id, -1, "password").unwrap();
        assert!(matches!(
            s.principal_for_token(&token),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn create_user_populates_scram_material_and_set_password_rederives() {
        let s = store();
        let u = s.create_user("frank", "pw-one", &[Role::User]).unwrap();
        let first = s
            .scram_credentials_for(&u.id, crate::scram::SCRAM_SHA_256)
            .unwrap()
            .expect("create_user derives SCRAM material");
        assert_eq!(first.iterations, crate::scram::SCRAM_DEFAULT_ITERATIONS);
        assert_eq!(first.stored_key.len(), 32);
        assert_eq!(first.server_key.len(), 32);

        // A password change re-derives (new salt => new keys).
        s.set_password(&u.id, "pw-two").unwrap();
        let second = s
            .scram_credentials_for(&u.id, crate::scram::SCRAM_SHA_256)
            .unwrap()
            .unwrap();
        assert_ne!(
            first.salt, second.salt,
            "set_password mints fresh SCRAM material"
        );
        assert_ne!(first.stored_key, second.stored_key);
    }

    #[test]
    fn scram_credentials_for_unknown_user_is_none() {
        let s = store();
        assert!(s
            .scram_credentials_for("nobody", crate::scram::SCRAM_SHA_256)
            .unwrap()
            .is_none());
    }

    #[test]
    fn set_scram_credentials_requires_existing_user() {
        let s = store();
        assert!(matches!(
            s.set_scram_credentials("ghost", "pw"),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn set_roles_replaces_and_disable_revokes_sessions() {
        let s = store();
        let u = s.create_user("erin", "pw", &[Role::User]).unwrap();
        let token = s
            .mint_session(&u.id, DEFAULT_SESSION_TTL_SECS, "password")
            .unwrap();
        s.set_roles(&u.id, &[Role::Admin]).unwrap();
        assert!(s
            .principal_for_token(&token)
            .unwrap()
            .has(Capability::AccessAdmin));
        // Disabling revokes the user's sessions, so the token no longer resolves.
        s.set_disabled(&u.id, true).unwrap();
        assert!(matches!(
            s.principal_for_token(&token),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn last_admin_cannot_be_disabled_or_demoted() {
        let s = store();
        let admin = s.create_user("root", "pw", &[Role::Admin]).unwrap();
        // Sole admin: disabling or demoting is refused.
        assert!(matches!(
            s.set_disabled_guarded(&admin.id, true),
            Err(Error::LastAdmin)
        ));
        assert!(matches!(
            s.set_roles_guarded(&admin.id, &[Role::Operator]),
            Err(Error::LastAdmin)
        ));
        // The guard did not mutate: the admin is still an enabled admin.
        assert!(!s.find_user("root").unwrap().unwrap().disabled);
        assert_eq!(s.roles_of(&admin.id).unwrap(), vec![Role::Admin]);

        // With a second admin, demoting/disabling the first is allowed.
        let admin2 = s.create_user("root2", "pw", &[Role::Admin]).unwrap();
        s.set_roles_guarded(&admin.id, &[Role::Operator]).unwrap();
        assert_eq!(s.roles_of(&admin.id).unwrap(), vec![Role::Operator]);
        // Now admin2 is the last admin again -> refused.
        assert!(matches!(
            s.set_disabled_guarded(&admin2.id, true),
            Err(Error::LastAdmin)
        ));
    }

    #[test]
    fn disabling_a_non_admin_is_never_blocked_by_the_guard() {
        let s = store();
        // No admins at all; disabling a plain user must still work (guard only protects admins).
        let u = s.create_user("viewer", "pw", &[Role::Viewer]).unwrap();
        s.set_disabled_guarded(&u.id, true).unwrap();
        assert!(s.find_user("viewer").unwrap().unwrap().disabled);
    }

    #[test]
    fn concurrent_disable_of_two_admins_keeps_one() {
        use std::sync::Arc;
        // Two admins; two threads each try to disable a *different* admin at the same time. The
        // single-transaction guard must let exactly one succeed (the other sees itself as the last
        // enabled admin and is refused), so the node never loses every admin.
        let s = Arc::new(store());
        let a = s.create_user("a", "pw", &[Role::Admin]).unwrap();
        let b = s.create_user("b", "pw", &[Role::Admin]).unwrap();
        let (s1, s2) = (s.clone(), s.clone());
        let (ida, idb) = (a.id.clone(), b.id.clone());
        let t1 = std::thread::spawn(move || s1.set_disabled_guarded(&ida, true));
        let t2 = std::thread::spawn(move || s2.set_disabled_guarded(&idb, true));
        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();
        // Exactly one succeeded; the other hit the last-admin guard.
        let successes = [r1.is_ok(), r2.is_ok()].iter().filter(|ok| **ok).count();
        assert_eq!(successes, 1, "exactly one disable may win");
        // At least one enabled admin remains.
        let enabled = [&a, &b]
            .iter()
            .filter(|u| !s.find_user(&u.username).unwrap().unwrap().disabled)
            .count();
        assert!(enabled >= 1, "an enabled admin must survive");
    }
}
