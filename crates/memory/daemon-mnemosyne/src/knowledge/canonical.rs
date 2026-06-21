//! CanonicalStore — port of `canonical.py`.
//!
//! Owner-scoped identity cards with monotonic version chains and a partial unique index on live rows
//! (`WHERE valid_until IS NULL`). `remember()` returns `created | unchanged | updated`
//! (`canonical.py` L196-L287).

use crate::error::{Error, Result};
use crate::util;
use rusqlite::{params, Connection, OptionalExtension};

/// Outcome of [`remember`] (`canonical.py` L213-L214).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The slot had no live value (brand-new or previously retired).
    Created,
    /// The current body was identical — no-op.
    Unchanged,
    /// A live value was superseded by a new version.
    Updated,
}

/// A canonical fact row.
#[derive(Clone, Debug)]
pub struct CanonicalRow {
    /// Row id.
    pub id: i64,
    /// Owner scope.
    pub owner_id: String,
    /// Category.
    pub category: String,
    /// Slot name.
    pub name: String,
    /// Current body.
    pub body: String,
    /// Monotonic version.
    pub version: i64,
}

/// Upsert the canonical value for `(owner_id, category, name)` (`canonical.py` `remember`
/// L196-L287). Inserts version 1 if empty; no-ops if the body is unchanged; otherwise supersedes
/// (stamps `valid_until` on the current row) and inserts `version + 1`. Returns the resulting live
/// row plus a [`Status`].
pub fn remember(
    conn: &Connection,
    owner_id: &str,
    category: &str,
    name: &str,
    body: &str,
    source: &str,
    confidence: f64,
) -> Result<(CanonicalRow, Status)> {
    if owner_id.is_empty() || category.is_empty() || name.is_empty() {
        return Err(Error::Invalid(
            "owner_id, category, and name are required".into(),
        ));
    }
    if body.trim().is_empty() {
        return Err(Error::Invalid("body is required and cannot be blank".into()));
    }

    let current: Option<(i64, String)> = conn
        .query_row(
            "SELECT id, body FROM canonical_facts \
             WHERE owner_id = ?1 AND category = ?2 AND name = ?3 AND valid_until IS NULL",
            params![owner_id, category, name],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)),
        )
        .optional()?;

    if let Some((id, current_body)) = &current {
        if current_body == body {
            return Ok((
                fetch_by_id(conn, *id)?,
                Status::Unchanged,
            ));
        }
    }

    let now = util::now_iso();
    let prior_max: Option<i64> = conn.query_row(
        "SELECT MAX(version) FROM canonical_facts WHERE owner_id = ?1 AND category = ?2 AND name = ?3",
        params![owner_id, category, name],
        |r| r.get(0),
    )?;
    let version = prior_max.unwrap_or(0) + 1;
    let status = match &current {
        None => Status::Created,
        Some((id, _)) => {
            conn.execute(
                "UPDATE canonical_facts SET valid_until = ?1 WHERE id = ?2",
                params![now, id],
            )?;
            Status::Updated
        }
    };
    conn.execute(
        "INSERT INTO canonical_facts \
         (owner_id, category, name, body, source, confidence, version, valid_from, valid_until) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL)",
        params![owner_id, category, name, body, source, confidence, version, now],
    )?;
    let new_id = conn.last_insert_rowid();
    Ok((fetch_by_id(conn, new_id)?, status))
}

/// Retire a canonical slot without replacing it (`canonical.py` `forget` L289+). Returns `true` if
/// a live row was retired.
pub fn forget(conn: &Connection, owner_id: &str, category: &str, name: &str) -> Result<bool> {
    let now = util::now_iso();
    let n = conn.execute(
        "UPDATE canonical_facts SET valid_until = ?1 \
         WHERE owner_id = ?2 AND category = ?3 AND name = ?4 AND valid_until IS NULL",
        params![now, owner_id, category, name],
    )?;
    Ok(n > 0)
}

fn fetch_by_id(conn: &Connection, id: i64) -> Result<CanonicalRow> {
    let row = conn.query_row(
        "SELECT id, owner_id, category, name, body, version FROM canonical_facts WHERE id = ?1",
        params![id],
        |r| {
            Ok(CanonicalRow {
                id: r.get(0)?,
                owner_id: r.get(1)?,
                category: r.get(2)?,
                name: r.get(3)?,
                body: r.get(4)?,
                version: r.get(5)?,
            })
        },
    )?;
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    #[test]
    fn versioned_remember_and_forget() {
        let store = Store::open_in_memory().unwrap();
        let c = store.conn.lock().unwrap();

        let (r1, s1) = remember(&c, "alice", "identity", "role", "engineer", "t", 1.0).unwrap();
        assert_eq!(s1, Status::Created);
        assert_eq!(r1.version, 1);

        let (_r2, s2) = remember(&c, "alice", "identity", "role", "engineer", "t", 1.0).unwrap();
        assert_eq!(s2, Status::Unchanged);

        let (r3, s3) = remember(&c, "alice", "identity", "role", "manager", "t", 1.0).unwrap();
        assert_eq!(s3, Status::Updated);
        assert_eq!(r3.version, 2);
        assert_eq!(r3.body, "manager");

        assert!(forget(&c, "alice", "identity", "role").unwrap());
        // After forget the slot is empty; a new remember climbs the version chain (created).
        let (r4, s4) = remember(&c, "alice", "identity", "role", "founder", "t", 1.0).unwrap();
        assert_eq!(s4, Status::Created);
        assert_eq!(r4.version, 3);
    }
}
