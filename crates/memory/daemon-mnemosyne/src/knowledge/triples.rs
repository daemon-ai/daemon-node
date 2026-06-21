//! Temporal TripleStore — port of `triples.py`.
//!
//! Single-current-truth SPO chains: `add(supersede=true)` stamps prior open rows' `valid_until` to
//! the new `valid_from` then inserts; `query(as_of)` selects `valid_from <= as_of AND (valid_until
//! IS NULL OR valid_until > as_of)` (`triples.py` L163-L227). Schema lives in
//! [`crate::store::schema`].

use crate::error::Result;
use crate::util;
use rusqlite::{params_from_iter, types::Value, Connection};

/// A temporal triple row (`triples` table, `triples.py` L91-L108).
#[derive(Clone, Debug)]
pub struct Triple {
    /// Subject.
    pub subject: String,
    /// Predicate.
    pub predicate: String,
    /// Object.
    pub object: String,
    /// Start of validity (ISO date).
    pub valid_from: String,
    /// End of validity (ISO date), `None` while open/current.
    pub valid_until: Option<String>,
    /// Source tag.
    pub source: String,
    /// Confidence `[0, 1]`.
    pub confidence: f64,
}

/// Add a triple (`triples.py` `add` L138-L176). `supersede=true` closes prior open rows for
/// `(subject, predicate)` by stamping their `valid_until` to the new `valid_from` (single-valued
/// truth); `supersede=false` leaves priors open (multi-valued predicates). `valid_from` defaults to
/// today; `valid_until` is an optional explicit expiry. Returns the new row id.
#[allow(clippy::too_many_arguments)]
pub fn add(
    conn: &Connection,
    subject: &str,
    predicate: &str,
    object: &str,
    valid_from: Option<&str>,
    valid_until: Option<&str>,
    source: &str,
    confidence: f64,
    supersede: bool,
) -> Result<i64> {
    let valid_from = valid_from.map(str::to_string).unwrap_or_else(util::today_iso);
    if supersede {
        conn.execute(
            "UPDATE triples SET valid_until = ?1 \
             WHERE subject = ?2 AND predicate = ?3 AND valid_until IS NULL",
            rusqlite::params![valid_from, subject, predicate],
        )?;
    }
    conn.execute(
        "INSERT INTO triples (subject, predicate, object, valid_from, valid_until, source, confidence) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![subject, predicate, object, valid_from, valid_until, source, confidence],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Expire open triples without replacing them (`triples.py` `end` L178-L197). Closes all open rows
/// for `(subject, predicate)`, or only the one matching `object` when supplied. Returns the number
/// of rows closed.
pub fn end(
    conn: &Connection,
    subject: &str,
    predicate: &str,
    object: Option<&str>,
    valid_until: Option<&str>,
) -> Result<usize> {
    let valid_until = valid_until
        .map(str::to_string)
        .unwrap_or_else(util::today_iso);
    let mut sql = String::from(
        "UPDATE triples SET valid_until = ?1 \
         WHERE subject = ?2 AND predicate = ?3 AND valid_until IS NULL",
    );
    let mut binds: Vec<Value> = vec![
        Value::Text(valid_until),
        Value::Text(subject.to_string()),
        Value::Text(predicate.to_string()),
    ];
    if let Some(obj) = object {
        sql.push_str(" AND object = ?4");
        binds.push(Value::Text(obj.to_string()));
    }
    let n = conn.execute(&sql, params_from_iter(binds))?;
    Ok(n)
}

/// Query triples valid at `as_of` (default today), optionally filtered by subject (case-insensitive),
/// predicate, and object (`triples.py` `query` L199-L229). Ordered by `valid_from DESC`.
pub fn query(
    conn: &Connection,
    subject: Option<&str>,
    predicate: Option<&str>,
    object: Option<&str>,
    as_of: Option<&str>,
) -> Result<Vec<Triple>> {
    let as_of = as_of.map(str::to_string).unwrap_or_else(util::today_iso);
    let mut conditions: Vec<String> = Vec::new();
    let mut binds: Vec<Value> = Vec::new();
    if let Some(s) = subject {
        conditions.push("subject = ? COLLATE NOCASE".to_string());
        binds.push(Value::Text(s.to_string()));
    }
    if let Some(p) = predicate {
        conditions.push("predicate = ?".to_string());
        binds.push(Value::Text(p.to_string()));
    }
    if let Some(o) = object {
        conditions.push("object = ?".to_string());
        binds.push(Value::Text(o.to_string()));
    }
    conditions.push("valid_from <= ?".to_string());
    binds.push(Value::Text(as_of.clone()));
    conditions.push("(valid_until IS NULL OR valid_until > ?)".to_string());
    binds.push(Value::Text(as_of));

    let sql = format!(
        "SELECT subject, predicate, object, valid_from, valid_until, source, confidence \
         FROM triples WHERE {} ORDER BY valid_from DESC",
        conditions.join(" AND ")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(params_from_iter(binds), |r| {
            Ok(Triple {
                subject: r.get(0)?,
                predicate: r.get(1)?,
                object: r.get(2)?,
                valid_from: r.get(3)?,
                valid_until: r.get::<_, Option<String>>(4)?,
                source: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                confidence: r.get::<_, Option<f64>>(6)?.unwrap_or(1.0),
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn conn() -> Store {
        Store::open_in_memory().expect("store")
    }

    #[test]
    fn supersede_closes_prior_and_as_of_history() {
        let store = conn();
        let c = store.conn.lock().unwrap();
        add(&c, "user", "city", "Berlin", Some("2024-01-01"), None, "test", 1.0, true).unwrap();
        add(&c, "user", "city", "Lisbon", Some("2024-06-01"), None, "test", 1.0, true).unwrap();

        // Current truth: only Lisbon is open.
        let now = query(&c, Some("user"), Some("city"), None, Some("2024-07-01")).unwrap();
        assert_eq!(now.len(), 1);
        assert_eq!(now[0].object, "Lisbon");

        // As-of before the move: Berlin was the open value.
        let then = query(&c, Some("user"), Some("city"), None, Some("2024-03-01")).unwrap();
        assert_eq!(then.len(), 1);
        assert_eq!(then[0].object, "Berlin");
    }

    #[test]
    fn multivalued_predicate_keeps_priors_open() {
        let store = conn();
        let c = store.conn.lock().unwrap();
        add(&c, "user", "speaks", "English", Some("2024-01-01"), None, "t", 1.0, false).unwrap();
        add(&c, "user", "speaks", "Spanish", Some("2024-01-01"), None, "t", 1.0, false).unwrap();
        let langs = query(&c, Some("user"), Some("speaks"), None, Some("2024-02-01")).unwrap();
        assert_eq!(langs.len(), 2);
    }

    #[test]
    fn end_closes_open_rows() {
        let store = conn();
        let c = store.conn.lock().unwrap();
        add(&c, "user", "role", "admin", Some("2024-01-01"), None, "t", 1.0, false).unwrap();
        let closed = end(&c, "user", "role", None, Some("2024-05-01")).unwrap();
        assert_eq!(closed, 1);
        let after = query(&c, Some("user"), Some("role"), None, Some("2024-06-01")).unwrap();
        assert!(after.is_empty());
    }
}
