// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tool-surface backing methods for the BEAM [`Engine`]: get/update/forget/invalidate/validate,
//! the audit-log writer, stats/diagnostics, scratchpad, and export/import. Split out of
//! `engine.rs` (W-MNEMO).

use super::*;
use crate::config::RecallScope;
use crate::util;
use rusqlite::{params, Connection};
use serde_json::json;

impl Engine {
    // ── Tool-surface backing methods (`beam.py` get/update/forget/invalidate/validate/stats/...) ──

    /// Fetch a single live memory by id, working tier first then episodic (`beam.py` `get`).
    pub fn get(&self, id: &str) -> Result<Option<MemoryRow>> {
        let conn = self.store.conn.lock().unwrap();
        let scope = RecallScope::default();
        if let Some(row) = self.fetch_working(&conn, id, &scope)? {
            return Ok(Some(row));
        }
        self.fetch_episodic(&conn, id, &scope)
    }

    /// Update a memory's `content` and/or `importance` in whichever tier holds it (`beam.py`
    /// `update`). FTS stays in sync via the content-update triggers. Returns whether a row changed.
    /// Fire-and-forget audit-log insert into the bank-co-located `audit_log`
    /// (`hermes_memory_provider/audit.py` `record` L69-L106). Uses the already-held connection (the
    /// audit table lives in the same bank DB) and swallows any error — auditing must never break a
    /// memory mutation. `timestamp` is unix epoch seconds (Python `time.time()`).
    pub(crate) fn audit(
        &self,
        conn: &Connection,
        action: &str,
        memory_id: Option<&str>,
        reason: Option<&str>,
    ) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let none = Option::<String>::None;
        let res = conn.execute(
            "INSERT INTO audit_log \
             (timestamp, action, memory_id, bank, scope, profile, session_id, source_tool, \
              tokens_used, reason, metadata_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                ts,
                action,
                memory_id,
                self.config.bank,
                none,
                none,
                self.config.session_id,
                none,
                Option::<i64>::None,
                reason,
                none,
            ],
        );
        if let Err(e) = res {
            tracing::debug!(error = %e, action, "audit log insert failed (non-fatal)");
        }
    }

    pub fn update(&self, id: &str, content: Option<&str>, importance: Option<f64>) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        let mut changed = false;
        for table in ["working_memory", "episodic_memory"] {
            if let Some(c) = content {
                changed |= conn.execute(
                    &format!("UPDATE {table} SET content = ?2 WHERE id = ?1"),
                    params![id, c],
                )? > 0;
            }
            if let Some(imp) = importance {
                changed |= conn.execute(
                    &format!("UPDATE {table} SET importance = ?2 WHERE id = ?1"),
                    params![id, imp],
                )? > 0;
            }
        }
        if changed {
            self.audit(&conn, "update", Some(id), None);
        }
        Ok(changed)
    }

    /// Hard-delete a memory from both tiers plus its stored embedding (`beam.py` `forget`). FTS rows
    /// are removed by the delete triggers. Returns whether anything was deleted.
    pub fn forget(&self, id: &str) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        let mut deleted = conn.execute("DELETE FROM working_memory WHERE id = ?1", params![id])?;
        deleted += conn.execute("DELETE FROM episodic_memory WHERE id = ?1", params![id])?;
        conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
            params![id],
        )?;
        if deleted > 0 {
            self.audit(&conn, "forget", Some(id), None);
        }
        Ok(deleted > 0)
    }

    /// Soft-invalidate a memory: stamp `valid_until` now and point `superseded_by` at an optional
    /// replacement (`beam.py` `invalidate` L7725). The row drops out of recall (which filters
    /// `valid_until IS NULL AND superseded_by IS NULL`). Returns whether a row changed.
    pub fn invalidate(&self, id: &str, replacement_id: Option<&str>) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();
        let mut changed = false;
        for table in ["working_memory", "episodic_memory"] {
            changed |= conn.execute(
                &format!(
                    "UPDATE {table} SET valid_until = ?2, superseded_by = ?3 \
                     WHERE id = ?1 AND valid_until IS NULL"
                ),
                params![id, now, replacement_id],
            )? > 0;
        }
        if changed {
            let reason = replacement_id.map(|r| format!("superseded_by={r}"));
            self.audit(&conn, "invalidate", Some(id), reason.as_deref());
        }
        Ok(changed)
    }

    /// Record a human/agent validation action on a memory (`beam.py` `validate`). Appends a
    /// `memory_validations` row and bumps the row's `validation_count`/`validated_at`/`validator`.
    /// `action = "correct"` with `new_content` rewrites the content; `action = "reject"` invalidates
    /// the row. Returns whether the target memory exists.
    pub fn validate(&self, v: &ValidateArgs) -> Result<bool> {
        let ValidateArgs {
            id,
            action,
            validator,
            new_content,
            note,
        } = *v;
        let now = util::now_iso();
        {
            let conn = self.store.conn.lock().unwrap();
            let exists: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM working_memory WHERE id = ?1 \
                 UNION ALL SELECT 1 FROM episodic_memory WHERE id = ?1)",
                params![id],
                |r| r.get(0),
            )?;
            if !exists {
                return Ok(false);
            }
            conn.execute(
                "INSERT INTO memory_validations (memory_id, validator, validated_at, action, \
                 new_content, note) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, validator, now, action, new_content, note],
            )?;
            for table in ["working_memory", "episodic_memory"] {
                conn.execute(
                    &format!(
                        "UPDATE {table} SET validation_count = validation_count + 1, \
                         validated_at = ?2, validator = ?3 WHERE id = ?1"
                    ),
                    params![id, now, validator],
                )?;
            }
            self.audit(&conn, "validate", Some(id), Some(action));
        }
        match action {
            "correct" => {
                if let Some(c) = new_content {
                    self.update(id, Some(c), None)?;
                }
            }
            "reject" => {
                self.invalidate(id, None)?;
            }
            _ => {}
        }
        Ok(true)
    }

    /// Bank statistics (`beam.py` `stats`): tier counts + structured-store sizes.
    pub fn stats(&self) -> Result<Stats> {
        let conn = self.store.conn.lock().unwrap();
        let count = |sql: &str| -> Result<i64> { Ok(conn.query_row(sql, [], |r| r.get(0))?) };
        Ok(Stats {
            working: count(
                "SELECT COUNT(*) FROM working_memory WHERE valid_until IS NULL AND superseded_by IS NULL",
            )?,
            episodic: count(
                "SELECT COUNT(*) FROM episodic_memory WHERE valid_until IS NULL AND superseded_by IS NULL",
            )?,
            episodic_tier1: count("SELECT COUNT(*) FROM episodic_memory WHERE tier = 1")?,
            episodic_tier2: count("SELECT COUNT(*) FROM episodic_memory WHERE tier = 2")?,
            episodic_tier3: count("SELECT COUNT(*) FROM episodic_memory WHERE tier = 3")?,
            facts: count("SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NULL")?,
            triples: count("SELECT COUNT(*) FROM triples WHERE valid_until IS NULL")?,
            conflicts: count("SELECT COUNT(*) FROM conflicts")?,
        })
    }

    /// A lightweight diagnostics summary (`beam.py` `health`).
    pub fn diagnose(&self) -> Result<Diagnostics> {
        let conn = self.store.conn.lock().unwrap();
        Ok(Diagnostics {
            pending_consolidation: conn.query_row(
                "SELECT COUNT(*) FROM working_memory WHERE consolidated_at IS NULL \
                 AND session_id = ?1 AND superseded_by IS NULL",
                params![self.config.session_id],
                |r| r.get(0),
            )?,
            embedded_episodic: conn.query_row(
                "SELECT COUNT(*) FROM episodic_memory WHERE binary_vector IS NOT NULL",
                [],
                |r| r.get(0),
            )?,
            episodic: conn.query_row("SELECT COUNT(*) FROM episodic_memory", [], |r| r.get(0))?,
            last_consolidation: conn
                .query_row(
                    "SELECT MAX(created_at) FROM consolidation_log WHERE items_consolidated > 0",
                    [],
                    |r| r.get::<_, Option<String>>(0),
                )
                .unwrap_or(None),
            open_conflicts: conn.query_row(
                "SELECT COUNT(*) FROM conflicts WHERE resolution IS NULL",
                [],
                |r| r.get(0),
            )?,
        })
    }

    /// Write a scratchpad note for the session (`beam.py` scratchpad). Returns the row id.
    pub fn scratchpad_write(&self, content: &str) -> Result<String> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();
        let id = util::memory_id(&format!(
            "scratch:{}:{}:{}",
            self.config.session_id, now, content
        ));
        conn.execute(
            "INSERT OR REPLACE INTO scratchpad (id, content, session_id) VALUES (?1, ?2, ?3)",
            params![id, content, self.config.session_id],
        )?;
        Ok(id)
    }

    /// Read the session's scratchpad notes, newest first (`(id, content)` pairs).
    pub fn scratchpad_read(&self) -> Result<Vec<(String, String)>> {
        let conn = self.store.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, content FROM scratchpad WHERE session_id = ?1 ORDER BY created_at DESC, id DESC",
        )?;
        let rows = stmt.query_map(params![self.config.session_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Clear the session's scratchpad. Returns the number of notes removed.
    pub fn scratchpad_clear(&self) -> Result<usize> {
        let conn = self.store.conn.lock().unwrap();
        Ok(conn.execute(
            "DELETE FROM scratchpad WHERE session_id = ?1",
            params![self.config.session_id],
        )?)
    }

    /// Export the session's working + episodic rows as a portable JSON bundle (`beam.py`
    /// `export`/sync surface). Knowledge structures are re-derivable from content on import.
    pub fn export(&self) -> Result<serde_json::Value> {
        let conn = self.store.conn.lock().unwrap();
        let dump = |table: &str| -> Result<Vec<serde_json::Value>> {
            let mut stmt = conn.prepare(&format!(
                "SELECT id, content, source, timestamp, importance, veracity, scope \
                 FROM {table} WHERE session_id = ?1 AND valid_until IS NULL AND superseded_by IS NULL"
            ))?;
            let rows = stmt.query_map(params![self.config.session_id], |r| {
                Ok(json!({
                    "id": r.get::<_, String>(0)?,
                    "content": r.get::<_, String>(1)?,
                    "source": r.get::<_, Option<String>>(2)?,
                    "timestamp": r.get::<_, Option<String>>(3)?,
                    "importance": r.get::<_, f64>(4)?,
                    "veracity": r.get::<_, Option<String>>(5)?,
                    "scope": r.get::<_, Option<String>>(6)?,
                }))
            })?;
            Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
        };
        Ok(json!({
            "version": 1,
            "session_id": self.config.session_id,
            "working_memory": dump("working_memory")?,
            "episodic_memory": dump("episodic_memory")?,
        }))
    }

    /// Import rows from an [`Engine::export`] bundle into this session, re-running knowledge + temporal
    /// ingestion for working rows. Returns the number of working rows imported.
    pub fn import(&self, bundle: &serde_json::Value) -> Result<usize> {
        let mut imported = 0usize;
        if let Some(rows) = bundle.get("working_memory").and_then(|v| v.as_array()) {
            for row in rows {
                let content = row.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if content.is_empty() {
                    continue;
                }
                let importance = row
                    .get("importance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5);
                let scope = row
                    .get("scope")
                    .and_then(|v| v.as_str())
                    .unwrap_or("session")
                    .to_string();
                let veracity = row
                    .get("veracity")
                    .and_then(|v| v.as_str())
                    .unwrap_or("imported")
                    .to_string();
                self.remember_with_vector(
                    content,
                    &RememberArgs {
                        source: "import".to_string(),
                        importance,
                        scope,
                        veracity,
                        ..Default::default()
                    },
                    None,
                    "",
                )?;
                imported += 1;
            }
        }
        Ok(imported)
    }
}
