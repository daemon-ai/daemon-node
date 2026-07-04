// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tool-surface backing methods for the BEAM [`Engine`]: get/update/forget/invalidate/validate,
//! the audit-log writer, stats/diagnostics, scratchpad, and export/import. Split out of
//! `engine.rs` (W-MNEMO).

use super::*;
use crate::util;
use rusqlite::{params, Connection};
use serde_json::json;

impl Engine {
    // ── Tool-surface backing methods (`beam.py` get/update/forget/invalidate/validate/stats/...) ──

    /// Fetch a single memory by id — a pure read with no recall bump and no validity filter
    /// (`beam.py` `get` L3855-L3911): the session-scoped working row first, then the
    /// session-or-global episodic row.
    pub fn get(&self, id: &str) -> Result<Option<MemoryRow>> {
        let conn = self.store.conn.lock().unwrap();
        let map = |tier: Tier| {
            move |r: &rusqlite::Row<'_>| -> rusqlite::Result<MemoryRow> {
                Ok(MemoryRow {
                    id: r.get(0)?,
                    content: r.get(1)?,
                    source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    importance: r.get::<_, Option<f64>>(4)?.unwrap_or(0.5),
                    veracity: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    scope: r.get::<_, Option<String>>(6)?.unwrap_or_default(),
                    tier,
                    tier_level: 1,
                    ..Default::default()
                })
            }
        };
        let working = conn
            .query_row(
                "SELECT id, content, source, timestamp, importance, veracity, scope \
                 FROM working_memory WHERE id = ?1 AND session_id = ?2",
                params![id, self.config.session_id],
                map(Tier::Working),
            )
            .ok();
        if working.is_some() {
            return Ok(working);
        }
        Ok(conn
            .query_row(
                "SELECT id, content, source, timestamp, importance, veracity, scope \
                 FROM episodic_memory WHERE id = ?1 AND (session_id = ?2 OR scope = 'global')",
                params![id, self.config.session_id],
                map(Tier::Episodic),
            )
            .ok())
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

    /// Tool-level audit event with explicit `bank`/`source_tool`/`metadata` stamps
    /// (`hermes_memory_provider/__init__.py` `_audit_event`). Like [`Engine::audit`] the row
    /// lands in THIS engine's bank-co-located `audit_log` — shared-surface tool events audit
    /// into the private provider DB with `bank="surface"`, matching Python's provider-side log.
    /// Fire-and-forget: locks the connection itself and swallows errors.
    pub(crate) fn audit_tool(
        &self,
        action: &str,
        memory_id: Option<&str>,
        bank: &str,
        source_tool: &str,
        metadata: Option<&serde_json::Value>,
    ) {
        let conn = self.store.conn.lock().unwrap();
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
                bank,
                none,
                none,
                self.config.session_id,
                source_tool,
                Option::<i64>::None,
                none,
                metadata.map(|m| m.to_string()),
            ],
        );
        if let Err(e) = res {
            tracing::debug!(error = %e, action, "tool audit log insert failed (non-fatal)");
        }
    }

    /// Test-only: the audit_log actions recorded against a bank label, insertion-ordered.
    #[cfg(test)]
    pub(crate) fn audit_rows_for_test(&self, bank: &str) -> Result<Vec<String>> {
        let conn = self.store.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT action FROM audit_log WHERE bank = ?1 ORDER BY rowid")?;
        let rows = stmt.query_map(params![bank], |r| r.get(0))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Update a session-scoped working-memory row's `content` and/or `importance` (`beam.py`
    /// `update_working` L3809-L3853). FTS stays in sync via the `wm_au` trigger; on a content
    /// change the stale dense embedding is dropped so vector recall can't serve outdated state
    /// (Python re-embeds inline; the sync engine defers that to the async provider/tool layer).
    /// Returns whether a row changed.
    pub fn update(&self, id: &str, content: Option<&str>, importance: Option<f64>) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        let mut sets: Vec<&str> = Vec::new();
        let mut bind: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(c) = content {
            sets.push("content = ?");
            bind.push(rusqlite::types::Value::Text(c.to_string()));
        }
        if let Some(imp) = importance {
            sets.push("importance = ?");
            bind.push(rusqlite::types::Value::Real(imp));
        }
        if sets.is_empty() {
            return Ok(false);
        }
        bind.push(rusqlite::types::Value::Text(id.to_string()));
        bind.push(rusqlite::types::Value::Text(self.config.session_id.clone()));
        let affected = conn.execute(
            &format!(
                "UPDATE working_memory SET {} WHERE id = ? AND session_id = ?",
                sets.join(", ")
            ),
            rusqlite::params_from_iter(bind),
        )?;
        if content.is_some() && affected > 0 {
            conn.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                params![id],
            )?;
        }
        if affected > 0 {
            self.audit(&conn, "update", Some(id), None);
        }
        Ok(affected > 0)
    }

    /// Hard-delete a working-memory row plus its derived state (`beam.py` `forget_working`
    /// L3913-L3958): the session-or-global-scoped delete is the authorization boundary for the
    /// annotation/embedding cascade (E6.a). FTS rows are removed by the delete trigger. Returns
    /// whether anything was deleted.
    pub fn forget(&self, id: &str) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        let deleted = conn.execute(
            "DELETE FROM working_memory WHERE id = ?1 AND (session_id = ?2 OR scope = 'global')",
            params![id, self.config.session_id],
        )?;
        if deleted > 0 {
            conn.execute("DELETE FROM annotations WHERE memory_id = ?1", params![id])?;
            conn.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                params![id],
            )?;
            self.audit(&conn, "forget", Some(id), None);
        }
        Ok(deleted > 0)
    }

    /// Soft-invalidate a memory: stamp `valid_until` now and point `superseded_by` at an optional
    /// replacement (`beam.py` `invalidate` L3610-L3632), session-or-global scoped, working tier
    /// first. The row drops out of recall (which filters validity). Returns whether a row changed.
    pub fn invalidate(&self, id: &str, replacement_id: Option<&str>) -> Result<bool> {
        let conn = self.store.conn.lock().unwrap();
        let now = util::now_iso();
        let mut changed = false;
        for table in ["working_memory", "episodic_memory"] {
            changed = conn.execute(
                &format!(
                    "UPDATE {table} SET valid_until = ?2, superseded_by = ?3 \
                     WHERE id = ?1 AND (session_id = ?4 OR scope = 'global')"
                ),
                params![id, now, replacement_id, self.config.session_id],
            )? > 0;
            if changed {
                break;
            }
        }
        if changed {
            let reason = replacement_id.map(|r| format!("superseded_by={r}"));
            self.audit(&conn, "invalidate", Some(id), reason.as_deref());
            if let Some(pm) = self.plugins_if_active() {
                pm.notify_invalidate(id);
            }
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

    /// All identity memories for the active session, importance-then-recency ordered — the
    /// always-inject prefetch rows (`__init__.py` `_identity_fichas` L1547-L1582). Query-independent
    /// and strictly session-scoped so there is zero cross-session leakage.
    pub fn identity_rows(&self) -> Result<Vec<crate::prefetch::IdentityRow>> {
        let conn = self.store.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT content, importance, timestamp FROM working_memory \
             WHERE source = 'identity' AND session_id = ?1 \
             ORDER BY importance DESC, timestamp DESC",
        )?;
        let rows = stmt.query_map(params![self.config.session_id], |r| {
            Ok(crate::prefetch::IdentityRow {
                content: r.get::<_, Option<String>>(0)?.unwrap_or_default(),
                importance: r.get::<_, Option<f64>>(1)?.unwrap_or(0.95),
                timestamp: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            })
        })?;
        Ok(rows.flatten().filter(|r| !r.content.is_empty()).collect())
    }

    /// Count working rows a non-forced sleep pass would claim right now (`beam.py`
    /// `_count_unconsolidated_before`; the auto-sleep eligibility gate, `__init__.py` L1735-L1738).
    /// Matches [`Engine::sleep_plan`]'s candidate WHERE exactly.
    pub fn eligible_for_sleep(&self) -> Result<i64> {
        let half_ttl_minutes = (self.config.working_memory_ttl_hours * 30.0) as i64;
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::minutes(half_ttl_minutes)).to_rfc3339();
        let conn = self.store.conn.lock().unwrap();
        Ok(conn.query_row(
            "SELECT COUNT(*) FROM working_memory \
             WHERE COALESCE(session_id, 'default') = ?1 \
               AND timestamp < ?2 \
               AND consolidated_at IS NULL \
               AND (pinned IS NULL OR pinned = 0) \
               AND superseded_by IS NULL",
            params![self.config.session_id, cutoff],
            |r| r.get(0),
        )?)
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
