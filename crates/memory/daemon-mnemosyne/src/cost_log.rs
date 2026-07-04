// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Cost logger — port of `mnemosyne/core/cost_log.py`.
//!
//! Tracks LLM memory-operation costs over time for benchmarking, in a dedicated `cost_log.db`
//! *separate* from the memory bank (Python: `~/.mnemosyne/data/cost_log.db`; Rust: the caller
//! passes the directory — the engine uses its bank's `data_dir`). The only production writer is
//! the tier-2 LLM conflict validator (`llm_conflict_detector.py` L198-L209), which logs one row
//! per validated pair; writes are fire-and-forget there (a cost-log failure never fails sleep).
//!
//! Cost estimation mirrors Python's fallback path: ~4 chars/token (`_estimate_tokens` L43) and
//! the `MODEL_PRICING["default"]` tier ($0.15/$0.60 per 1M tokens, L35-L40) — the injected
//! daemon-core [`Provider`](daemon_core::Provider) exposes no per-model pricing, so every call
//! prices at the default tier under the model label the caller passes.

use crate::Result;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};

/// Default-tier pricing, USD per 1M tokens (`MODEL_PRICING["default"]` L39).
const DEFAULT_INPUT_PRICE: f64 = 0.15;
const DEFAULT_OUTPUT_PRICE: f64 = 0.60;

/// Rough token estimation, ~4 chars per token (`_estimate_tokens` L43-L45).
pub fn estimate_tokens(text: &str) -> i64 {
    (text.len() as i64 / 4).max(1)
}

/// Estimate a call cost in USD at the default pricing tier (`_calculate_cost` L48-L53 with the
/// `default` entry — the Rust node has no per-model pricing catalog).
pub fn calculate_cost(input_tokens: i64, output_tokens: i64) -> f64 {
    (input_tokens as f64 / 1_000_000.0) * DEFAULT_INPUT_PRICE
        + (output_tokens as f64 / 1_000_000.0) * DEFAULT_OUTPUT_PRICE
}

/// One aggregate over `cost_entries` (`get_cost_stats` L54-L78).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct CostStats {
    /// Number of logged calls.
    pub total_calls: i64,
    /// Sum of `memory_count`.
    pub total_memories_injected: i64,
    /// Sum of `token_count`.
    pub total_tokens: i64,
    /// Sum of `estimated_cost_usd`, rounded to 6 places.
    pub total_estimated_cost_usd: f64,
}

/// The `cost_log.db` path inside a data directory.
pub fn cost_log_path(dir: &Path) -> PathBuf {
    dir.join("cost_log.db")
}

fn open(dir: &Path) -> Result<Connection> {
    std::fs::create_dir_all(dir)?;
    let conn = Connection::open(cost_log_path(dir))?;
    // `init_cost_log` (L24-L38): idempotent create.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS cost_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            memory_count INTEGER,
            token_count INTEGER,
            estimated_cost_usd REAL,
            model TEXT DEFAULT 'default',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );",
    )?;
    Ok(conn)
}

/// Append one cost entry (`log_cost` L41-L51). Creates the DB/table on first use.
pub fn log_cost(
    dir: &Path,
    session_id: &str,
    memory_count: i64,
    token_count: i64,
    estimated_cost_usd: f64,
    model: &str,
) -> Result<()> {
    let conn = open(dir)?;
    conn.execute(
        "INSERT INTO cost_entries \
         (session_id, memory_count, token_count, estimated_cost_usd, model, timestamp) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            session_id,
            memory_count,
            token_count,
            estimated_cost_usd,
            model,
            crate::util::now_iso(),
        ],
    )?;
    Ok(())
}

/// Aggregate cost stats, optionally scoped to one session (`get_cost_stats` L54-L78).
pub fn get_cost_stats(dir: &Path, session_id: Option<&str>) -> Result<CostStats> {
    let conn = open(dir)?;
    let map = |r: &rusqlite::Row<'_>| -> rusqlite::Result<CostStats> {
        Ok(CostStats {
            total_calls: r.get::<_, Option<i64>>(0)?.unwrap_or(0),
            total_memories_injected: r.get::<_, Option<i64>>(1)?.unwrap_or(0),
            total_tokens: r.get::<_, Option<i64>>(2)?.unwrap_or(0),
            total_estimated_cost_usd: (r.get::<_, Option<f64>>(3)?.unwrap_or(0.0) * 1e6).round()
                / 1e6,
        })
    };
    let stats = match session_id {
        Some(sid) => conn.query_row(
            "SELECT COUNT(*), SUM(memory_count), SUM(token_count), SUM(estimated_cost_usd) \
             FROM cost_entries WHERE session_id = ?1",
            params![sid],
            map,
        )?,
        None => conn.query_row(
            "SELECT COUNT(*), SUM(memory_count), SUM(token_count), SUM(estimated_cost_usd) \
             FROM cost_entries",
            [],
            map,
        )?,
    };
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_and_aggregate_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        log_cost(dir.path(), "s1", 2, 100, 0.000123, "host").unwrap();
        log_cost(dir.path(), "s1", 2, 200, 0.000200, "host").unwrap();
        log_cost(dir.path(), "s2", 1, 50, 0.000050, "host").unwrap();

        let s1 = get_cost_stats(dir.path(), Some("s1")).unwrap();
        assert_eq!(s1.total_calls, 2);
        assert_eq!(s1.total_memories_injected, 4);
        assert_eq!(s1.total_tokens, 300);
        assert!((s1.total_estimated_cost_usd - 0.000323).abs() < 1e-9);

        let all = get_cost_stats(dir.path(), None).unwrap();
        assert_eq!(all.total_calls, 3);
        assert_eq!(all.total_tokens, 350);
    }

    #[test]
    fn empty_db_reports_zeroes() {
        let dir = tempfile::tempdir().unwrap();
        let stats = get_cost_stats(dir.path(), None).unwrap();
        assert_eq!(stats, CostStats::default());
    }

    #[test]
    fn token_estimate_and_default_pricing() {
        assert_eq!(
            estimate_tokens(""),
            1,
            "floor of 1 like Python's max(1, ...)"
        );
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        // 1M input + 1M output at the default tier = $0.75.
        assert!((calculate_cost(1_000_000, 1_000_000) - 0.75).abs() < 1e-12);
    }
}
