//! The BEAM engine facade — port of `beam.py` `BeamMemory` (the `remember`/`recall`/`get_context`/
//! `sleep` surface, L2836 / L5027 / L3526 / L7576) plus the `memory.py` facade.
//!
//! Scaffold: `remember`, `get_context`, and a linear-hybrid `recall` (lexical + importance + recency
//! via [`crate::recall::scoring`]) are wired end-to-end over the SQLite store so the provider has a
//! working default. The vector path, knowledge ingestion, and `sleep`/consolidation are TODO and
//! reference the spec sections they implement.

use crate::config::MnemosyneConfig;
use crate::dynamics::typed_memory;
use crate::error::Result;
use crate::recall::scoring;
use crate::store::Store;
use crate::{sanitize, util};
use rusqlite::params;

/// Which BEAM tier a row lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Hot, recent, auto-injected context.
    Working,
    /// Long-term consolidated memory.
    Episodic,
}

/// A recalled / stored memory row (the `recall` result shape, `beam.py` L5996+).
#[derive(Clone, Debug)]
pub struct MemoryRow {
    /// Memory id.
    pub id: String,
    /// Content text.
    pub content: String,
    /// Ingestion source.
    pub source: String,
    /// ISO timestamp.
    pub timestamp: String,
    /// Importance `[0, 1]`.
    pub importance: f64,
    /// Trust label (`stated`/`inferred`/`tool`/`imported`/`unknown`).
    pub veracity: String,
    /// Trust tier (`STATED`/`DERIVED`/...).
    pub trust_tier: String,
    /// Which tier the row came from.
    pub tier: Tier,
    /// The recall score (0 for direct fetches).
    pub score: f64,
}

/// Arguments for [`Engine::remember`] (`beam.py` `remember` L2836).
#[derive(Clone, Debug)]
pub struct RememberArgs {
    /// Ingestion source (default `conversation`).
    pub source: String,
    /// Importance `[0, 1]` (default 0.5).
    pub importance: f64,
    /// Scope: `session` (default) or `global`. Note: the column default is `global` but
    /// `remember()` defaults to `session` (`beam.py` L2838).
    pub scope: String,
    /// Trust label (default `unknown`).
    pub veracity: String,
}

impl Default for RememberArgs {
    fn default() -> Self {
        Self {
            source: "conversation".to_string(),
            importance: 0.5,
            scope: "session".to_string(),
            veracity: "unknown".to_string(),
        }
    }
}

/// The BEAM engine over a single bank store.
pub struct Engine {
    store: Store,
    config: MnemosyneConfig,
}

impl Engine {
    /// Open the engine for the configured bank.
    pub fn open(config: MnemosyneConfig) -> Result<Self> {
        let store = Store::open(config.bank_db_path())?;
        Ok(Self { store, config })
    }

    /// Open an ephemeral in-memory engine (tests).
    pub fn open_in_memory(config: MnemosyneConfig) -> Result<Self> {
        let store = Store::open_in_memory()?;
        Ok(Self { store, config })
    }

    /// The active session id.
    pub fn session_id(&self) -> &str {
        &self.config.session_id
    }

    /// Store a memory in the working tier (`beam.py` `remember` L2836). Scaffold: sanitize +
    /// classify + insert (dedup, embedding, and knowledge ingestion are TODO).
    pub fn remember(&self, content: &str, args: &RememberArgs) -> Result<String> {
        let (content, _meta) = sanitize::sanitize_content(content);
        let id = util::memory_id(&format!("{}:{}", self.config.session_id, content));
        let memory_type = typed_memory::classify(&content).as_str();
        let now = util::now_iso();
        let conn = self.store.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO working_memory \
             (id, content, source, timestamp, session_id, importance, metadata_json, veracity, \
              memory_type, scope) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, '{}', ?7, ?8, ?9)",
            params![
                id,
                content,
                args.source,
                now,
                self.config.session_id,
                args.importance,
                args.veracity,
                memory_type,
                args.scope,
            ],
        )?;
        Ok(id)
    }

    /// Auto-inject context: global then session-local working memory ordered by importance/recency
    /// (`beam.py` `get_context` L3526-L3606).
    pub fn get_context(&self, limit: usize) -> Result<Vec<MemoryRow>> {
        let conn = self.store.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, content, source, timestamp, importance, veracity, trust_tier \
             FROM working_memory \
             WHERE (valid_until IS NULL) AND superseded_by IS NULL \
               AND (session_id = ?1 OR scope = 'global') \
             ORDER BY importance DESC, timestamp DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![self.config.session_id, limit as i64], |r| {
                Ok(MemoryRow {
                    id: r.get(0)?,
                    content: r.get(1)?,
                    source: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    timestamp: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    importance: r.get(4)?,
                    veracity: r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    trust_tier: r.get::<_, Option<String>>(6)?.unwrap_or_default(),
                    tier: Tier::Working,
                    score: 0.0,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Linear-hybrid recall over working memory (`beam.py` `recall` L5027). Scaffold: candidate rows
    /// are fetched by scope/validity, scored with the working-memory formula
    /// ([`scoring::working_memory_score`]) using in-Rust lexical relevance, then ranked. The vector
    /// path, episodic tier, and bonuses are TODO.
    pub fn recall(&self, query: &str, top_k: usize) -> Result<Vec<MemoryRow>> {
        let candidates = self.get_context(2000)?;
        let q_tokens: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();
        let floor = scoring::lexical_floor(q_tokens.len());
        let (_vw, _fw, iw) = scoring::DEFAULT_WEIGHTS;

        let mut scored: Vec<MemoryRow> = Vec::new();
        for mut row in candidates {
            let relevance = lexical_relevance(&q_tokens, &row.content);
            if relevance < floor {
                continue;
            }
            let decay = scoring::recency_decay(age_hours(&row.timestamp));
            let base = scoring::working_memory_score(relevance, row.importance, iw, 0.0, decay);
            row.score = base * scoring::veracity_multiplier(&row.veracity);
            scored.push(row);
        }
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

/// Simple token-overlap lexical relevance (a subset of `beam.py` L1573-L1638). Full-query substring
/// scores 1.0; otherwise the fraction of query tokens present.
fn lexical_relevance(query_tokens: &[String], content: &str) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let lc = content.to_lowercase();
    if lc.contains(&query_tokens.join(" ")) {
        return 1.0;
    }
    let hits = query_tokens
        .iter()
        .filter(|t| lc.contains(t.as_str()))
        .count();
    hits as f64 / query_tokens.len() as f64
}

/// Hours since an ISO timestamp (`None` if unparseable -> decay falls back to 0.5).
fn age_hours(timestamp: &str) -> Option<f64> {
    let parsed = chrono::DateTime::parse_from_rfc3339(timestamp).ok()?;
    let now = chrono::Utc::now();
    let delta = now.signed_duration_since(parsed.with_timezone(&chrono::Utc));
    Some(delta.num_seconds().max(0) as f64 / 3600.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        Engine::open_in_memory(MnemosyneConfig::default()).expect("engine")
    }

    #[test]
    fn remember_then_recall() {
        let e = engine();
        e.remember(
            "the authentication flow uses JWT tokens",
            &RememberArgs::default(),
        )
        .unwrap();
        e.remember("lunch was pizza", &RememberArgs::default())
            .unwrap();
        let hits = e.recall("authentication flow", 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].content.contains("authentication"));
    }

    #[test]
    fn get_context_orders_by_importance() {
        let e = engine();
        e.remember(
            "low",
            &RememberArgs {
                importance: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        e.remember(
            "high",
            &RememberArgs {
                importance: 0.9,
                ..Default::default()
            },
        )
        .unwrap();
        let ctx = e.get_context(10).unwrap();
        assert_eq!(ctx[0].content, "high");
    }
}
