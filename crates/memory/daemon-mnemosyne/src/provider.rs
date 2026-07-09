// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `MemoryProvider` implementation — port of `hermes_memory_provider/__init__.py`.
//!
//! Maps Mnemosyne onto the daemon-core seam: `prompt_block` = memory-override instructions
//! (`system_prompt_block` L1437), `recall` = the hardened prefetch pipeline
//! ([`crate::prefetch`], `prefetch` L1474 / `_prefetch_bank` L1584 / `_prefetch_identity` L1514),
//! `after_turn` = the `sync_turn` persist gates + identity-signal capture + gated auto-sleep
//! (L1668-L1748), and `tools`/`call_tool` = the JSON tool dispatch (L1750) including the
//! shared-surface bank (`_handle_shared_*` L1973-L2054).

use crate::embeddings::Embedder;
use crate::engine::{Engine, RememberArgs};
use crate::extract::Extractor;
use crate::{prefetch, MnemosyneConfig};
use daemon_core::command::{
    CommandCx, CommandError, CommandInvocation, CommandOutput, CommandProvider,
    CommandProviderHandle, CommandSpec,
};
use daemon_core::conversation::{Conversation, Turn};
use daemon_core::memory::{MemoryProvider, PromptBlock, RecallQuery, RecalledBlock, SwitchReason};
use daemon_core::tools::ToolDef;
use daemon_core::{EmbeddingProvider, Provider};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// Auto-sleep cadence: check the consolidation gate every N persisted turns (`__init__.py`
/// `sync_turn` L1689: `turn_count % 10 == 0`).
const AUTO_SLEEP_EVERY_TURNS: u64 = 10;

/// Identity-significant expressions the user may voice about themselves or their relationship to
/// their work (`__init__.py` `_IDENTITY_SIGNALS` L1698-L1709). A match persists the turn again as
/// a high-importance global `[IDENTITY]` memory.
const IDENTITY_SIGNALS: &[&str] = &[
    "feeling like",
    "imposter",
    "impostor",
    "barely know",
    "don't know my own",
    "don't even know how",
    "want them to feel",
    "i'm proud",
    "i feel like a",
    "i don't know how to",
];

/// The Mnemosyne memory provider over a single bank engine, with optional embedding + LLM backends.
pub struct MnemosyneProvider {
    engine: Arc<Engine>,
    embedder: Embedder,
    extractor: Extractor,
    turns: AtomicU64,
    /// The lazily-opened shared-surface bank (`__init__.py` `_ensure_surface_beam` L1954): a
    /// separate cross-profile DB at `<shared_surface_dir>/mnemosyne.db`, session
    /// `hermes_shared_surface`. `Some(None)` = init failed (reported per tool call).
    surface: OnceLock<Option<Arc<Engine>>>,
}

impl MnemosyneProvider {
    /// Wrap an existing engine in keyword-only mode (no embeddings, no LLM).
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            embedder: Embedder::new(),
            extractor: Extractor::new(),
            turns: AtomicU64::new(0),
            surface: OnceLock::new(),
        }
    }

    /// Wrap an existing engine with an injected embedding provider (hybrid lexical + vector recall).
    pub fn with_embedder(engine: Arc<Engine>, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            engine,
            embedder: Embedder::with_provider(embedder),
            extractor: Extractor::new(),
            turns: AtomicU64::new(0),
            surface: OnceLock::new(),
        }
    }

    /// Wrap an existing engine with optional embedding and LLM backends.
    pub fn with_backends(
        engine: Arc<Engine>,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
        llm: Option<Arc<dyn Provider>>,
    ) -> Self {
        Self {
            engine,
            embedder: embedder.map(Embedder::with_provider).unwrap_or_default(),
            extractor: llm.map(Extractor::with_provider).unwrap_or_default(),
            turns: AtomicU64::new(0),
            surface: OnceLock::new(),
        }
    }

    /// Open a provider for the configured bank in keyword-only mode.
    pub fn open(config: MnemosyneConfig) -> crate::Result<Self> {
        Ok(Self::new(Arc::new(Engine::open(config)?)))
    }

    /// Open a provider for the configured bank with an injected embedding provider.
    pub fn open_with_embedder(
        config: MnemosyneConfig,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> crate::Result<Self> {
        Ok(Self::with_embedder(
            Arc::new(Engine::open(config)?),
            embedder,
        ))
    }

    /// Open a provider for the configured bank with optional embedding and LLM backends.
    pub fn open_with_backends(
        config: MnemosyneConfig,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
        llm: Option<Arc<dyn Provider>>,
    ) -> crate::Result<Self> {
        Ok(Self::with_backends(
            Arc::new(Engine::open(config)?),
            embedder,
            llm,
        ))
    }

    /// The lazily-opened shared-surface engine (`__init__.py` `_ensure_surface_beam` L1954-L1962).
    /// `None` when opening failed (each shared tool call reports it, never panics). An in-memory
    /// private bank gets an in-memory surface so the ephemeral default node touches no disk.
    pub(crate) fn surface_engine(&self) -> Option<&Arc<Engine>> {
        self.surface
            .get_or_init(|| {
                let cfg = self.engine.config();
                let surface_config = MnemosyneConfig {
                    data_dir: cfg.shared_surface_dir(),
                    bank: "default".to_string(),
                    session_id: "hermes_shared_surface".to_string(),
                    ..cfg.clone()
                };
                let opened = if self.engine.is_persistent() {
                    std::fs::create_dir_all(&surface_config.data_dir)
                        .map_err(crate::Error::from)
                        .and_then(|()| Engine::open(surface_config))
                } else {
                    Engine::open_in_memory(surface_config)
                };
                match opened {
                    Ok(e) => Some(Arc::new(e)),
                    Err(e) => {
                        tracing::warn!(error = %e, "mnemosyne shared surface init failed");
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Should this turn role be persisted (`sync_roles` config, `__init__.py` L1673/L1681)?
    fn syncs_role(&self, role: &str) -> bool {
        self.engine
            .config()
            .sync_roles
            .iter()
            .any(|r| r.eq_ignore_ascii_case(role))
    }

    /// Whether content matches an ignore pattern (`__init__.py` `_should_filter` L1215-L1226).
    /// Case-insensitive regex search; invalid patterns are skipped.
    fn should_filter(&self, content: &str) -> bool {
        self.engine.config().ignore_patterns.iter().any(|p| {
            regex::Regex::new(&format!("(?i){p}"))
                .map(|re| re.is_match(content))
                .unwrap_or(false)
        })
    }

    /// Persist a user turn that voices an identity-significant feeling as a durable global
    /// `[IDENTITY]` memory (`__init__.py` `_capture_identity_signals` L1711-L1723). One per turn.
    fn capture_identity_signals(&self, user_content: &str) {
        let lower = user_content.to_lowercase();
        if IDENTITY_SIGNALS.iter().any(|sig| lower.contains(sig)) {
            let _ = self.engine.remember(
                &format!(
                    "[IDENTITY] {}",
                    user_content.chars().take(400).collect::<String>()
                ),
                &RememberArgs {
                    source: "identity".to_string(),
                    importance: 0.85,
                    scope: "global".to_string(),
                    veracity: "stated".to_string(),
                    ..Default::default()
                },
            );
        }
    }
}

#[async_trait::async_trait]
impl MemoryProvider for MnemosyneProvider {
    fn name(&self) -> &str {
        "mnemosyne"
    }

    fn prompt_block(&self) -> Option<PromptBlock> {
        // The active-memory branch of `system_prompt_block` (`__init__.py` L1444-L1450); the
        // init-failed/skip-context branches are host concerns (the daemon simply doesn't register
        // the provider), and the hermes legacy-tool deprecation sentence has no daemon equivalent.
        Some(PromptBlock {
            text: "# Mnemosyne Memory\n\
                   Active (native local memory). Use mnemosyne_remember to store ANY durable \
                   fact, preference, identity, or insight. Use mnemosyne_recall to search. \
                   Use mnemosyne_shared_* tools for manual shared surface CRUD."
                .to_string(),
        })
    }

    async fn recall(&self, q: &RecallQuery) -> Option<RecalledBlock> {
        // The hardened prefetch pipeline (`__init__.py` `prefetch` L1474 / `_prefetch_bank`
        // L1584): over-fetch, filter junk/raw transcript, rank by adjusted score, semantic-dedup,
        // cap to the profile's top_k — then always-inject the session's identity rows up front.
        let cfg = self.engine.config();
        let profile = prefetch::resolve_profile(&cfg.prefetch_profile);
        let content_limit = if cfg.prefetch_content_chars > 0 {
            cfg.prefetch_content_chars
        } else {
            profile.content_char_limit
        };

        let query_vec = self.embedder.embed_query(&q.text).await;
        let overfetch = (profile.top_k * 2).max(prefetch::PREFETCH_OVERFETCH);
        let filters = crate::config::RecallFilters {
            temporal_weight: profile.temporal_weight,
            temporal_halflife: Some(profile.temporal_halflife),
            vec_weight: profile.vec_weight,
            fts_weight: profile.fts_weight,
            importance_weight: profile.importance_weight,
            ..Default::default()
        };
        let bank_block = self
            .engine
            .recall_with_scope(&crate::engine::RecallReq {
                query: &q.text,
                top_k: overfetch,
                query_vector: query_vec.as_deref(),
                scope: &self.engine.config_scope(),
                filters,
            })
            .ok()
            .map(|rows| prefetch::filter_and_rank(rows, &profile))
            .map(|rows| prefetch::render_bank_block(&rows, content_limit))
            .unwrap_or_default();

        // Per-contact identity memories surface on EVERY turn, independent of the recall query
        // (`_prefetch_identity` L1499-L1545), deduplicated against what recall already produced.
        let identity_block = self
            .engine
            .identity_rows()
            .map(|rows| prefetch::render_identity_block(&rows, &bank_block, content_limit))
            .unwrap_or_default();

        let text = [identity_block, bank_block]
            .into_iter()
            .filter(|b| !b.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if text.is_empty() {
            return None;
        }
        Some(RecalledBlock { text })
    }

    async fn after_turn(&self, turn: &Turn, _conv: &Conversation) {
        // sync_turn gates (`__init__.py` L1668-L1692): persist the user text (>5 chars, capped at
        // 500) and the assistant text (>10 chars, capped at 800) with their respective
        // importances, honoring the `sync_roles` and `ignore_patterns` config.
        let (content, importance, is_user) = match turn {
            Turn::User(u)
                if u.text.len() > 5 && self.syncs_role("user") && !self.should_filter(&u.text) =>
            {
                (
                    format!("[USER] {}", u.text.chars().take(500).collect::<String>()),
                    0.5,
                    true,
                )
            }
            Turn::Assistant(a)
                if a.text.len() > 10
                    && self.syncs_role("assistant")
                    && !self.should_filter(&a.text) =>
            {
                (
                    format!(
                        "[ASSISTANT] {}",
                        a.text.chars().take(800).collect::<String>()
                    ),
                    0.15,
                    false,
                )
            }
            _ => return,
        };
        // Embed once at this async seam; the precomputed vector is persisted with the row.
        let vector = self.embedder.embed_query(&content).await;
        let model = self.embedder.model().unwrap_or("");
        let memory_id = match self.engine.remember_with_vector(
            &content,
            &RememberArgs {
                importance,
                extract_entities: true,
                ..Default::default()
            },
            vector.as_deref(),
            model,
        ) {
            Ok(id) => id,
            Err(_) => return,
        };

        // Identity-signal capture is gated by user sync (`__init__.py` L1680).
        if is_user {
            if let Turn::User(u) = turn {
                self.capture_identity_signals(&u.text);
            }
        }

        // LLM extraction layered on top of the always-on regex baseline (`extraction.py`): extract
        // at this async seam, then merge into the knowledge layer synchronously.
        if self.extractor.available() {
            if let Some(extracted) = self.extractor.extract(&content).await {
                let _ = self.engine.ingest_extracted(&memory_id, &extracted);
            }
        }

        // Gated auto-sleep (`__init__.py` `sync_turn` L1688-L1690 + `_maybe_auto_sleep`
        // L1725-L1748): every 10 persisted turns, when enabled, working memory exceeds the
        // threshold, AND a non-forced pass would actually claim rows (the cheap eligibility check
        // that avoids spinning up a full pass just to find nothing).
        let turns = self.turns.fetch_add(1, Ordering::Relaxed) + 1;
        let cfg = self.engine.config();
        if cfg.auto_sleep_enabled && turns.is_multiple_of(AUTO_SLEEP_EVERY_TURNS) {
            let working = self.engine.stats().map(|s| s.working).unwrap_or(0);
            let eligible = self.engine.eligible_for_sleep().unwrap_or(0);
            if working > cfg.auto_sleep_threshold as i64 && eligible > 0 {
                self.run_sleep(false).await;
            }
        }
    }

    async fn before_compact(&self, _conv: &Conversation) {
        // Deliberate no-op: every turn is already persisted at `after_turn` time, so compaction
        // loses nothing this provider hasn't stored. (Python has no compact hook at all.)
    }

    async fn on_session_switch(&self, reason: SwitchReason) {
        // Run a full, forced sleep pass at session boundaries (`beam.py` sleep L7576): flush this
        // session's working memory into the episodic tiers regardless of age, then degrade.
        if matches!(reason, SwitchReason::End | SwitchReason::Handoff) {
            self.run_sleep(true).await;
        }
    }

    /// Expose this provider as a [`CommandProvider`] so the node command registry folds in `/memory`
    /// (the operator memory-maintenance surface).
    fn command_provider(self: Arc<Self>) -> Option<CommandProviderHandle> {
        Some(self)
    }
}

/// The `/memory` operator command surface: inspect (`stats`/`diagnose`), consolidate (`sleep`), and
/// `export` the bank. Read subcommands run on the per-session engine; `sleep` is a mutating
/// consolidation pass.
#[async_trait::async_trait]
impl CommandProvider for MnemosyneProvider {
    fn name(&self) -> &str {
        "mnemosyne"
    }

    fn commands(&self) -> Vec<CommandSpec> {
        command_specs()
    }

    async fn run_command(
        &self,
        invocation: &CommandInvocation,
        _cx: &CommandCx<'_>,
    ) -> std::result::Result<CommandOutput, CommandError> {
        let sub = invocation
            .subcommand()
            .unwrap_or("stats")
            .to_ascii_lowercase();
        match sub.as_str() {
            "" | "stats" => {
                let stats = self
                    .engine
                    .stats()
                    .map_err(|e| CommandError::Failed(e.to_string()))?;
                Ok(CommandOutput::text(pretty(&stats)))
            }
            "diagnose" => {
                let diag = self
                    .engine
                    .diagnose()
                    .map_err(|e| CommandError::Failed(e.to_string()))?;
                Ok(CommandOutput::text(pretty(&diag)))
            }
            "sleep" => {
                let force = matches!(invocation.rest().trim(), "force" | "--force" | "-f");
                let report = self
                    .engine
                    .sleep(force)
                    .map_err(|e| CommandError::Failed(e.to_string()))?;
                Ok(CommandOutput::text(pretty(&report)))
            }
            "export" => {
                let bundle = self
                    .engine
                    .export()
                    .map_err(|e| CommandError::Failed(e.to_string()))?;
                Ok(CommandOutput::text(
                    serde_json::to_string_pretty(&bundle).unwrap_or_else(|_| bundle.to_string()),
                ))
            }
            other => Err(CommandError::BadArgs(format!(
                "unknown /memory subcommand: {other} (try stats|diagnose|sleep|export)"
            ))),
        }
    }
}

/// The static `/memory` command catalog — the single source for the node command registry (the
/// binary's per-session wrapper advertises these without a live provider instance) and the
/// instance-level [`CommandProvider::commands`].
pub fn command_specs() -> Vec<CommandSpec> {
    vec![CommandSpec::new("memory")
        .summary("Mnemosyne memory: stats, diagnose, sleep, export")
        .category("Memory")
        .args_hint("<stats|diagnose|sleep|export>")
        .subcommands(["stats", "diagnose", "sleep", "export"])]
}

/// Pretty-print a serializable value for human command output.
fn pretty<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "<unserializable>".to_string())
}

impl MnemosyneProvider {
    /// Drive one sleep/consolidation pass (`beam.py` sleep L7576). When an LLM is present, each
    /// claimed source group is summarized at this async seam before the synchronous engine writes
    /// the episodic summary; otherwise the engine falls back to the deterministic AAAK summary.
    async fn run_sleep(&self, force: bool) {
        let _ = crate::tools::run_sleep(&self.engine, &self.embedder, &self.extractor, force).await;
        self.validate_conflicts().await;
    }

    /// Tier-2 LLM conflict validation (`llm_conflict_detector.py`), layered atop the deterministic
    /// `(subject, predicate)` contradictions recorded in `conflicts` during consolidation. Opt-in
    /// (`MNEMOSYNE_LLM_CONFLICT_DETECTION`) and only when an LLM backend is injected: each unresolved
    /// pair is validated, and a confirmed conflict marks the older fact superseded by the newer one.
    async fn validate_conflicts(&self) {
        if !self.engine.llm_conflict_detection() || !self.extractor.available() {
            return;
        }
        let pending = match self.engine.pending_conflicts() {
            Ok(p) => p,
            Err(_) => return,
        };
        for c in pending {
            if let Some(verdict) = crate::knowledge::conflict::validate_conflict_pair_logged(
                &self.extractor,
                &self.engine,
                &c.older_text,
                &c.newer_text,
            )
            .await
            {
                let _ = self.engine.resolve_conflict(
                    c.conflict_id,
                    verdict.is_conflict,
                    &c.newer_fact_id,
                    &c.older_fact_id,
                );
            }
        }
    }
}

impl MnemosyneProvider {
    /// The memory-management tools this backend exposes (`mnemosyne_remember`/`mnemosyne_recall`).
    ///
    /// These are *not* part of the §11 [`MemoryProvider`] seam — that seam is about context, not
    /// dispatch. A host that wants to expose them to the model registers them through the §12
    /// [`ToolRegistry`](daemon_core::tools) like any other tool, calling [`Self::call_tool`].
    pub fn tools(&self) -> Vec<ToolDef> {
        crate::tools::defs()
    }

    /// Dispatch one of [`Self::tools`] by name, returning a JSON string result.
    ///
    /// The shared-surface engine is materialized lazily on the first `shared_*` call (or
    /// `mnemosyne_recall` with surface read enabled); when init fails, `shared_*` tools report
    /// the error instead of silently writing to the private bank.
    pub async fn call_tool(&self, name: &str, args: Value) -> String {
        let cx = crate::tools::ToolCx {
            engine: &self.engine,
            embedder: &self.embedder,
            extractor: &self.extractor,
            surface: self.surface_engine().map(|e| &**e),
        };
        crate::tools::dispatch(&cx, name, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use daemon_core::conversation::{Conversation, SystemPrompt, Turn, UserMsg};
    use daemon_core::memory::RecallQuery;
    use daemon_core::MockEmbedder;
    use serde_json::json;

    #[tokio::test]
    async fn vector_recall_surfaces_semantic_match_through_provider() {
        // Pin the stored memory and the query to the same vector, and a distractor orthogonal — all
        // with content that shares NO tokens with the query, so only the vector path can match.
        // Working-memory candidates are lexically gated (`beam.py` L5313), so the pure semantic
        // match must come from the episodic tier: consolidate before recalling.
        let stored = "[USER] the deployment uses a blue-green rollout";
        let distractor = "[USER] lunch yesterday was margherita pizza";
        let query = "shipping strategy";
        let embedder = Arc::new(MockEmbedder::scripted(
            3,
            [
                (stored.to_string(), vec![1.0, 0.0, 0.0]),
                (query.to_string(), vec![1.0, 0.0, 0.0]),
                (distractor.to_string(), vec![0.0, 1.0, 0.0]),
            ],
        ));
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::with_embedder(engine.clone(), embedder);
        let conv = Conversation::new(SystemPrompt::new(""));

        provider
            .after_turn(
                &Turn::User(UserMsg::new("the deployment uses a blue-green rollout")),
                &conv,
            )
            .await;
        provider
            .after_turn(
                &Turn::User(UserMsg::new("lunch yesterday was margherita pizza")),
                &conv,
            )
            .await;
        engine.consolidate().unwrap();

        let block = provider
            .recall(&RecallQuery {
                text: query.to_string(),
                top_k: 5,
            })
            .await
            .expect("recall returns a block via the vector match");
        assert!(
            block.text.contains("blue-green"),
            "vector recall should surface the semantically-close memory; got: {}",
            block.text
        );
        assert!(
            !block.text.contains("pizza"),
            "the orthogonal distractor must not pass the vector gate; got: {}",
            block.text
        );
    }

    #[tokio::test]
    async fn session_end_promotes_working_memory_to_episodic() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine.clone());
        let conv = Conversation::new(SystemPrompt::new(""));

        provider
            .after_turn(
                &Turn::User(UserMsg::new("a memory worth keeping around")),
                &conv,
            )
            .await;
        assert!(
            !engine.recall("memory keeping", 5).unwrap().is_empty(),
            "the turn should have been stored"
        );

        // Ending the session should consolidate; a subsequent consolidate then finds nothing pending.
        provider.on_session_switch(SwitchReason::End).await;
        assert_eq!(
            engine.consolidate().unwrap(),
            0,
            "on_session_switch(End) should have already promoted the row"
        );
    }

    #[tokio::test]
    async fn after_turn_runs_llm_extraction_into_knowledge_layer() {
        use daemon_core::MockProvider;
        let json = r#"{"entities":["Atlas"],"triples":[{"subject":"Denis","predicate":"manages","object":"Atlas","confidence":0.95}],"facts":[]}"#;
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::with_backends(
            engine.clone(),
            None,
            Some(Arc::new(MockProvider::completing(json))),
        );
        let conv = Conversation::new(SystemPrompt::new(""));
        provider
            .after_turn(&Turn::User(UserMsg::new("a note about the team")), &conv)
            .await;

        // The LLM triple should have been consolidated even though the regex baseline wouldn't
        // extract "Denis manages Atlas" from that sentence.
        assert!(
            engine.stats().unwrap().facts >= 1,
            "LLM extraction should reach the knowledge layer as a consolidated fact"
        );
    }

    #[tokio::test]
    async fn tool_dispatch_remember_and_recall_round_trip() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);

        let defs = provider.tools();
        assert!(defs.iter().any(|d| d.name == "mnemosyne_remember"));
        assert!(defs.iter().any(|d| d.name == "mnemosyne_sleep"));
        assert!(defs.len() >= 25, "full tool surface, got {}", defs.len());

        let remembered = provider
            .call_tool(
                "mnemosyne_remember",
                json!({"content": "the cache uses an LRU eviction policy"}),
            )
            .await;
        assert!(
            remembered.contains("\"status\":\"stored\""),
            "got: {remembered}"
        );
        // Tool writes default to source "user" (`_handle_remember` L1838).
        let parsed: Value = serde_json::from_str(&remembered).unwrap();
        let row = provider
            .engine
            .get(parsed["memory_id"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(row.source, "user");

        let recalled = provider
            .call_tool("mnemosyne_recall", json!({"query": "eviction policy"}))
            .await;
        assert!(recalled.contains("LRU"), "recall via tool: {recalled}");
        assert!(
            recalled.contains("\"bank\":\"private\""),
            "private rows are bank-tagged: {recalled}"
        );

        let stats = provider.call_tool("mnemosyne_stats", json!({})).await;
        assert!(stats.contains("\"working\":1"), "stats: {stats}");

        let unknown = provider.call_tool("mnemosyne_nope", json!({})).await;
        assert!(unknown.contains("unknown_tool"));
    }

    /// Seed a private memory through the tool surface and stamp its `author_id` (the
    /// `_seed_private` fixture, tests/test_hermes_memory_provider_validation.py:30).
    async fn seed_with_author(provider: &MnemosyneProvider, content: &str, author: &str) -> String {
        let res = provider
            .call_tool(
                "mnemosyne_remember",
                json!({"content": content, "importance": 0.7, "source": "fact"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(parsed["status"], "stored", "seed failed: {res}");
        let mid = parsed["memory_id"].as_str().unwrap().to_string();
        provider
            .engine
            .with_conn(|c| {
                c.execute(
                    "UPDATE working_memory SET author_id = ?1 WHERE id = ?2",
                    rusqlite::params![author, mid],
                )?;
                Ok(())
            })
            .unwrap();
        mid
    }

    /// `(author_id, validator, validated_at, validation_count, valid_until, content)` for a row
    /// (the `_row` fixture, tests/test_hermes_memory_provider_validation.py:57).
    type RowFields = (
        Option<String>,
        Option<String>,
        Option<String>,
        i64,
        Option<String>,
        String,
    );
    fn row_fields(engine: &Engine, mid: &str) -> Option<RowFields> {
        use rusqlite::OptionalExtension;
        engine
            .with_conn(|c| {
                Ok(c.query_row(
                    "SELECT author_id, validator, validated_at, COALESCE(validation_count, 0), \
                     valid_until, content FROM working_memory WHERE id = ?1",
                    rusqlite::params![mid],
                    |r| {
                        Ok((
                            r.get(0)?,
                            r.get(1)?,
                            r.get(2)?,
                            r.get(3)?,
                            r.get(4)?,
                            r.get(5)?,
                        ))
                    },
                )
                .optional()?)
            })
            .unwrap()
    }

    /// `(validator, action, new_content)` rows, insertion-ordered (the `_validation_log`
    /// fixture, tests/test_hermes_memory_provider_validation.py:67).
    fn validation_log(engine: &Engine, mid: &str) -> Vec<(String, String, Option<String>)> {
        engine
            .with_conn(|c| {
                let mut stmt = c.prepare(
                    "SELECT validator, action, new_content FROM memory_validations \
                     WHERE memory_id = ?1 ORDER BY validation_id",
                )?;
                let rows = stmt.query_map(rusqlite::params![mid], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })?;
                Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
            })
            .unwrap()
    }

    // parity: test_hermes_memory_provider_validation.py::test_validate_attest_preserves_author_and_records_validator (tests/test_hermes_memory_provider_validation.py:106)
    #[tokio::test]
    async fn tool_validate_attest_records_validator_preserving_author() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);
        let mid = seed_with_author(&provider, "SSH key at /home/user/.ssh/pc", "Sisyphus").await;

        let res = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": mid, "action": "attest", "validator": "Albedo",
                       "note": "confirmed during deploy"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(parsed["status"], "validation_attest", "got: {res}");
        assert_eq!(parsed["validator"], "Albedo");
        assert_eq!(parsed["author_id"], "Sisyphus");

        let (author, validator, validated_at, count, _, _) =
            row_fields(&provider.engine, &mid).expect("row");
        assert_eq!(author.as_deref(), Some("Sisyphus"), "author preserved");
        assert_eq!(validator.as_deref(), Some("Albedo"), "validator updated");
        assert!(validated_at.is_some(), "validated_at stamped");
        assert_eq!(count, 1, "validation_count incremented");
    }

    // parity: test_hermes_memory_provider_validation.py::test_validate_attest_falls_back_to_agent_identity (tests/test_hermes_memory_provider_validation.py:129)
    #[tokio::test]
    async fn tool_validate_attest_falls_back_to_agent_identity() {
        // The engine's configured author_id is the Rust analog of the Python provider's
        // `_agent_identity` fallback.
        let engine = Arc::new(
            Engine::open_in_memory(MnemosyneConfig {
                author_id: Some("Hopz".to_string()),
                ..MnemosyneConfig::default()
            })
            .unwrap(),
        );
        let provider = MnemosyneProvider::new(engine);
        let mid = seed_with_author(&provider, "Project at /tmp/proj", "Sisyphus").await;

        let res = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": mid, "action": "attest"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(parsed["validator"], "Hopz", "got: {res}");
    }

    // parity: test_hermes_memory_provider_validation.py::test_validate_update_replaces_content_and_keeps_author (tests/test_hermes_memory_provider_validation.py:143)
    // parity: test_hermes_memory_provider_validation.py::test_validate_update_requires_new_content (tests/test_hermes_memory_provider_validation.py:161)
    #[tokio::test]
    async fn tool_validate_update_replaces_content_and_requires_new_content() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);
        let mid = seed_with_author(&provider, "SSH key at /home/user/.ssh/pc", "Sisyphus").await;

        let res = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": mid, "action": "update", "validator": "Albedo",
                       "new_content": "SSH key at /home/user/.ssh/laptop"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(parsed["status"], "validation_update", "got: {res}");
        let (author, validator, _, _, _, content) =
            row_fields(&provider.engine, &mid).expect("row");
        assert_eq!(
            content, "SSH key at /home/user/.ssh/laptop",
            "content updated"
        );
        assert_eq!(author.as_deref(), Some("Sisyphus"), "author preserved");
        assert_eq!(validator.as_deref(), Some("Albedo"), "validator updated");

        // `update` without new_content is rejected.
        let bad = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": mid, "action": "update", "validator": "Albedo"}),
            )
            .await;
        assert!(bad.contains("new_content is required"), "got: {bad}");
    }

    // parity: test_hermes_memory_provider_validation.py::test_validate_invalidate_sets_valid_until (tests/test_hermes_memory_provider_validation.py:175)
    #[tokio::test]
    async fn tool_validate_invalidate_sets_valid_until() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);
        let mid = seed_with_author(&provider, "outdated fact about VPN", "Sisyphus").await;

        let res = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": mid, "action": "invalidate", "validator": "Hopz",
                       "note": "user changed VPN"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(parsed["status"], "validation_invalidate", "got: {res}");
        let (author, validator, _, _, valid_until, _) =
            row_fields(&provider.engine, &mid).expect("row");
        assert!(valid_until.is_some(), "valid_until set");
        assert_eq!(validator.as_deref(), Some("Hopz"), "validator recorded");
        assert_eq!(author.as_deref(), Some("Sisyphus"), "author preserved");
    }

    // parity: test_hermes_memory_provider_validation.py::test_validate_delete_removes_row (tests/test_hermes_memory_provider_validation.py:195)
    #[tokio::test]
    async fn tool_validate_delete_removes_row() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);
        let mid = seed_with_author(&provider, "stale fact", "Sisyphus").await;

        let res = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": mid, "action": "delete", "validator": "Albedo"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(parsed["status"], "validation_delete", "got: {res}");
        assert!(
            row_fields(&provider.engine, &mid).is_none(),
            "row must be deleted"
        );
    }

    // parity: test_hermes_memory_provider_validation.py::test_validate_works_on_shared_surface (tests/test_hermes_memory_provider_validation.py:211)
    #[tokio::test]
    async fn tool_validate_works_on_shared_surface() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);
        let stored = provider
            .call_tool(
                "mnemosyne_shared_remember",
                json!({"content": "User prefers Tailscale over OpenVPN", "kind": "preference"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&stored).unwrap();
        assert_eq!(parsed["status"], "stored_shared", "seed failed: {stored}");
        let mid = parsed["memory_id"].as_str().unwrap().to_string();

        let res = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": mid, "action": "attest", "validator": "Albedo",
                       "bank": "surface"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(parsed["status"], "validation_attest", "got: {res}");
        assert_eq!(parsed["bank"], "surface");
        let surface = provider.surface_engine().expect("surface engine");
        let (_, validator, _, _, _, _) = row_fields(surface, &mid).expect("surface row");
        assert_eq!(validator.as_deref(), Some("Albedo"));
    }

    // parity: test_hermes_memory_provider_validation.py::test_ring_buffer_keeps_only_last_three_validations (tests/test_hermes_memory_provider_validation.py:230)
    // parity: test_hermes_memory_provider_validation.py::test_validation_count_grows_unbounded (tests/test_hermes_memory_provider_validation.py:247)
    #[tokio::test]
    async fn tool_validate_ring_buffer_keeps_last_three_while_count_grows() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);
        let mid = seed_with_author(&provider, "SSH key location", "Sisyphus").await;

        for who in ["v1", "v2", "v3", "v4", "v5", "v6"] {
            provider
                .call_tool(
                    "mnemosyne_validate",
                    json!({"memory_id": mid, "action": "attest", "validator": who}),
                )
                .await;
        }

        let log = validation_log(&provider.engine, &mid);
        let validators: Vec<&str> = log.iter().map(|(v, _, _)| v.as_str()).collect();
        assert_eq!(
            validators,
            vec!["v4", "v5", "v6"],
            "ring buffer keeps only the last 3"
        );
        let (_, _, _, count, _, _) = row_fields(&provider.engine, &mid).expect("row");
        assert_eq!(count, 6, "validation_count on the live row grows unbounded");
    }

    // parity: test_hermes_memory_provider_validation.py::test_validate_unknown_memory_returns_error (tests/test_hermes_memory_provider_validation.py:265)
    // parity: test_hermes_memory_provider_validation.py::test_validate_unknown_action_rejected (tests/test_hermes_memory_provider_validation.py:274)
    // parity: test_hermes_memory_provider_validation.py::test_validate_unknown_bank_rejected (tests/test_hermes_memory_provider_validation.py:284)
    // parity: test_hermes_memory_provider_validation.py::test_validate_missing_memory_id_rejected (tests/test_hermes_memory_provider_validation.py:295)
    #[tokio::test]
    async fn tool_validate_rejects_bad_requests() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);
        let mid = seed_with_author(&provider, "fact", "Sisyphus").await;

        let unknown_memory = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": "nonexistent", "action": "attest"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&unknown_memory).unwrap();
        assert_eq!(parsed["error"], "memory_not_found", "got: {unknown_memory}");

        let unknown_action = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": mid, "action": "frobnicate"}),
            )
            .await;
        assert!(
            unknown_action.contains("unknown action"),
            "got: {unknown_action}"
        );
        assert!(
            validation_log(&provider.engine, &mid).is_empty(),
            "a rejected action must not append a validation row"
        );

        let unknown_bank = provider
            .call_tool(
                "mnemosyne_validate",
                json!({"memory_id": mid, "action": "attest", "bank": "weird"}),
            )
            .await;
        assert!(unknown_bank.contains("unknown bank"), "got: {unknown_bank}");

        let missing_id = provider
            .call_tool("mnemosyne_validate", json!({"action": "attest"}))
            .await;
        assert!(
            missing_id.contains("memory_id is required"),
            "got: {missing_id}"
        );
    }

    // parity: test_hermes_memory_provider_validation.py::test_collaborative_attestation_chain (tests/test_hermes_memory_provider_validation.py:303)
    #[tokio::test]
    async fn tool_validate_collaborative_attestation_chain() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);
        let mid = seed_with_author(&provider, "SSH key at /home/user/.ssh/pc", "Sisyphus").await;

        for (action, validator, new_content) in [
            (
                "update",
                "Albedo",
                Some("SSH key at /home/user/.ssh/laptop"),
            ),
            (
                "update",
                "Sisyphus",
                Some("SSH key at /home/user/.ssh/main"),
            ),
            ("attest", "Hopz", None),
        ] {
            let mut args = json!({"memory_id": mid, "action": action, "validator": validator});
            if let Some(c) = new_content {
                args["new_content"] = json!(c);
            }
            provider.call_tool("mnemosyne_validate", args).await;
        }

        let (author, validator, _, count, _, content) =
            row_fields(&provider.engine, &mid).expect("row");
        assert_eq!(author.as_deref(), Some("Sisyphus"), "author preserved");
        assert_eq!(validator.as_deref(), Some("Hopz"), "latest validator");
        assert_eq!(count, 3);
        assert!(content.contains("main"), "latest content: {content}");

        let log = validation_log(&provider.engine, &mid);
        assert_eq!(
            log.iter().map(|(v, _, _)| v.as_str()).collect::<Vec<_>>(),
            vec!["Albedo", "Sisyphus", "Hopz"]
        );
        assert_eq!(
            log.iter().map(|(_, a, _)| a.as_str()).collect::<Vec<_>>(),
            vec!["update", "update", "attest"]
        );
    }

    // parity: test_e2_remember_batch_enrichment.py::test_remember_batch_writes_temporal_annotations_for_every_row (tests/test_e2_remember_batch_enrichment.py:82)
    // parity: test_e2_remember_batch_enrichment.py::test_per_row_veracity_threads_into_consolidated_facts (tests/test_e2_remember_batch_enrichment.py:151)
    // parity: test_e2_remember_batch_enrichment.py::test_remember_batch_writes_has_source_when_source_is_non_default (tests/test_e2_remember_batch_enrichment.py:101)
    #[tokio::test]
    async fn tool_remember_batch_enriches_every_row() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);

        let res = provider
            .call_tool(
                "mnemosyne_remember_batch",
                json!({"items": [
                    {"content": "Dana is a developer", "veracity": "stated"},
                    {"content": "Eric is a tester", "veracity": "inferred"},
                    {"content": "From a wiki page", "source": "wiki"},
                ]}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(parsed["status"], "stored_batch", "got: {res}");
        let ids: Vec<String> = parsed["memory_ids"]
            .as_array()
            .unwrap_or_else(|| panic!("memory_ids missing: {res}"))
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids.len(), 3, "one id per batch item");

        provider
            .engine
            .with_conn(|c| {
                // Always-on temporal enrichment: every row gets `occurred_on`.
                for id in &ids {
                    let occurred: i64 = c.query_row(
                        "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1 \
                         AND kind = 'occurred_on'",
                        rusqlite::params![id],
                        |r| r.get(0),
                    )?;
                    assert_eq!(occurred, 1, "{id}: missing occurred_on annotation");
                }
                // `has_source` only for the non-conversational source, with the row's OWN value.
                let wiki_sources: Vec<String> = {
                    let mut stmt = c.prepare(
                        "SELECT value FROM annotations WHERE memory_id = ?1 \
                         AND kind = 'has_source'",
                    )?;
                    let rows = stmt.query_map(rusqlite::params![ids[2]], |r| r.get(0))?;
                    rows.collect::<std::result::Result<Vec<_>, _>>()?
                };
                assert_eq!(wiki_sources, vec!["wiki".to_string()]);
                let convo_has_source: i64 = c.query_row(
                    "SELECT COUNT(*) FROM annotations WHERE memory_id = ?1 \
                     AND kind = 'has_source'",
                    rusqlite::params![ids[0]],
                    |r| r.get(0),
                )?;
                assert_eq!(convo_has_source, 0, "conversational rows get no has_source");
                // Per-row veracity threads into consolidated-fact confidence: stated (Dana)
                // must weigh above inferred (Eric).
                let confidence = |subject: &str| -> crate::Result<f64> {
                    Ok(c.query_row(
                        "SELECT confidence FROM consolidated_facts WHERE subject = ?1",
                        rusqlite::params![subject],
                        |r| r.get(0),
                    )?)
                };
                let dana = confidence("Dana")?;
                let eric = confidence("Eric")?;
                assert!(
                    dana > eric,
                    "stated ({dana}) must outweigh inferred ({eric}) — per-row veracity collapsed"
                );
                Ok(())
            })
            .unwrap();
    }

    // parity: test_configurable_scoring.py::TestPublicRecallConfigurableWeights::test_mnemosyne_recall_accepts_weight_params (tests/test_configurable_scoring.py:241)
    // parity: test_configurable_scoring.py::TestPublicRecallConfigurableWeights::test_module_recall_accepts_weight_params (tests/test_configurable_scoring.py:257)
    #[tokio::test]
    async fn tool_recall_forwards_weight_overrides() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);
        for (content, importance) in [
            ("critical alert generic text", 0.1),
            ("critical system status", 0.9),
        ] {
            provider
                .call_tool(
                    "mnemosyne_remember",
                    json!({"content": content, "importance": importance}),
                )
                .await;
        }
        let top = |res: &str| -> String {
            let parsed: Value = serde_json::from_str(res).unwrap();
            parsed["results"][0]["content"]
                .as_str()
                .unwrap()
                .to_string()
        };

        // Keyword-heavy overrides rank the exact lexical match first…
        let lexical = provider
            .call_tool(
                "mnemosyne_recall",
                json!({"query": "critical alert", "vec_weight": 0.5, "fts_weight": 0.45,
                       "importance_weight": 0.05}),
            )
            .await;
        assert!(top(&lexical).contains("generic"), "got: {lexical}");

        // …while importance-heavy overrides flip the ranking through the same tool wire.
        let important = provider
            .call_tool(
                "mnemosyne_recall",
                json!({"query": "critical alert", "vec_weight": 0.1, "fts_weight": 0.1,
                       "importance_weight": 0.8}),
            )
            .await;
        assert!(
            top(&important).contains("system status"),
            "got: {important}"
        );
    }

    // parity: test_e2_remember_batch_enrichment.py::test_extract_false_does_not_call_llm (tests/test_e2_remember_batch_enrichment.py:244)
    // parity: test_e2_remember_batch_enrichment.py::test_extract_true_calls_llm_fact_extractor_per_row (tests/test_e2_remember_batch_enrichment.py:258)
    #[tokio::test]
    async fn tool_remember_batch_extract_flag_gates_llm_enrichment() {
        use daemon_core::MockProvider;
        let llm_json = r#"{"entities":["Atlas"],"triples":[{"subject":"Denis","predicate":"manages","object":"Atlas","confidence":0.95}],"facts":[]}"#;
        let make = || {
            let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
            MnemosyneProvider::with_backends(
                engine,
                None,
                Some(Arc::new(MockProvider::completing(llm_json))),
            )
        };
        let llm_fact_count = |p: &MnemosyneProvider| -> i64 {
            p.engine
                .with_conn(|c| {
                    Ok(c.query_row(
                        "SELECT COUNT(*) FROM consolidated_facts \
                         WHERE subject = 'Denis' AND predicate = 'manages' AND object = 'Atlas'",
                        [],
                        |r| r.get(0),
                    )?)
                })
                .unwrap()
        };

        // Default `extract=false`: the LLM extractor must not fire.
        let provider = make();
        provider
            .call_tool(
                "mnemosyne_remember_batch",
                json!({"items": [{"content": "a note about the team"}]}),
            )
            .await;
        assert_eq!(
            llm_fact_count(&provider),
            0,
            "extract=false but LLM fact extraction fired anyway"
        );

        // `extract=true`: the LLM triple lands per row (the regex baseline can't produce it).
        let provider = make();
        let res = provider
            .call_tool(
                "mnemosyne_remember_batch",
                json!({"items": [{"content": "a note about the team"}], "extract": true}),
            )
            .await;
        assert!(res.contains("stored_batch"), "got: {res}");
        assert_eq!(
            llm_fact_count(&provider),
            1,
            "extract=true must run LLM extraction per batch row"
        );
    }

    #[tokio::test]
    async fn shared_tools_write_to_separate_surface_bank() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine);

        // Kind validation (`_handle_shared_remember` L1984).
        let bad = provider
            .call_tool(
                "mnemosyne_shared_remember",
                json!({"content": "x", "kind": "gossip"}),
            )
            .await;
        assert!(bad.contains("kind must be one of"), "got: {bad}");

        // Raw conversation content is rejected (L1979).
        let raw = provider
            .call_tool(
                "mnemosyne_shared_remember",
                json!({"content": "[USER] hi there"}),
            )
            .await;
        assert!(raw.contains("raw conversation content"), "got: {raw}");

        let stored = provider
            .call_tool(
                "mnemosyne_shared_remember",
                json!({"content": "the user prefers dark mode", "kind": "preference"}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&stored).unwrap();
        assert_eq!(parsed["status"], "stored_shared", "got: {stored}");
        let sid = parsed["memory_id"].as_str().unwrap();
        assert!(sid.starts_with("sf_"), "stable surface id, got {sid}");

        // Exact repeat hits the dedup path (`_find_duplicate`) -> reported as existing.
        let again = provider
            .call_tool(
                "mnemosyne_shared_remember",
                json!({"content": "the user prefers dark mode", "kind": "preference"}),
            )
            .await;
        assert!(again.contains("existing_shared"), "got: {again}");

        // The private bank saw none of it.
        assert_eq!(provider.engine.stats().unwrap().working, 0);

        // Surface recall finds it, tagged with its bank.
        let recalled = provider
            .call_tool("mnemosyne_shared_recall", json!({"query": "dark mode"}))
            .await;
        assert!(recalled.contains("Surface preference"), "got: {recalled}");
        assert!(recalled.contains("\"bank\":\"surface\""), "got: {recalled}");

        // Forget via the Python arg name (`memory_id`).
        let forgotten = provider
            .call_tool("mnemosyne_shared_forget", json!({"memory_id": sid}))
            .await;
        assert!(
            forgotten.contains("\"status\":\"deleted\""),
            "got: {forgotten}"
        );

        // Tool-level audit rows landed in the PRIVATE bank's audit_log with bank="surface".
        let audits = provider
            .engine
            .audit_rows_for_test("surface")
            .expect("audit query");
        assert!(
            audits.iter().any(|a| a == "shared_remember")
                && audits.iter().any(|a| a == "shared_forget"),
            "audit actions: {audits:?}"
        );
    }

    // ---- Unified private + surface recall (`tests/test_hermes_memory_provider_unified_recall.py`) ----

    fn surface_read_provider(shared_surface_read: bool) -> MnemosyneProvider {
        let engine = Arc::new(
            Engine::open_in_memory(MnemosyneConfig {
                shared_surface_read,
                ..MnemosyneConfig::default()
            })
            .unwrap(),
        );
        MnemosyneProvider::new(engine)
    }

    async fn seed_private_tool(provider: &MnemosyneProvider, content: &str) {
        let res = provider
            .call_tool(
                "mnemosyne_remember",
                json!({"content": content, "importance": 0.6, "source": "fact", "scope": "global"}),
            )
            .await;
        assert!(res.contains("\"status\":\"stored\""), "seed private: {res}");
    }

    async fn seed_surface_tool(provider: &MnemosyneProvider, content: &str) {
        let res = provider
            .call_tool(
                "mnemosyne_shared_remember",
                json!({"content": content, "kind": "preference", "importance": 0.8}),
            )
            .await;
        assert!(
            res.contains("\"status\":\"stored_shared\""),
            "seed surface: {res}"
        );
    }

    // PARITY: Mnemosyne tests/test_hermes_memory_provider_unified_recall.py::test_recall_default_returns_private_only
    // PARITY: Mnemosyne tests/test_hermes_memory_provider_unified_recall.py::test_recall_default_tags_results_as_private
    #[tokio::test]
    async fn recall_default_returns_private_only_and_tags_private() {
        let provider = surface_read_provider(false);
        seed_private_tool(&provider, "Project root lives at /tmp/project directory").await;
        seed_surface_tool(&provider, "User prefers Tailscale over OpenVPN").await;

        let res = provider
            .call_tool(
                "mnemosyne_recall",
                json!({"query": "Tailscale", "limit": 10}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(
            parsed["shared_surface_read"], false,
            "surface read off: {res}"
        );
        for r in parsed["results"].as_array().unwrap() {
            assert!(
                !r["content"].as_str().unwrap().contains("Tailscale"),
                "surface content must not leak when read is off: {res}"
            );
        }

        // A query that DOES match the private row is tagged bank=private.
        let res2 = provider
            .call_tool(
                "mnemosyne_recall",
                json!({"query": "project root", "limit": 5}),
            )
            .await;
        let parsed2: Value = serde_json::from_str(&res2).unwrap();
        let rows = parsed2["results"].as_array().unwrap();
        assert!(!rows.is_empty(), "private row must surface: {res2}");
        for r in rows {
            assert_eq!(r["bank"], "private", "private tag: {res2}");
        }
    }

    // PARITY: Mnemosyne tests/test_hermes_memory_provider_unified_recall.py::test_recall_merges_results_from_both_banks
    // PARITY: Mnemosyne tests/test_hermes_memory_provider_unified_recall.py::test_recall_tags_surface_results_with_bank_surface
    #[tokio::test]
    async fn recall_with_surface_read_merges_both_banks() {
        let provider = surface_read_provider(true);
        seed_private_tool(&provider, "User project Acme uses Postgres on port 5432").await;
        seed_surface_tool(&provider, "User prefers Postgres for production databases").await;

        let res = provider
            .call_tool(
                "mnemosyne_recall",
                json!({"query": "Postgres", "limit": 10}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(
            parsed["shared_surface_read"], true,
            "surface read on: {res}"
        );
        let banks: std::collections::HashSet<&str> = parsed["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["bank"].as_str().unwrap())
            .collect();
        assert!(banks.contains("private"), "private bank present: {res}");
        assert!(banks.contains("surface"), "surface bank merged: {res}");
        // Surface rows carry the shared_surface flag.
        for r in parsed["results"].as_array().unwrap() {
            if r["bank"] == "surface" {
                assert_eq!(r["shared_surface"], true, "surface flag: {res}");
            }
        }
    }

    // PARITY: Mnemosyne tests/test_hermes_memory_provider_thread_isolation.py::test_gateway_session_key_isolates_session_memories
    // Two providers over the SAME on-disk bank but distinct session ids (the gateway thread key is
    // the Rust session_id) must not surface each other's session-scoped rows through the recall
    // tool, while scope='global' rows cross both.
    #[tokio::test]
    async fn recall_isolates_session_memories_across_providers() {
        let dir = std::env::temp_dir().join(format!("mnemosyne-prov-iso-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let make = |session_id: &str| {
            let engine = Arc::new(
                Engine::open(MnemosyneConfig {
                    data_dir: dir.clone(),
                    session_id: session_id.to_string(),
                    ..MnemosyneConfig::default()
                })
                .unwrap(),
            );
            MnemosyneProvider::new(engine)
        };
        let prov_a = make("hermes_agent:main:telegram:dm:12345:11111");
        let prov_b = make("hermes_agent:main:telegram:dm:12345:22222");

        async fn remember(p: &MnemosyneProvider, content: &str, scope: &str) {
            let res = p
                .call_tool(
                    "mnemosyne_remember",
                    json!({"content": content, "scope": scope}),
                )
                .await;
            assert!(res.contains("\"status\":\"stored\""), "remember: {res}");
        }
        async fn recall_contents(p: &MnemosyneProvider, query: &str) -> Vec<String> {
            let res = p
                .call_tool("mnemosyne_recall", json!({"query": query}))
                .await;
            let parsed: Value = serde_json::from_str(&res).unwrap();
            parsed["results"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r["content"].as_str().unwrap().to_string())
                .collect()
        }

        remember(&prov_a, "Secret A the sky is green", "session").await;
        remember(&prov_b, "Secret B the ocean is purple", "session").await;
        remember(&prov_a, "Global water is wet", "global").await;

        let a_secret = recall_contents(&prov_a, "secret").await;
        assert!(
            a_secret.iter().any(|c| c.contains("Secret A")),
            "A sees its own session row: {a_secret:?}"
        );
        assert!(
            !a_secret.iter().any(|c| c.contains("Secret B")),
            "A must NOT see B's session row: {a_secret:?}"
        );

        // The global row crosses both threads.
        assert!(
            recall_contents(&prov_a, "global water")
                .await
                .iter()
                .any(|c| c.contains("Global water is wet")),
            "A sees the global row"
        );
        assert!(
            recall_contents(&prov_b, "global water")
                .await
                .iter()
                .any(|c| c.contains("Global water is wet")),
            "B sees the global row"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // PARITY: Mnemosyne tests/test_hermes_memory_provider_unified_recall.py::test_recall_truncates_to_top_k_after_merge
    #[tokio::test]
    async fn recall_truncates_to_top_k_after_merge() {
        let provider = surface_read_provider(true);
        for i in 0..5 {
            seed_private_tool(
                &provider,
                &format!("User runs script number {i} for migration tasks"),
            )
            .await;
        }
        for i in 0..5 {
            seed_surface_tool(
                &provider,
                &format!("User prefers tool variant {i} for migration tasks"),
            )
            .await;
        }

        let res = provider
            .call_tool(
                "mnemosyne_recall",
                json!({"query": "migration tasks", "limit": 4}),
            )
            .await;
        let parsed: Value = serde_json::from_str(&res).unwrap();
        assert!(
            parsed["count"].as_u64().unwrap() <= 4,
            "merged results truncated to top-k: {res}"
        );
        assert!(
            parsed["results"].as_array().unwrap().len() <= 4,
            "len<=4: {res}"
        );
    }
}
