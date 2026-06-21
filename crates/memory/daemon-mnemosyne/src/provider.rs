//! The `MemoryProvider` implementation — port of `hermes_memory_provider/__init__.py`.
//!
//! Maps Mnemosyne onto the daemon-core seam: `prompt_block` = memory-override instructions
//! (`system_prompt_block` L1437), `recall` = formatted BEAM recall block (`prefetch` L1474 / block
//! format L1645-L1659), `after_turn` = the `sync_turn` persist gates (L1668), and `tools`/`call_tool`
//! = the JSON tool dispatch (L1750). Scaffold: the core hooks are wired; the full 26-tool table and
//! identity-signal capture are TODO.

use crate::embeddings::Embedder;
use crate::engine::{Engine, MemoryRow, RememberArgs};
use crate::extract::Extractor;
use crate::MnemosyneConfig;
use daemon_core::conversation::{Conversation, Turn};
use daemon_core::memory::{MemoryProvider, PromptBlock, RecallQuery, RecalledBlock, SwitchReason};
use daemon_core::tools::ToolDef;
use daemon_core::{EmbeddingProvider, Provider};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Auto-sleep cadence: run a consolidation pass every N persisted turns (`__init__.py` auto-sleep
/// every 10 turns, L1690).
const AUTO_SLEEP_EVERY_TURNS: u64 = 10;

/// The Mnemosyne memory provider over a single bank engine, with optional embedding + LLM backends.
pub struct MnemosyneProvider {
    engine: Arc<Engine>,
    embedder: Embedder,
    extractor: Extractor,
    turns: AtomicU64,
}

impl MnemosyneProvider {
    /// Wrap an existing engine in keyword-only mode (no embeddings, no LLM).
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            embedder: Embedder::new(),
            extractor: Extractor::new(),
            turns: AtomicU64::new(0),
        }
    }

    /// Wrap an existing engine with an injected embedding provider (hybrid lexical + vector recall).
    pub fn with_embedder(engine: Arc<Engine>, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            engine,
            embedder: Embedder::with_provider(embedder),
            extractor: Extractor::new(),
            turns: AtomicU64::new(0),
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

    /// Format recall rows into a prompt block (`__init__.py` L1645-L1659):
    /// `  [ts] (importance X.XX[, source S])[ TRUST] content`.
    fn format_block(rows: &[MemoryRow]) -> String {
        let mut out = String::from("## Mnemosyne Context\n");
        for row in rows {
            let ts = row.timestamp.chars().take(16).collect::<String>();
            let mut meta = format!("importance {:.2}", row.importance);
            if row.source != "conversation" && !row.source.is_empty() {
                meta.push_str(&format!(", source {}", row.source));
            }
            let trust = if row.trust_tier != "STATED" && !row.trust_tier.is_empty() {
                format!(" [{}]", row.trust_tier)
            } else {
                String::new()
            };
            out.push_str(&format!("  [{}] ({}){} {}\n", ts, meta, trust, row.content));
        }
        out
    }
}

#[async_trait::async_trait]
impl MemoryProvider for MnemosyneProvider {
    fn name(&self) -> &str {
        "mnemosyne"
    }

    fn prompt_block(&self) -> Option<PromptBlock> {
        Some(PromptBlock {
            text: "You have a persistent memory (Mnemosyne). Recalled context is injected below; \
                   use the mnemosyne_* tools to remember, recall, and manage long-term memory."
                .to_string(),
        })
    }

    async fn recall(&self, q: &RecallQuery) -> Option<RecalledBlock> {
        // Embed the query at this async seam (keyword-only -> None) and pass the vector into the
        // synchronous hybrid recall so the engine never blocks on a model call.
        let query_vec = self.embedder.embed_query(&q.text).await;
        let rows = self
            .engine
            .recall_with_vector(&q.text, q.top_k, query_vec.as_deref())
            .ok()?;
        if rows.is_empty() {
            return None;
        }
        Some(RecalledBlock {
            text: Self::format_block(&rows),
        })
    }

    async fn after_turn(&self, turn: &Turn, _conv: &Conversation) {
        // sync_turn gates (`__init__.py` L1668-L1692): persist the user text (>5 chars) and the
        // assistant text (>10 chars) with their respective importances.
        let (content, importance) = match turn {
            Turn::User(u) if u.text.len() > 5 => (format!("[USER] {}", u.text), 0.5),
            Turn::Assistant(a) if a.text.len() > 10 => (format!("[ASSISTANT] {}", a.text), 0.15),
            _ => return,
        };
        // Embed once at this async seam; the precomputed vector is persisted with the row.
        let vector = self.embedder.embed_query(&content).await;
        let model = self.embedder.model().unwrap_or("");
        let memory_id = match self.engine.remember_with_vector(
            &content,
            &RememberArgs {
                importance,
                ..Default::default()
            },
            vector.as_deref(),
            model,
        ) {
            Ok(id) => id,
            Err(_) => return,
        };

        // LLM extraction layered on top of the always-on regex baseline (`extraction.py`): extract
        // at this async seam, then merge into the knowledge layer synchronously.
        if self.extractor.available() {
            if let Some(extracted) = self.extractor.extract(&content).await {
                let _ = self.engine.ingest_extracted(&memory_id, &extracted);
            }
        }

        // Turn-counter auto-sleep (`__init__.py` L1690): every N persisted turns, run a
        // consolidation pass (summarizing through the LLM when present).
        let turns = self.turns.fetch_add(1, Ordering::Relaxed) + 1;
        if turns % AUTO_SLEEP_EVERY_TURNS == 0 {
            self.run_sleep(false).await;
        }
    }

    async fn before_compact(&self, _conv: &Conversation) {
        // TODO: persist salient facts before the body is compacted (`on_pre_compress`).
    }

    async fn on_session_switch(&self, reason: SwitchReason) {
        // Run a full, forced sleep pass at session boundaries (`beam.py` sleep L7576): flush this
        // session's working memory into the episodic tiers regardless of age, then degrade.
        if matches!(reason, SwitchReason::End | SwitchReason::Handoff) {
            self.run_sleep(true).await;
        }
    }
}

impl MnemosyneProvider {
    /// Drive one sleep/consolidation pass (`beam.py` sleep L7576). When an LLM is present, each
    /// claimed source group is summarized at this async seam before the synchronous engine writes
    /// the episodic summary; otherwise the engine falls back to the deterministic AAAK summary.
    async fn run_sleep(&self, force: bool) {
        let _ = crate::tools::run_sleep(&self.engine, &self.extractor, force).await;
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
    pub async fn call_tool(&self, name: &str, args: Value) -> String {
        crate::tools::dispatch(&self.engine, &self.embedder, &self.extractor, name, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use serde_json::json;
    use daemon_core::conversation::{Conversation, SystemPrompt, Turn, UserMsg};
    use daemon_core::memory::RecallQuery;
    use daemon_core::MockEmbedder;

    #[tokio::test]
    async fn vector_recall_surfaces_semantic_match_through_provider() {
        // Pin the stored memory and the query to the same vector, and a distractor orthogonal — all
        // with content that shares NO tokens with the query, so only the vector path can match.
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
        let provider = MnemosyneProvider::with_embedder(engine, embedder);
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
        assert!(remembered.contains("\"status\":\"ok\""), "got: {remembered}");

        let recalled = provider
            .call_tool("mnemosyne_recall", json!({"query": "eviction policy"}))
            .await;
        assert!(recalled.contains("LRU"), "recall via tool: {recalled}");

        let stats = provider.call_tool("mnemosyne_stats", json!({})).await;
        assert!(stats.contains("\"working\":1"), "stats: {stats}");

        let unknown = provider.call_tool("mnemosyne_nope", json!({})).await;
        assert!(unknown.contains("unknown_tool"));
    }
}
